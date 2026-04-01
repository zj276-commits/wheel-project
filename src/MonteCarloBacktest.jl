# MonteCarloBacktest.jl — PDF §7B: Portfolio-Level Stress Tests + Monte Carlo Robust Simulation
#
# Extracts the MC and stress-test orchestration from Backtest.jl into
# reusable functions. All simulation primitives live in Simulation.jl;
# this file only handles the backtest-loop and reporting layer.

"""
    run_portfolio_stress_tests(...)

Part 6: Run the full Wheel backtest under each extended stress scenario.
Reports final NAV, return, Sharpe, MaxDD, and premium for each scenario.
Also runs single-ticker stress + earnings-jump GBM for representative tickers.
"""
function run_portfolio_stress_tests(;
        price_data, div_data, vol_map, vix_data,
        all_tickers, sleeves, weights, sleeves_map,
        initial_nav, prices_day1, config,
        earnings_cal, sector_map,
        safe_tickers, aggressive_tickers,
        chart_defaults,
        heston_ts::Dict{String, Dict{Date, HestonCalibration}}=Dict{String, Dict{Date, HestonCalibration}}())

    println("\n--- Part 6: Portfolio-Level Stress Tests (PDF §7B) ---\n")

    stress_results = DataFrame(
        Scenario=String[], FinalNAV=Float64[], Return=Float64[],
        Sharpe=Float64[], MaxDD=Float64[], Premium=Float64[]
    )

    for sc in EXTENDED_STRESS_SCENARIOS
        stressed_prices = apply_stress_to_prices(price_data;
            vol_mult=sc.vol_mult, drift_adj=sc.drift_adj,
            gap_pct=sc.gap_pct, gap_day=sc.gap_day,
            spread_widening=sc.spread_w,
            liquidity_thin_pct=sc.liq_thin)

        stressed_rolling = compute_rolling_volatility(stressed_prices; window=30)

        pf = initialize_portfolio(all_tickers, sleeves, weights, initial_nav, prices_day1, config)
        stressed_days = get_trading_days(stressed_prices)

        stressed_iv = build_heston_iv_map(stressed_prices, stressed_days;
                                           r=0.045, heston_ts=heston_ts, vix_data=vix_data)
        run_backtest!(pf, stressed_prices, div_data, vol_map, stressed_days;
                      earnings_cal=earnings_cal, rolling_vol=stressed_rolling,
                      sector_map=sector_map,
                      rolling_iv=stressed_iv,
                      heston_ts=heston_ts)

        recs = pf.daily_records
        isempty(recs) && continue
        fin = recs[end].nav
        ret = (fin - initial_nav) / initial_nav * 100.0
        navs = [r.nav for r in recs]
        local dr = if all(x -> x > 0.0, navs)
            diff(log.(navs))
        else
            diff(navs) ./ max.(abs.(navs[1:end-1]), 1.0)
        end
        sh = length(dr) > 0 && std(dr) > 0 ? (mean(dr)*252) / (std(dr)*sqrt(252)) : 0.0
        pk, mdd = 0.0, 0.0
        for v in navs; pk = max(pk, v); mdd = pk > 0.0 ? max(mdd, (pk-v)/pk) : mdd; end
        tp = recs[end].cumulative_premium

        push!(stress_results, (sc.label, round(fin/1e6, digits=2), round(ret, digits=2),
              round(sh, digits=3), round(mdd*100, digits=2), round(tp/1e6, digits=2)))
    end

    pretty_table(stress_results,        column_labels=["Scenario", "Final NAV (M\$)", "Return %",
                        "Sharpe", "MaxDD %", "Premium (M\$)"])

    stress_df = DataFrame(stress_results)
    rename!(stress_df, [:Scenario, :FinalNAV_M, :Return_pct, :Sharpe, :MaxDD_pct, :Premium_M])
    yr = string(year(trading_days[1]))
    CSV.write(joinpath(_PATH_TO_DATA, "stress_test_$(yr).csv"), stress_df)
    println("  -> Stress test results saved to data/stress_test_$(yr).csv")

    println("\n-- Single-Ticker Stress (incl. Earnings Jump GBM) --")
    for (label, ticker) in [("Safe", safe_tickers[1]), ("Aggressive", aggressive_tickers[1])]
        haskey(prices_day1, ticker) || continue
        S0 = prices_day1[ticker]
        mu = vol_map[ticker] * 0.5
        sig = vol_map[ticker]

        println("\n  $label representative: $ticker (S0=\$$(round(S0, digits=2)), sig=$(round(sig, digits=2)))")

        results = run_stress_scenarios(S0, mu, sig, 1.0; n_paths=5000)
        pretty_table(results,            column_labels=["Scenario", "Mean Ret%", "Median Ret%",
                            "VaR 95%", "Avg MaxDD%", "% Below -20%"])

        earnings_steps = [63, 126, 189, 252]
        ejump_paths = simulate_earnings_jump_gbm(S0, mu, sig, 1.0, earnings_steps;
            jump_mean=0.0, jump_std=0.07, vol_crush=0.60, n_paths=5000)
        ej_summary = mc_summary(ejump_paths)
        println("  Earnings-Jump GBM (4 events, jump_std=7%, vol_crush=60%):")
        println("    Mean return: $(round(ej_summary.mean_return*100, digits=2))%")
        println("    VaR 95:      $(round(ej_summary.var_95*100, digits=2))%")
        println("    Std return:  $(round(ej_summary.std_return*100, digits=2))%")
    end
end


"""
    run_robust_mc_simulation(...)

Part 7: Monte Carlo robust simulation.
- 7a: Calibrate HMM regimes per ticker
- 7b: Run N synthetic backtests via Heston-aware regime-switching GBM
- 7c: Report distribution of outcomes
- 7d: Fan chart
- 7e: Single-ticker Heston regime + earnings jump analysis
"""
function run_robust_mc_simulation(;
        price_data, div_data, vol_map, vix_data,
        all_tickers, sleeves, weights, sleeves_map,
        initial_nav, prices_day1, config,
        earnings_cal, sector_map,
        trading_days, daily_df,
        safe_tickers, aggressive_tickers,
        chart_defaults, yr::String,
        heston_ts::Dict{String, Dict{Date, HestonCalibration}}=Dict{String, Dict{Date, HestonCalibration}}(),
        n_mc_runs::Int=20)

    println("\n--- Part 7: Monte Carlo Robust Simulation (PDF §7B) ---\n")

    # ── 7a: Calibrate Jump-HMM regimes per ticker ─────────────────────────────
    println("  7a: Fitting Jump-HMM from historical prices...")
    hmm_models = fit_all_hmm(price_data, all_tickers; min_obs=60)

    regime_params_map = Dict{String, NamedTuple}()
    regime_df = DataFrame(
        Ticker=String[], N=Int[], LL=Float64[],
        mu_N=Float64[], sig_N=Float64[],
        mu_S=Float64[], sig_S=Float64[],
        P_toS=Float64[], P_toN=Float64[]
    )
    for tk in sort(collect(keys(hmm_models)))
        prices_tk = Float64.(price_data[tk].adj_close)
        rp = extract_regime_params(hmm_models[tk]; prices=prices_tk)
        regime_params_map[tk] = rp
        push!(regime_df, (
            tk, rp.N, round(rp.ll, digits=1),
            round(rp.mu_normal * 252 * 100, digits=2),
            round(rp.sig_normal * sqrt(252) * 100, digits=1),
            round(rp.mu_stressed * 252 * 100, digits=2),
            round(rp.sig_stressed * sqrt(252) * 100, digits=1),
            round(rp.p_to_stressed * 100, digits=2),
            round(rp.p_to_normal * 100, digits=2)
        ))
    end
    println("  JumpHMM fitted for $(length(hmm_models))/$(length(all_tickers)) tickers\n")
    pretty_table(regime_df,        column_labels=["Ticker","N","LL","μ_N(%ann)","σ_N(%ann)",
                        "μ_S(%ann)","σ_S(%ann)","P(→S)%","P(→N)%"])

    # ── 7b: Monte Carlo loop ─────────────────────────────────────────────────
    println("\n  7b: Running $n_mc_runs Monte Carlo simulations (HMM + Heston GBM)...\n")

    mc_results = DataFrame(
        Run=Int[], FinalNAV_M=Float64[], Return=Float64[],
        Sharpe=Float64[], Sortino=Float64[], MaxDD=Float64[],
        Premium_M=Float64[], Assigns=Int[], Trades=Int[]
    )
    n_days = length(trading_days)
    mc_nav_paths = Matrix{Float64}(undef, n_days, n_mc_runs)

    for run_idx in 1:n_mc_runs
        synth_prices = Dict{String, DataFrame}()

        for tk in all_tickers
            S0 = get(prices_day1, tk, NaN)
            isnan(S0) && continue

            sleeve = get(sleeves_map, tk, "Safe")
            rv = vol_map[tk]

            hp_lookup = lookup_heston_params(heston_ts, tk, trading_days[1])
            if haskey(regime_params_map, tk) && hp_lookup !== nothing
                rp = regime_params_map[tk]
                paths = simulate_heston_regime_gbm(S0, rp, hp_lookup, n_days / 252.0;
                                                    Δt=1.0/252.0, n_paths=1)
            else
                paths = simulate_gbm(S0, 0.05, rv, n_days / 252.0; n_paths=1)
            end

            path_len = min(size(paths, 1), n_days)
            pp = paths[1:path_len, 1]

            synth_prices[tk] = DataFrame(
                date      = trading_days[1:path_len],
                open      = pp,
                high      = pp .* (1.0 .+ abs.(randn(path_len)) .* 0.005),
                low       = pp .* (1.0 .- abs.(randn(path_len)) .* 0.005),
                close     = pp,
                adj_close = pp,
                volume    = fill(1_000_000.0, path_len)
            )
        end

        synth_rolling_vol = compute_rolling_volatility(synth_prices; window=30)
        synth_trading_days = get_trading_days(synth_prices)
        isempty(synth_trading_days) && continue

        synth_iv = build_heston_iv_map(synth_prices, synth_trading_days;
                                        r=0.045, heston_ts=heston_ts, vix_data=vix_data)

        synth_p1 = Dict(tk => df.adj_close[1]
                        for (tk, df) in synth_prices if nrow(df) > 0)
        valid_tks = [t for t in all_tickers if haskey(synth_p1, t)]
        v_mask = [t in valid_tks for t in all_tickers]
        mc_w = weights[v_mask]
        sum(mc_w) > 0 && (mc_w ./= sum(mc_w))
        mc_sl = sleeves[v_mask]

        pf = initialize_portfolio(valid_tks, mc_sl, mc_w, initial_nav, synth_p1, config)
        run_backtest!(pf, synth_prices, div_data, vol_map, synth_trading_days;
                      earnings_cal=earnings_cal, rolling_vol=synth_rolling_vol,
                      sector_map=sector_map,
                      rolling_iv=synth_iv,
                      heston_ts=heston_ts)

        recs = pf.daily_records
        isempty(recs) && continue
        fin = recs[end].nav
        ret = (fin - initial_nav) / initial_nav * 100.0
        navs_mc = [r.nav for r in recs]

        for i in 1:min(length(navs_mc), n_days)
            mc_nav_paths[i, run_idx] = navs_mc[i]
        end
        if length(navs_mc) < n_days
            mc_nav_paths[(length(navs_mc)+1):n_days, run_idx] .= navs_mc[end]
        end

        dr_mc = if all(x -> x > 0.0, navs_mc)
            diff(log.(navs_mc))
        else
            diff(navs_mc) ./ max.(abs.(navs_mc[1:end-1]), 1.0)
        end
        sh = length(dr_mc) > 0 && std(dr_mc) > 0 ? (mean(dr_mc)*252) / (std(dr_mc)*sqrt(252)) : 0.0
        ds_mc = filter(x -> x < 0, dr_mc)
        so = length(ds_mc) > 0 && std(ds_mc) > 0 ? (mean(dr_mc)*252) / (std(ds_mc)*sqrt(252)) : 0.0
        pk, mdd = 0.0, 0.0
        for v in navs_mc; pk = max(pk, v); mdd = pk > 0.0 ? max(mdd, (pk-v)/pk) : mdd; end
        tp = recs[end].cumulative_premium
        ta = sum(sum(s.assignment_count for s in st.slots) for (_,st) in pf.states)
        tt = sum(sum(s.trades for s in st.slots) for (_,st) in pf.states)

        push!(mc_results, (run_idx, round(fin/1e6, digits=2), round(ret, digits=2),
              round(sh, digits=3), round(so, digits=3), round(mdd*100, digits=2),
              round(tp/1e6, digits=2), ta, tt))

        println("    Run $(lpad(run_idx,2))/$n_mc_runs: Return=$(lpad(string(round(ret,digits=1)),6))%  Sharpe=$(lpad(string(round(sh,digits=2)),5))  MaxDD=$(lpad(string(round(mdd*100,digits=1)),5))%")
    end

    # ── 7c: Results distribution ──────────────────────────────────────────────
    println("\n-- Monte Carlo Run Results --")
    pretty_table(mc_results,        column_labels=["Run","Final NAV(M\$)","Return(%)","Sharpe","Sortino",
                        "MaxDD(%)","Premium(M\$)","Assigns","Trades"])

    println("\n-- Monte Carlo Summary Statistics --")
    mc_summary_df = DataFrame(Statistic=String[],
        Return=String[], Sharpe=String[], MaxDD=String[], Premium=String[])
    for (label, fn) in [("Mean", mean), ("Median", median), ("Std Dev", std),
                         ("Min", minimum), ("Max", maximum)]
        push!(mc_summary_df, (label,
            string(round(fn(mc_results.Return), digits=2)),
            string(round(fn(mc_results.Sharpe), digits=3)),
            string(round(fn(mc_results.MaxDD), digits=2)),
            string(round(fn(mc_results.Premium_M), digits=2))))
    end
    sorted_ret = sort(mc_results.Return)
    p5  = sorted_ret[max(1, ceil(Int, 0.05 * length(sorted_ret)))]
    p95 = sorted_ret[min(length(sorted_ret), floor(Int, 0.95 * length(sorted_ret)))]
    push!(mc_summary_df, ("5th Pctl",  string(round(p5, digits=2)), "-", "-", "-"))
    push!(mc_summary_df, ("95th Pctl", string(round(p95, digits=2)), "-", "-", "-"))
    pretty_table(mc_summary_df,        column_labels=["Statistic","Return(%)","Sharpe","MaxDD(%)","Premium(M\$)"])

    # ── 7d: Fan chart ─────────────────────────────────────────────────────────
    if n_days > 5
        p_mc = plot(trading_days, mc_nav_paths[:, 1] ./ 1e6;
            chart_defaults...,
            title = "Figure MC: NAV Fan Chart — $n_mc_runs Monte Carlo Runs ($(yr))",
            xlabel = "Date", ylabel = "NAV (\$ millions)",
            label = nothing, linewidth = 0.5, color = RGB(0.2, 0.4, 0.7), alpha = 0.25)
        for i in 2:n_mc_runs
            plot!(trading_days, mc_nav_paths[:, i] ./ 1e6,
                  label = nothing, linewidth = 0.5, color = RGB(0.2, 0.4, 0.7), alpha = 0.25)
        end
        mc_med = [median(mc_nav_paths[t, :]) for t in 1:n_days]
        mc_p10 = [sort(mc_nav_paths[t, :])[max(1, ceil(Int, 0.1*n_mc_runs))] for t in 1:n_days]
        mc_p90 = [sort(mc_nav_paths[t, :])[min(n_mc_runs, floor(Int, 0.9*n_mc_runs))] for t in 1:n_days]
        plot!(trading_days, mc_med  ./ 1e6, label = "Median",    linewidth = 2.5, color = :black)
        plot!(trading_days, mc_p10 ./ 1e6, label = "10th pctl", linewidth = 1.5, color = RGB(0.8, 0.15, 0.15), linestyle = :dash)
        plot!(trading_days, mc_p90 ./ 1e6, label = "90th pctl", linewidth = 1.5, color = RGB(0.2, 0.6, 0.3), linestyle = :dash)
        plot!(daily_df.Date, daily_df.NAV ./ 1e6,
              label = "Historical Backtest", linewidth = 2, color = RGB(0.85, 0.45, 0.15))
        hline!([initial_nav / 1e6], linestyle = :dot, color = :gray60, label = "Initial NAV", linewidth = 0.8)
        savefig(p_mc, joinpath(_PATH_TO_DATA, "mc_fan_chart_$(yr).png"))
        display(p_mc)
        println("  -> MC fan chart saved to data/mc_fan_chart_$(yr).png")
    end

    CSV.write(joinpath(_PATH_TO_DATA, "mc_results_$(yr).csv"), mc_results)
    println("  -> MC results saved to data/mc_results_$(yr).csv")

    # ── 7e: Single-ticker Heston regime simulation with earnings jumps ────────
    println("\n-- Single-Ticker Robust Simulation (Earnings Jump + Heston Regime) --")
    for (label, ticker) in [("Safe", safe_tickers[1]), ("Aggressive", aggressive_tickers[1])]
        haskey(prices_day1, ticker) || continue
        S0 = prices_day1[ticker]
        sig = vol_map[ticker]

        println("\n  $label representative: $ticker (S0=\$$(round(S0, digits=2)), σ=$(round(sig, digits=3)))")

        sc_results = run_stress_scenarios(S0, 0.05, sig, 1.0; n_paths=5000)
        pretty_table(sc_results,            column_labels=["Scenario","Mean Ret%","Median Ret%","VaR 95%","Avg MaxDD%","% Below -20%"])

        earnings_steps = [63, 126, 189, 252]
        ej_paths = simulate_earnings_jump_gbm(S0, 0.05, sig, 1.0, earnings_steps;
            jump_mean=0.0, jump_std=0.07, vol_crush=0.60, n_paths=5000)
        ej = mc_summary(ej_paths)
        println("  Earnings-Jump GBM (4 events, jump_std=7%, vol_crush=60%):")
        println("    Mean return:   $(round(ej.mean_return*100, digits=2))%")
        println("    Median return: $(round(ej.median_return*100, digits=2))%")
        println("    VaR 95:        $(round(ej.var_95*100, digits=2))%")
        println("    Std return:    $(round(ej.std_return*100, digits=2))%")

        hp_single = lookup_heston_params(heston_ts, ticker, trading_days[1])
        if haskey(regime_params_map, ticker) && hp_single !== nothing
            rp = regime_params_map[ticker]
            hrg_paths = simulate_heston_regime_gbm(S0, rp, hp_single, 1.0; n_paths=5000)
            hrg = mc_summary(hrg_paths)
            println("  Heston Regime GBM (HMM-calibrated):")
            println("    Mean return:   $(round(hrg.mean_return*100, digits=2))%")
            println("    Median return: $(round(hrg.median_return*100, digits=2))%")
            println("    VaR 95:        $(round(hrg.var_95*100, digits=2))%")
            println("    Std return:    $(round(hrg.std_return*100, digits=2))%")
        end
    end
end
