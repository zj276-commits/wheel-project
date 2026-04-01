# PaperPortfolio.jl — PDF §7C: Paper/Shadow Portfolio Validation
#
# Run a small-scale portfolio on recent real data and compare
# micro-level metrics against the full backtest to validate
# fill, cost, and timing assumptions.

"""
    run_paper_portfolio_validation(...)

Part 8: Paper/shadow portfolio validation (PDF §7C).
- 8a: Initialize small portfolio on last ~63 trading days
- 8b-c: Compute paper and full-backtest metrics
- 8d: Micro-metrics comparison table
- 8e: Per-ticker detail
- 8f: Validation checklist
- 8g: NAV chart
"""
function run_paper_portfolio_validation(;
        price_data, div_data, vol_map,
        all_tickers, sleeves, weights,
        initial_nav, config,
        earnings_cal, sector_map, div_yields,
        rolling_vol, rolling_iv,
        trading_days, daily_df,
        portfolio,
        chart_defaults, yr::String,
        heston_ts::Dict{String, Dict{Date, HestonCalibration}}=Dict{String, Dict{Date, HestonCalibration}}(),
        vix_regime::Dict{Date, Float64}=Dict{Date, Float64}())

    println("\n--- Part 8: Paper/Shadow Portfolio — 60-Day Validation (PDF §7C) ---\n")

    paper_nav    = 1_000_000.0
    paper_window = min(63, length(trading_days))
    paper_days   = trading_days[end-paper_window+1:end]

    # ── 8a: Initialize paper portfolio on recent window ───────────────────────
    paper_p1 = Dict{String, Float64}()
    for (tk, pdf) in price_data
        p = get_price_on_date(pdf, paper_days[1])
        p !== nothing && (paper_p1[tk] = p)
    end

    valid_paper_tks = [t for t in all_tickers if haskey(paper_p1, t)]
    vp_mask = [t in valid_paper_tks for t in all_tickers]
    paper_w  = weights[vp_mask]
    sum(paper_w) > 0 && (paper_w ./= sum(paper_w))
    paper_sl = sleeves[vp_mask]

    println("  Paper NAV:    \$$(round(Int, paper_nav))")
    println("  Window:       $(paper_days[1]) to $(paper_days[end]) ($(paper_window) days)")
    println("  Tickers:      $(length(valid_paper_tks))\n")

    paper_pf = initialize_portfolio(valid_paper_tks, paper_sl, paper_w,
                                     paper_nav, paper_p1, config)

    run_backtest!(paper_pf, price_data, div_data, vol_map, paper_days;
                  earnings_cal=earnings_cal, rolling_vol=rolling_vol,
                  sector_map=sector_map, div_yields=div_yields,
                  rolling_iv=rolling_iv,
                  heston_ts=heston_ts, vix_regime=vix_regime)

    paper_recs = paper_pf.daily_records
    if isempty(paper_recs)
        println("  [!] Paper portfolio produced no records — skipping validation.")
        return
    end

    # ── 8b: Paper portfolio metrics ───────────────────────────────────────
    p_fin  = paper_recs[end].nav
    p_ret  = (p_fin - paper_nav) / paper_nav * 100.0
    p_navs = [r.nav for r in paper_recs]
    p_dr   = diff(log.(p_navs))
    p_sh   = length(p_dr) > 0 && std(p_dr) > 0 ? (mean(p_dr)*252) / (std(p_dr)*sqrt(252)) : 0.0
    p_ds   = filter(x -> x < 0, p_dr)
    p_so   = length(p_ds) > 0 && std(p_ds) > 0 ? (mean(p_dr)*252) / (std(p_ds)*sqrt(252)) : 0.0
    ppk, p_mdd = -Inf, 0.0
    for v in p_navs; ppk = max(ppk, v); p_mdd = max(p_mdd, (ppk-v)/ppk); end

    p_tp = paper_recs[end].cumulative_premium
    p_td = paper_recs[end].cumulative_dividends
    p_tc = paper_recs[end].cumulative_costs
    p_assigns  = sum(sum(s.assignment_count for s in st.slots) for (_,st) in paper_pf.states)
    p_callaway = sum(sum(s.callaway_count  for s in st.slots) for (_,st) in paper_pf.states)
    p_trades   = sum(sum(s.trades          for s in st.slots) for (_,st) in paper_pf.states)
    p_repairs  = sum(sum(s.repair_count    for s in st.slots) for (_,st) in paper_pf.states)

    # ── 8c: Full backtest metrics over same window ────────────────────────
    full_window_recs = filter(r -> r.date >= paper_days[1], portfolio.daily_records)

    f_ret_w = NaN
    f_sh_w  = NaN
    f_mdd_w = NaN
    if length(full_window_recs) > 1
        f_navs_w = [r.nav for r in full_window_recs]
        f_ret_w  = (f_navs_w[end] - f_navs_w[1]) / f_navs_w[1] * 100.0
        f_dr_w   = diff(log.(f_navs_w))
        f_sh_w   = std(f_dr_w) > 0 ? (mean(f_dr_w)*252) / (std(f_dr_w)*sqrt(252)) : 0.0
        fpk, f_mdd_w = -Inf, 0.0
        for v in f_navs_w; fpk = max(fpk, v); f_mdd_w = max(f_mdd_w, (fpk-v)/fpk); end
    end

    full_trades  = sum(sum(s.trades for s in st.slots) for (_,st) in portfolio.states)
    full_assigns = sum(sum(s.assignment_count for s in st.slots) for (_,st) in portfolio.states)
    full_prem    = portfolio.daily_records[end].cumulative_premium
    full_costs   = portfolio.daily_records[end].cumulative_costs
    full_days    = length(trading_days)

    # ── 8d: Micro-metrics comparison ──────────────────────────────────────
    paper_prem_per_trade = p_trades > 0 ? p_tp / p_trades : 0.0
    full_prem_per_trade  = full_trades > 0 ? full_prem / full_trades : 0.0

    paper_cost_ratio = p_tp != 0.0 ? abs(p_tc / p_tp) * 100.0 : 0.0
    full_cost_ratio  = full_prem != 0.0 ? abs(full_costs / full_prem) * 100.0 : 0.0

    paper_assign_rate = p_trades > 0 ? p_assigns / p_trades * 100.0 : 0.0
    full_assign_rate  = full_trades > 0 ? full_assigns / full_trades * 100.0 : 0.0

    paper_trades_day  = p_trades / paper_window
    full_trades_day   = full_trades / full_days

    paper_ann_yield = (p_tp + p_td) / paper_nav * (252.0 / paper_window) * 100.0
    full_ann_yield  = (full_prem + portfolio.daily_records[end].cumulative_dividends) /
                      initial_nav * 100.0

    println("-- Paper vs Full Backtest Comparison --")
    comparison_df = DataFrame(
        Metric = [
            "Return (%)",
            "Sharpe (ann.)",
            "MaxDD (%)",
            "Premium / Trade (\$)",
            "Trades / Day",
            "Assignment Rate (%)",
            "Cost / Premium (%)",
            "Ann. Distribution Yield (%)",
        ],
        Paper = [
            round(p_ret, digits=2),
            round(p_sh, digits=3),
            round(p_mdd*100, digits=2),
            round(paper_prem_per_trade, digits=2),
            round(paper_trades_day, digits=2),
            round(paper_assign_rate, digits=1),
            round(paper_cost_ratio, digits=1),
            round(paper_ann_yield, digits=2),
        ],
        FullBacktest = [
            isnan(f_ret_w) ? NaN : round(f_ret_w, digits=2),
            isnan(f_sh_w)  ? NaN : round(f_sh_w, digits=3),
            isnan(f_mdd_w) ? NaN : round(f_mdd_w*100, digits=2),
            round(full_prem_per_trade, digits=2),
            round(full_trades_day, digits=2),
            round(full_assign_rate, digits=1),
            round(full_cost_ratio, digits=1),
            round(full_ann_yield, digits=2),
        ]
    )
    pretty_table(comparison_df,        column_labels=["Metric",
                       "Paper (\$$(round(Int, paper_nav/1000))K, $(paper_window)d)",
                       "Full (\$$(round(Int, initial_nav/1e6))M, $(full_days)d)"])

    # ── 8e: Per-ticker paper portfolio summary ────────────────────────────
    println("\n-- Paper Portfolio Per-Ticker Detail --")
    paper_tk_df = DataFrame(
        Ticker=String[], Sleeve=String[], Trades=Int[], Premium=Float64[],
        Assigns=Int[], CallAways=Int[], Repairs=Int[]
    )
    for tk in sort(collect(keys(paper_pf.states)))
        st = paper_pf.states[tk]
        push!(paper_tk_df, (
            tk, st.sleeve,
            sum(s.trades for s in st.slots),
            round(sum(s.total_premium for s in st.slots), digits=0),
            sum(s.assignment_count for s in st.slots),
            sum(s.callaway_count for s in st.slots),
            sum(s.repair_count for s in st.slots)
        ))
    end
    sort!(paper_tk_df, :Premium, rev=true)
    pretty_table(paper_tk_df,        column_labels=["Ticker","Sleeve","Trades","Premium(\$)","Assigns","CallAways","Repairs"])

    # ── 8f: Validation checklist ──────────────────────────────────────────
    println("\n-- Validation Checklist (PDF §7C) --")

    chk1 = paper_cost_ratio
    println("  [1] Cost/Premium ratio: $(round(chk1, digits=1))% vs full $(round(full_cost_ratio, digits=1))% — ",
            abs(chk1 - full_cost_ratio) < 20.0 ? "PASS (consistent)" : "REVIEW (divergent)")

    chk2 = paper_assign_rate
    println("  [2] Assignment rate: $(round(chk2, digits=1))% vs full $(round(full_assign_rate, digits=1))% — ",
            abs(chk2 - full_assign_rate) < 5.0 ? "PASS (stable)" : "REVIEW (shifted)")

    chk3 = paper_trades_day
    println("  [3] Trade frequency: $(round(chk3, digits=2))/day vs full $(round(full_trades_day, digits=2))/day — ",
            abs(chk3 - full_trades_day) / max(full_trades_day, 0.01) < 0.5 ? "PASS (stable)" : "REVIEW (changed)")

    chk4 = paper_ann_yield
    println("  [4] Annualized yield: $(round(chk4, digits=2))% vs full $(round(full_ann_yield, digits=2))% — ",
            chk4 > -5.0 ? "PASS (viable)" : "REVIEW (negative distribution)")

    if !isnan(f_ret_w)
        chk5 = abs(p_ret - f_ret_w)
        println("  [5] Return gap (same window): $(round(chk5, digits=2))pp — ",
                chk5 < 5.0 ? "PASS (scale-invariant)" : "REVIEW (scale effects present)")
    end

    println("\n  Note: Paper portfolio uses model fills/costs (not live market data).")
    println("  Pair with live shadow execution to validate spread and fill assumptions.\n")

    # ── 8g: Paper NAV chart ───────────────────────────────────────────────
    if nrow(daily_df) > 5
        paper_dates = [r.date for r in paper_recs]
        p_paper = plot(paper_dates, p_navs ./ (paper_nav/1e6) ./ 1e6;
            chart_defaults..., size = (960, 340),
            title = "Figure Paper: Shadow Portfolio NAV ($(yr), last $(paper_window)d)",
            xlabel = "Date", ylabel = "NAV (indexed to 1.0)",
            label = "Paper (\$$(round(Int,paper_nav/1000))K)",
            linewidth = 2, color = RGB(0.2, 0.4, 0.7))
        if length(full_window_recs) > 1
            f_navs_w = [r.nav for r in full_window_recs]
            plot!(paper_dates[1:min(length(paper_dates),length(f_navs_w))],
                  f_navs_w[1:min(length(paper_dates),length(f_navs_w))] ./ f_navs_w[1],
                  label = "Full Backtest (indexed)",
                  linewidth = 1.5, color = RGB(0.85, 0.45, 0.15), linestyle = :dash)
        end
        hline!([1.0], linestyle = :dot, color = :gray60, label = nothing, linewidth = 0.8)
        savefig(p_paper, joinpath(_PATH_TO_DATA, "paper_portfolio_$(yr).png"))
        display(p_paper)
        println("  -> Paper portfolio chart saved to data/paper_portfolio_$(yr).png")
    end
end
