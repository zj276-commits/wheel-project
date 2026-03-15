# Backtest.jl — Wheel Strategy 2025 Backtest (S&P 500 Universe)
#
# Full-featured backtest with:
#   CRR American pricing with dividend yield, portfolio Greeks,
#   dynamic rolling volatility, earnings avoidance, configurable laddering,
#   bid-ask spread model, liquidity screens, risk overlays (VaR/ES),
#   partial fills, extended cost model, sector/name caps,
#   adaptive tenor/delta, benchmark comparison,
#   comprehensive parameter sweep, portfolio-level stress tests.

include(joinpath(@__DIR__, "Include.jl"));

const RUN_PARAMETER_SWEEP = false
const RUN_STRESS_TEST     = false

# ── Step 1: Universe — 25 Safe + 10 Aggressive ──────────────────────────────

safe_tickers = [
    "PEP", "KO", "PG", "JNJ", "CME", "CMCSA", "VZ", "T", "IBM",
    "MO", "PM", "MDLZ", "EXC", "KMB", "PAYX", "TROW", "PFG",
    "SO", "DUK", "ED", "LNT", "GIS", "CAG", "REG", "CPB",
];

aggressive_tickers = [
    "TSLA", "NVDA", "AMD",
    "AAPL", "MSFT", "AMZN",
    "GOOG", "META", "NFLX",
    "DVN",
];

all_tickers = vcat(safe_tickers, aggressive_tickers);
sleeves = vcat(fill("Safe", length(safe_tickers)), fill("Aggressive", length(aggressive_tickers)));

n_safe = length(safe_tickers)
n_aggr = length(aggressive_tickers)
safe_weight = 0.60 / n_safe
aggr_weight = 0.40 / n_aggr
weights = vcat(fill(safe_weight, n_safe), fill(aggr_weight, n_aggr));

# Sector map for sector cap enforcement (TODO item 8)
sector_map = Dict{String, String}(
    "PEP" => "Consumer Staples", "KO" => "Consumer Staples", "PG" => "Consumer Staples",
    "MO" => "Consumer Staples", "PM" => "Consumer Staples", "MDLZ" => "Consumer Staples",
    "KMB" => "Consumer Staples", "GIS" => "Consumer Staples", "CAG" => "Consumer Staples",
    "CPB" => "Consumer Staples",
    "JNJ" => "Health Care",
    "CME" => "Financials", "TROW" => "Financials", "PFG" => "Financials",
    "CMCSA" => "Communication", "VZ" => "Communication", "T" => "Communication",
    "GOOG" => "Communication", "META" => "Communication", "NFLX" => "Communication",
    "IBM" => "Technology", "AAPL" => "Technology", "MSFT" => "Technology",
    "AMZN" => "Consumer Discretionary", "TSLA" => "Consumer Discretionary",
    "NVDA" => "Technology", "AMD" => "Technology",
    "EXC" => "Utilities", "SO" => "Utilities", "DUK" => "Utilities",
    "ED" => "Utilities", "LNT" => "Utilities",
    "PAYX" => "Industrials",
    "REG" => "Real Estate",
    "DVN" => "Energy",
)

# ── Step 2: Load SAGBM volatility (fallback when rolling vol unavailable) ────

sagbm_df = load_sagbm_parameters();
vol_map = Dict{String, Float64}()
for row in eachrow(sagbm_df)
    vol_map[row.ticker] = row.volatility
end

missing_vol = [t for t in all_tickers if !haskey(vol_map, t)]
if !isempty(missing_vol)
    @warn "Tickers missing from SAGBM (using default σ=0.25): $missing_vol"
end

# ── Step 3: Download 2025 daily price & dividend data ────────────────────────

start_date = Date(2025, 1, 2)
end_date   = Date(2025, 12, 31)

price_data = download_all_prices(all_tickers, start_date, end_date);
div_data = download_all_dividends(all_tickers, start_date, end_date);

# ── Step 4: Compute rolling volatility & dividend yields ─────────────────────

rolling_vol = compute_rolling_volatility(price_data; window=30);
div_yields = compute_dividend_yields(div_data, price_data);

# ── Step 4b: Calibrate implied volatility (VRP model) ─────────────────────

sleeves_map = Dict{String, String}(
    all_tickers[i] => sleeves[i] for i in 1:length(all_tickers)
)
rolling_iv = compute_rolling_iv(rolling_vol, sleeves_map);
println("  IV calibration: $(length(rolling_iv)) tickers calibrated")

# ── Step 5: Load earnings calendar ──────────────────────────────────────────

earnings_cal = load_earnings_calendar(all_tickers; year=2025);

# ── Step 6: Trading days and day-1 prices ────────────────────────────────────

trading_days = get_trading_days(price_data);

prices_day1 = Dict{String, Float64}()
for (ticker, pdf) in price_data
    p = get_price_on_date(pdf, trading_days[1])
    if p !== nothing
        prices_day1[ticker] = p
    else
        for d in trading_days
            p2 = get_price_on_date(pdf, d)
            if p2 !== nothing
                prices_day1[ticker] = p2
                break
            end
        end
    end
end

# ── Step 7: Download benchmark (SPY) for comparison (TODO item 17) ───────────

benchmark_navs = nothing
try
    global spy_data = download_all_prices(["SPY"], start_date, end_date)
    if haskey(spy_data, "SPY") && nrow(spy_data["SPY"]) > 0
        spy_df = spy_data["SPY"]
        spy_day1 = get_price_on_date(spy_df, trading_days[1])
        if spy_day1 !== nothing
            scale = 600_000_000.0 / spy_day1
            global benchmark_navs = Float64[]
            for d in trading_days
                p = get_price_on_date(spy_df, d)
                push!(benchmark_navs, p !== nothing ? p * scale : (isempty(benchmark_navs) ? 600e6 : benchmark_navs[end]))
            end
        end
    end
catch e
    @warn "Could not download SPY benchmark: $e"
end

# ── Step 8: Initialize portfolio and run simulation ──────────────────────────

initial_nav = 600_000_000.0
config = default_config()

portfolio = initialize_portfolio(all_tickers, sleeves, weights, initial_nav,
                                  prices_day1, config);

run_backtest!(portfolio, price_data, div_data, vol_map, trading_days;
              earnings_cal=earnings_cal, rolling_vol=rolling_vol,
              sector_map=sector_map, div_yields=div_yields,
              rolling_iv=rolling_iv);

# ── Step 9: Report ───────────────────────────────────────────────────────────

generate_report(portfolio; benchmark_navs=benchmark_navs, benchmark_label="SPY")

# ── Step 10: Plot daily NAV curve with benchmark ─────────────────────────────

if length(portfolio.daily_records) > 0
    dates_plot = [r.date for r in portfolio.daily_records]
    navs_plot = [r.nav for r in portfolio.daily_records]

    p1 = plot(dates_plot, navs_plot ./ 1e6,
        title = "Wheel Strategy — 2025 Daily NAV",
        xlabel = "Date", ylabel = "NAV (\$ millions)",
        label = "Wheel Strategy", linewidth = 2, color = :steelblue,
        size = (900, 400), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)

    hline!([initial_nav / 1e6], linestyle = :dash, color = :gray, label = "Initial NAV")

    if benchmark_navs !== nothing && length(benchmark_navs) >= length(navs_plot)
        plot!(dates_plot, benchmark_navs[1:length(navs_plot)] ./ 1e6,
              label = "SPY Buy-and-Hold", linewidth = 1.5, color = :orange, linestyle = :dash)
    end

    nav_plot_path = joinpath(_PATH_TO_DATA, "nav_curve_2025.png")
    savefig(p1, nav_plot_path)
    display(p1)
end

# ── Step 11 (optional): Comprehensive Parameter Sweep (TODO item 14) ─────────

if RUN_PARAMETER_SWEEP
    println("\n══════════════════════════════════════════════════════")
    println("         Parameter Sweep — Varner PDF §6 Variables")
    println("══════════════════════════════════════════════════════\n")

    sweep_results = DataFrame(
        Config=String[], FinalNAV=Float64[], Return=Float64[],
        Sharpe=Float64[], Sortino=Float64[], MaxDD=Float64[],
        Premium=Float64[], Assigns=Int[], CallAways=Int[], Trades=Int[]
    )

    sweep_configs = [
        ("Baseline 60/40",
         WheelConfig(), weights),
        ("Conservative 80/20",
         WheelConfig(delta_put_safe=(0.20,0.25), delta_call_safe=(0.25,0.30),
                     delta_put_aggr=(0.20,0.25), delta_call_aggr=(0.25,0.30)),
         vcat(fill(0.80/n_safe, n_safe), fill(0.20/n_aggr, n_aggr))),
        ("Aggressive 40/60",
         WheelConfig(delta_put_aggr=(0.30,0.40), delta_call_aggr=(0.35,0.45),
                     earnings_policy=:widen),
         vcat(fill(0.40/n_safe, n_safe), fill(0.60/n_aggr, n_aggr))),
        ("Weekly tenor only",
         WheelConfig(tenor_days=[7]), weights),
        ("Monthly tenor only",
         WheelConfig(tenor_days=[30]), weights),
        ("2 ladders",
         WheelConfig(max_ladders=2), weights),
        ("3 ladders",
         WheelConfig(max_ladders=3), weights),
        ("Widen at earnings",
         WheelConfig(earnings_policy=:widen, earnings_wider_delta=0.10), weights),
        ("High delta puts",
         WheelConfig(delta_put_safe=(0.30,0.35), delta_put_aggr=(0.30,0.40)), weights),
        ("Low delta puts",
         WheelConfig(delta_put_safe=(0.15,0.20), delta_put_aggr=(0.20,0.25)), weights),
        ("No risk overlay",
         WheelConfig(var_limit_daily=1.0, es_limit_daily=1.0), weights),
        ("Tight risk overlay",
         WheelConfig(var_limit_daily=0.015, es_limit_daily=0.020), weights),
        ("No adaptive controls",
         WheelConfig(adaptive_tenor=false, adaptive_delta=false), weights),
        ("3 ladders + weekly + widen",
         WheelConfig(max_ladders=3, tenor_days=[7], earnings_policy=:widen), weights),
        ("Aggressive 40/60 + 2 ladders",
         WheelConfig(max_ladders=2, delta_put_aggr=(0.30,0.40), delta_call_aggr=(0.35,0.45)),
         vcat(fill(0.40/n_safe, n_safe), fill(0.60/n_aggr, n_aggr))),
    ]

    for (label, cfg, sw) in sweep_configs
        pf = initialize_portfolio(all_tickers, sleeves, sw, initial_nav, prices_day1, cfg)
        run_backtest!(pf, price_data, div_data, vol_map, trading_days;
                      earnings_cal=earnings_cal, rolling_vol=rolling_vol,
                      sector_map=sector_map, div_yields=div_yields,
                      rolling_iv=rolling_iv)

        recs = pf.daily_records
        isempty(recs) && continue
        fin = recs[end].nav
        ret = (fin - initial_nav) / initial_nav * 100.0
        navs = [r.nav for r in recs]
        dr = diff(log.(navs))
        sh = length(dr) > 0 && std(dr) > 0 ? (mean(dr)*252) / (std(dr)*sqrt(252)) : 0.0
        ds = filter(x -> x < 0, dr)
        so = length(ds) > 0 && std(ds) > 0 ? (mean(dr)*252) / (std(ds)*sqrt(252)) : 0.0
        pk, mdd = -Inf, 0.0
        for v in navs; pk = max(pk, v); mdd = max(mdd, (pk-v)/pk); end

        tp = recs[end].cumulative_premium
        ta = sum(sum(s.assignment_count for s in st.slots) for (_,st) in pf.states)
        tc2 = sum(sum(s.callaway_count for s in st.slots) for (_,st) in pf.states)
        tt = sum(sum(s.trades for s in st.slots) for (_,st) in pf.states)

        push!(sweep_results, (label, round(fin/1e6, digits=2), round(ret, digits=2),
              round(sh, digits=3), round(so, digits=3), round(mdd*100, digits=2),
              round(tp/1e6, digits=2), ta, tc2, tt))
    end

    pretty_table(sweep_results,
        column_labels=["Configuration", "Final NAV (M\$)", "Return %",
                        "Sharpe", "Sortino", "MaxDD %", "Premium (M\$)",
                        "Assigns", "CallAways", "Trades"])
end

# ── Step 12 (optional): Portfolio-Level Stress Test (TODO item 15) ────────────

if RUN_STRESS_TEST
    println("\n══════════════════════════════════════════════════════")
    println("      Portfolio-Level Stress Tests (Varner PDF §7B)")
    println("══════════════════════════════════════════════════════\n")

    stress_scenarios = [
        ("Normal (baseline)",           1.0,  0.0,   0.0,   0),
        ("Vol Spike (2× vol)",          2.0,  0.0,   0.0,   0),
        ("Bear Market (-20% drift)",    1.5, -0.20,  0.0,   0),
        ("Flash Crash (-10% gap d30)",  2.0,  0.0,  -0.10, 30),
        ("Name Blowup (-30% gap d60)",  1.5,  0.0,  -0.30, 60),
        ("Bull Squeeze (+15% gap d45)", 1.5,  0.30,  0.15, 45),
    ]

    stress_results = DataFrame(
        Scenario=String[], FinalNAV=Float64[], Return=Float64[],
        Sharpe=Float64[], MaxDD=Float64[], Premium=Float64[]
    )

    for (label, vol_mult, drift_adj, gap_pct, gap_day) in stress_scenarios
        stressed_prices = apply_stress_to_prices(price_data;
            vol_mult=vol_mult, drift_adj=drift_adj,
            gap_pct=gap_pct, gap_day=gap_day)

        stressed_rolling = compute_rolling_volatility(stressed_prices; window=30)

        pf = initialize_portfolio(all_tickers, sleeves, weights, initial_nav, prices_day1, config)
        stressed_days = get_trading_days(stressed_prices)

        stressed_iv = compute_rolling_iv(stressed_rolling, sleeves_map)
        run_backtest!(pf, stressed_prices, div_data, vol_map, stressed_days;
                      earnings_cal=earnings_cal, rolling_vol=stressed_rolling,
                      sector_map=sector_map, div_yields=div_yields,
                      rolling_iv=stressed_iv)

        recs = pf.daily_records
        isempty(recs) && continue
        fin = recs[end].nav
        ret = (fin - initial_nav) / initial_nav * 100.0
        navs = [r.nav for r in recs]
        dr = diff(log.(navs))
        sh = length(dr) > 0 && std(dr) > 0 ? (mean(dr)*252) / (std(dr)*sqrt(252)) : 0.0
        pk, mdd = -Inf, 0.0
        for v in navs; pk = max(pk, v); mdd = max(mdd, (pk-v)/pk); end
        tp = recs[end].cumulative_premium

        push!(stress_results, (label, round(fin/1e6, digits=2), round(ret, digits=2),
              round(sh, digits=3), round(mdd*100, digits=2), round(tp/1e6, digits=2)))
    end

    pretty_table(stress_results,
        column_labels=["Scenario", "Final NAV (M\$)", "Return %",
                        "Sharpe", "MaxDD %", "Premium (M\$)"])

    # Single-ticker stress scenarios for Safe/Aggressive representatives
    println("\n── Single-Ticker Stress Scenarios ──")
    for (label, ticker) in [("Safe", safe_tickers[1]), ("Aggressive", aggressive_tickers[1])]
        haskey(prices_day1, ticker) || continue
        S₀ = prices_day1[ticker]
        μ = get(vol_map, ticker, 0.08) * 0.5
        σ = get(vol_map, ticker, 0.25)

        println("\n  $label representative: $ticker (S₀=\$$(round(S₀, digits=2)), σ=$(round(σ, digits=2)))")
        results = run_stress_scenarios(S₀, μ, σ, 1.0; n_paths=5000)
        pretty_table(results,
            column_labels=["Scenario", "Mean Ret%", "Median Ret%",
                            "VaR 95%", "Avg MaxDD%", "% Below -20%"])
    end
end
