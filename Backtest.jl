# =============================================================================
# Backtest.jl — Wheel Strategy 2025 Backtest (S&P 500 Universe)
# =============================================================================
# Uses pre-2024 SAGBM parameters for volatility estimation and real 2025 daily
# prices from Yahoo Finance to simulate the complete Wheel strategy day-by-day.
# =============================================================================

include(joinpath(@__DIR__, "Include.jl"));

println("═══════════════════════════════════════════════════════════════")
println("  WHEEL STRATEGY — S&P 500 UNIVERSE — 2025 BACKTEST")
println("═══════════════════════════════════════════════════════════════")
println()

# =============================================================================
# STEP 1: Define the S&P 500 Universe (all tickers in SAGBM parameter set)
# =============================================================================
safe_tickers = [
    "PEP", "KO", "PG", "JNJ", "CME", "CMCSA", "VZ", "T", "IBM",
    "MO", "PM", "MDLZ", "EXC", "KMB", "PAYX", "TROW", "PFG",
    "SO", "DUK", "ED", "LNT", "GIS", "CAG", "REG", "CPB",
];

aggressive_tickers = [
    "FITB", "HBAN", "HST", "HAS", "DVN", "APA", "SWKS", "ZION", "LKQ", "HAL",
];

all_tickers = vcat(safe_tickers, aggressive_tickers);
sleeves = vcat(fill("Safe", length(safe_tickers)), fill("Aggressive", length(aggressive_tickers)));

n_safe = length(safe_tickers)
n_aggr = length(aggressive_tickers)
safe_weight = 0.60 / n_safe
aggr_weight = 0.40 / n_aggr
weights = vcat(fill(safe_weight, n_safe), fill(aggr_weight, n_aggr));

println("Universe: $(length(all_tickers)) S&P 500 stocks")
println("  Safe sleeve:       $n_safe names ($(round(safe_weight*100, digits=2))% each)")
println("  Aggressive sleeve: $n_aggr names ($(round(aggr_weight*100, digits=2))% each)")
println()

# =============================================================================
# STEP 2: Load SAGBM volatility parameters (calibrated from 2014–2024)
# =============================================================================
println("Loading SAGBM volatility parameters...")
sagbm_df = load_sagbm_parameters();
vol_map = Dict{String, Float64}()
for row in eachrow(sagbm_df)
    vol_map[row.ticker] = row.volatility
end

missing_vol = [t for t in all_tickers if !haskey(vol_map, t)]
if !isempty(missing_vol)
    @warn "Tickers missing from SAGBM (using default σ=0.25): $missing_vol"
end

println("  Loaded volatility for $(length(vol_map)) S&P 500 tickers")
println()

# =============================================================================
# STEP 3: Download 2025 daily price data from Yahoo Finance
# =============================================================================
start_date = Date(2025, 1, 2)
end_date   = Date(2025, 12, 31)

println("Downloading 2025 price data from Yahoo Finance...")
price_data = download_all_prices(all_tickers, start_date, end_date);
println("  Downloaded price data for $(length(price_data)) tickers")
println()

println("Downloading 2025 dividend data...")
div_data = download_all_dividends(all_tickers, start_date, end_date);
total_div_events = sum(nrow(df) for (_, df) in div_data)
println("  Found $total_div_events dividend events across all tickers")
println()

# =============================================================================
# STEP 4: Determine trading days and day-1 prices
# =============================================================================
trading_days = get_trading_days(price_data);
println("Trading period: $(trading_days[1]) to $(trading_days[end]) ($(length(trading_days)) days)")
println()

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

# =============================================================================
# STEP 5: Initialize portfolio and run simulation
# =============================================================================
initial_nav = 600_000_000.0
config = default_config()

println("Initializing portfolio with \$$(round(Int, initial_nav)) NAV...")
portfolio = initialize_portfolio(all_tickers, sleeves, weights, initial_nav,
                                  prices_day1, config);
println("  Cash after Block A purchases: \$$(round(Int, portfolio.cash))")
println()

println("Running daily Wheel strategy simulation...")
println("─────────────────────────────────────────────")
run_backtest!(portfolio, price_data, div_data, vol_map, trading_days);
println("─────────────────────────────────────────────")
println("Simulation complete!")
println()

# =============================================================================
# STEP 6: Generate comprehensive report
# =============================================================================
generate_report(portfolio)

# =============================================================================
# STEP 7: Plot daily NAV curve
# =============================================================================
if length(portfolio.daily_records) > 0
    dates_plot = [r.date for r in portfolio.daily_records]
    navs_plot = [r.nav for r in portfolio.daily_records]

    p1 = plot(dates_plot, navs_plot ./ 1e6,
        title = "Wheel Strategy — 2025 Daily NAV",
        xlabel = "Date", ylabel = "NAV (\$ millions)",
        legend = false, linewidth = 2, color = :steelblue,
        size = (900, 400), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)

    hline!([initial_nav / 1e6], linestyle = :dash, color = :gray, label = "Initial NAV")

    nav_plot_path = joinpath(_PATH_TO_DATA, "nav_curve_2025.png")
    savefig(p1, nav_plot_path)
    println("NAV curve saved to: $nav_plot_path")
    display(p1)
end
