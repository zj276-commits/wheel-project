# Backtest2024.jl — Wheel Strategy 2024 Backtest
#
# Same logic as Backtest.jl but for calendar year 2024.
# Downloads data to data/prices_2024/ and data/dividends_2024/.
# Outputs to data/*_2024.csv and data/*_2024.png.

include(joinpath(@__DIR__, "Include.jl"));

const YEAR = 2024

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: UNIVERSE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

println("══════════════════════════════════════════════════════")
println("    Wheel Strategy — $(YEAR) Backtest")
println("══════════════════════════════════════════════════════\n")

sagbm_df = load_sagbm_parameters();

vol_df = select(sagbm_df, :ticker, :volatility);

finviz_df = load_finviz_screener();
div_lookup = Dict{String, Float64}()
if nrow(finviz_df) > 0
    div_lookup = Dict(row.Ticker => row.div_yield for row in eachrow(finviz_df));
end

safe_tickers = [
    "PEP", "KO", "PG", "JNJ", "CME", "CMCSA", "VZ", "T", "IBM",
    "MO", "PM", "MDLZ", "EXC", "KMB", "PAYX", "TROW", "PFG",
    "SO", "DUK", "ED", "LNT", "GIS", "CAG", "REG", "CPB",
];

aggressive_tickers = [
    "TSLA", "NVDA", "AMD", "AAPL", "MSFT", "AMZN",
    "GOOG", "META", "NFLX", "DVN",
];

all_tickers = vcat(safe_tickers, aggressive_tickers);
sleeves = vcat(fill("Safe", length(safe_tickers)), fill("Aggressive", length(aggressive_tickers)));

n_safe = length(safe_tickers)
n_aggr = length(aggressive_tickers)

total_nav_initial = 600_000_000.0;
safe_weight = 0.60 / n_safe;
aggr_weight = 0.40 / n_aggr;
weights = vcat(fill(safe_weight, n_safe), fill(aggr_weight, n_aggr));

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
    "NVDA" => "Technology", "AMD" => "Technology",
    "AMZN" => "Consumer Discretionary", "TSLA" => "Consumer Discretionary",
    "EXC" => "Utilities", "SO" => "Utilities", "DUK" => "Utilities",
    "ED" => "Utilities", "LNT" => "Utilities",
    "PAYX" => "Industrials", "REG" => "Real Estate", "DVN" => "Energy",
)

universe_df = DataFrame(
    Ticker  = all_tickers,
    Sleeve  = sleeves,
    Sector  = [get(sector_map, t, "Unknown") for t in all_tickers],
    DivYield = [get(div_lookup, t, 0.0) for t in all_tickers],
    Weight  = weights .* 100.0,
    Notional = weights .* total_nav_initial,
)
println("  Universe: $(nrow(universe_df)) names ($(n_safe) Safe + $(n_aggr) Aggressive)")
pretty_table(universe_df, column_labels=["Ticker", "Sleeve", "Sector", "Div(%)", "Wt(%)", "Notional(\$)"])

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

println("\n══════════════════════════════════════════════════════")
println("    Part 2: Data Loading & Preprocessing ($(YEAR))")
println("══════════════════════════════════════════════════════\n")

vol_map = Dict{String, Float64}()
for row in eachrow(sagbm_df)
    vol_map[row.ticker] = row.volatility
end

start_date = Date(YEAR, 1, 2)
end_date   = Date(YEAR, 12, 31)

println("  Downloading $(YEAR) price data...")
price_data = download_all_prices(all_tickers, start_date, end_date; cache_year=YEAR);
println("  Downloading $(YEAR) dividend data...")
div_data = download_all_dividends(all_tickers, start_date, end_date; cache_year=YEAR);

rolling_vol = compute_rolling_volatility(price_data; window=30);
div_yields = compute_dividend_yields(div_data, price_data);

sleeves_map = Dict{String, String}(
    all_tickers[i] => sleeves[i] for i in 1:length(all_tickers)
)
rolling_iv = compute_rolling_iv(rolling_vol, sleeves_map);
println("  IV calibration: $(length(rolling_iv)) tickers calibrated")

earnings_cal = load_earnings_calendar(all_tickers; year=YEAR);

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

# Rolling Volatility Summary
println("\n── Rolling Volatility Summary ($(YEAR)) ──")
vol_summary_df = DataFrame(
    Ticker=String[], Sleeve=String[], RV_Last=Float64[], IV_Last=Float64[]
)
for (i, tk) in enumerate(all_tickers)
    rv_dict = get(rolling_vol, tk, Dict{Date, Float64}())
    iv_dict = get(rolling_iv, tk, Dict{Date, Float64}())
    if !isempty(rv_dict)
        last_date = maximum(keys(rv_dict))
        push!(vol_summary_df, (
            tk, sleeves[i],
            round(rv_dict[last_date]*100, digits=1),
            round(get(iv_dict, last_date, NaN)*100, digits=1)
        ))
    end
end
sort!(vol_summary_df, :RV_Last, rev=true)
pretty_table(vol_summary_df, column_labels=["Ticker", "Sleeve", "RV(%)", "IV(%)"])

# Benchmark (SPY)
benchmark_navs = nothing
try
    global spy_data = download_all_prices(["SPY"], start_date, end_date; cache_year=YEAR)
    if haskey(spy_data, "SPY") && nrow(spy_data["SPY"]) > 0
        spy_df = spy_data["SPY"]
        spy_day1 = get_price_on_date(spy_df, trading_days[1])
        if spy_day1 !== nothing
            scale = total_nav_initial / spy_day1
            global benchmark_navs = Float64[]
            for d in trading_days
                p = get_price_on_date(spy_df, d)
                push!(benchmark_navs, p !== nothing ? p * scale : (isempty(benchmark_navs) ? total_nav_initial : benchmark_navs[end]))
            end
        end
    end
catch e
    @warn "Could not download SPY benchmark: $e"
end

# ══════════════════════════════════════════════════════════════════════════════
# PART 3: RUN BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

println("\n══════════════════════════════════════════════════════")
println("    Part 3: Running $(YEAR) Backtest")
println("══════════════════════════════════════════════════════\n")

initial_nav = total_nav_initial
config = default_config()

portfolio = initialize_portfolio(all_tickers, sleeves, weights, initial_nav,
                                  prices_day1, config);

println("  Initial cash: \$$(round(Int, portfolio.cash))")
println("  Running $(length(trading_days)) trading days...\n")

run_backtest!(portfolio, price_data, div_data, vol_map, trading_days;
              earnings_cal=earnings_cal, rolling_vol=rolling_vol,
              sector_map=sector_map, div_yields=div_yields,
              rolling_iv=rolling_iv);

# ══════════════════════════════════════════════════════════════════════════════
# PART 4: RESULTS & REPORTING
# ══════════════════════════════════════════════════════════════════════════════

generate_report(portfolio; benchmark_navs=benchmark_navs, benchmark_label="SPY")

# ── Daily NAV DataFrame ──
daily_df = DataFrame(
    Date         = [r.date for r in portfolio.daily_records],
    NAV          = [r.nav for r in portfolio.daily_records],
    Cash         = [r.cash for r in portfolio.daily_records],
    SharesValue  = [r.shares_value for r in portfolio.daily_records],
    OptionMTM    = [r.option_mtm for r in portfolio.daily_records],
    BlockA_Value = [r.block_a_value for r in portfolio.daily_records],
    CumPremium   = [r.cumulative_premium for r in portfolio.daily_records],
    CumDividends = [r.cumulative_dividends for r in portfolio.daily_records],
    CumCosts     = [r.cumulative_costs for r in portfolio.daily_records],
    Delta        = [r.portfolio_delta for r in portfolio.daily_records],
    Gamma        = [r.portfolio_gamma for r in portfolio.daily_records],
    Vega         = [r.portfolio_vega for r in portfolio.daily_records],
)
daily_df[!, :DailyReturn] = vcat([0.0], diff(log.(daily_df.NAV)))
if benchmark_navs !== nothing && length(benchmark_navs) >= nrow(daily_df)
    daily_df[!, :SPY_NAV] = benchmark_navs[1:nrow(daily_df)]
    daily_df[!, :SPY_Return] = vcat([0.0], diff(log.(daily_df.SPY_NAV)))
    daily_df[!, :ExcessReturn] = daily_df.DailyReturn .- daily_df.SPY_Return
end

CSV.write(joinpath(_PATH_TO_DATA, "daily_nav_$(YEAR).csv"), daily_df)
println("\n  → Saved to data/daily_nav_$(YEAR).csv")

# ── Per-Ticker Performance ──
ticker_perf_df = DataFrame(
    Ticker=String[], Sleeve=String[], Sector=String[],
    Premium=Float64[], Dividends=Float64[], Costs=Float64[],
    Assigns=Int[], CallAways=Int[], Repairs=Int[], Trades=Int[],
    BlockA_PnL=Float64[]
)
for tk in sort(collect(keys(portfolio.states)))
    st = portfolio.states[tk]
    p1 = get(prices_day1, tk, NaN)
    last_p = NaN
    if haskey(price_data, tk)
        lp = get_price_on_date(price_data[tk], trading_days[end])
        lp !== nothing && (last_p = lp)
    end
    ba_pnl = st.block_a_shares * (last_p - p1)
    push!(ticker_perf_df, (
        tk, st.sleeve, get(sector_map, tk, "?"),
        round(sum(s.total_premium for s in st.slots), digits=0),
        round(st.total_dividends, digits=0),
        round(st.total_costs, digits=0),
        sum(s.assignment_count for s in st.slots),
        sum(s.callaway_count for s in st.slots),
        sum(s.repair_count for s in st.slots),
        sum(s.trades for s in st.slots),
        round(ba_pnl, digits=0)
    ))
end
sort!(ticker_perf_df, :Premium, rev=true)
CSV.write(joinpath(_PATH_TO_DATA, "ticker_performance_$(YEAR).csv"), ticker_perf_df)
println("  → Saved to data/ticker_performance_$(YEAR).csv")

# ── Monthly Returns ──
if nrow(daily_df) > 20
    monthly_df = DataFrame(Month=String[], NAV_Start=Float64[], NAV_End=Float64[],
                            Return=Float64[], MaxDD=Float64[], Premium=Float64[])
    ym_vec = [Dates.yearmonth(d) for d in daily_df.Date]
    months = unique(ym_vec)
    for ym in months
        mask = [y == ym for y in ym_vec]
        mdata = daily_df[mask, :]
        nrow(mdata) == 0 && continue
        nav_s = mdata.NAV[1]; nav_e = mdata.NAV[end]
        ret = (nav_e - nav_s) / nav_s * 100.0
        pk, dd = -Inf, 0.0
        for v in mdata.NAV; pk = max(pk, v); dd = max(dd, (pk-v)/pk); end
        prem_chg = mdata.CumPremium[end] - mdata.CumPremium[1]
        push!(monthly_df, (
            Dates.format(Date(ym[1], ym[2], 1), "yyyy-mm"),
            round(nav_s/1e6, digits=2), round(nav_e/1e6, digits=2),
            round(ret, digits=2), round(dd*100, digits=2),
            round(prem_chg/1e6, digits=3)
        ))
    end
    println("\n── Monthly Return Summary ($(YEAR)) ──")
    pretty_table(monthly_df,
        column_labels=["Month", "NAV Start(M\$)", "NAV End(M\$)", "Return(%)", "MaxDD(%)", "Premium(M\$)"])
end

# ── Charts ──
if nrow(daily_df) > 5
    dates_plot = daily_df.Date
    navs_plot = daily_df.NAV
    plot_dir = _PATH_TO_DATA
    yr = string(YEAR)

    # Chart 1: NAV Curve
    p1 = plot(dates_plot, navs_plot ./ 1e6,
        title = "Wheel Strategy — $(yr) Daily NAV",
        xlabel = "Date", ylabel = "NAV (\$ millions)",
        label = "Wheel Strategy", linewidth = 2, color = :steelblue,
        size = (900, 400), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
    hline!([initial_nav / 1e6], linestyle = :dash, color = :gray, label = "Initial NAV")
    if benchmark_navs !== nothing && length(benchmark_navs) >= nrow(daily_df)
        plot!(dates_plot, benchmark_navs[1:nrow(daily_df)] ./ 1e6,
              label = "SPY Buy-and-Hold", linewidth = 1.5, color = :orange, linestyle = :dash)
    end
    savefig(p1, joinpath(plot_dir, "nav_curve_$(yr).png"))
    display(p1)

    # Chart 2: Drawdown
    peak_nav = accumulate(max, navs_plot)
    drawdown_pct = (peak_nav .- navs_plot) ./ peak_nav .* 100.0
    p2 = plot(dates_plot, -drawdown_pct,
        title = "Drawdown from Peak ($(yr))",
        xlabel = "Date", ylabel = "Drawdown (%)",
        label = "Wheel Strategy", linewidth = 1.5, color = :red,
        fill = (0, 0.2, :red), size = (900, 300), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
    if benchmark_navs !== nothing && length(benchmark_navs) >= nrow(daily_df)
        bm = benchmark_navs[1:nrow(daily_df)]
        bm_peak = accumulate(max, bm)
        bm_dd = (bm_peak .- bm) ./ bm_peak .* 100.0
        plot!(dates_plot, -bm_dd, label = "SPY", linewidth = 1.0, color = :orange, linestyle = :dash)
    end
    savefig(p2, joinpath(plot_dir, "drawdown_$(yr).png"))
    display(p2)

    # Chart 3: Income Decomposition
    p3 = plot(dates_plot, daily_df.CumPremium ./ 1e6,
        title = "Cumulative Income Decomposition ($(yr))",
        xlabel = "Date", ylabel = "\$ millions",
        label = "Premium", linewidth = 2, color = :green,
        size = (900, 400), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
    plot!(dates_plot, daily_df.CumDividends ./ 1e6, label = "Dividends", linewidth = 2, color = :blue)
    plot!(dates_plot, daily_df.CumCosts ./ 1e6, label = "Trading Costs", linewidth = 2, color = :red, linestyle = :dash)
    net_income = (daily_df.CumPremium .+ daily_df.CumDividends .- daily_df.CumCosts) ./ 1e6
    plot!(dates_plot, net_income, label = "Net Income", linewidth = 2.5, color = :black)
    savefig(p3, joinpath(plot_dir, "income_decomposition_$(yr).png"))
    display(p3)

    # Chart 4: Monthly Returns
    if @isdefined(monthly_df) && nrow(monthly_df) > 0
        month_colors = [r >= 0 ? :green : :red for r in monthly_df.Return]
        p10 = bar(monthly_df.Month, monthly_df.Return,
            title = "Monthly Returns ($(yr)) (%)",
            xlabel = "Month", ylabel = "Return (%)",
            label = nothing, color = month_colors, alpha = 0.8,
            size = (900, 350), dpi = 150, rotation = 45,
            left_margin = 10Plots.mm, bottom_margin = 12Plots.mm)
        hline!([0.0], linestyle = :dash, color = :black, label = nothing)
        savefig(p10, joinpath(plot_dir, "monthly_returns_$(yr).png"))
        display(p10)
    end

    println("\n  → Charts saved to data/ folder:")
    println("    1. nav_curve_$(yr).png")
    println("    2. drawdown_$(yr).png")
    println("    3. income_decomposition_$(yr).png")
    println("    4. monthly_returns_$(yr).png")
end

println("\n✓ $(YEAR) Backtest complete.")
