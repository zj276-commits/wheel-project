# Backtest.jl — Wheel Strategy 2025 Full Backtest
#
# Merged from: Wheel Strategy Code (Part 3: Universe Construction) + Backtest.jl
# This file is the single entry point for running a complete backtest.
#
# Features:
#   CRR American pricing with dividend yield, portfolio Greeks,
#   dynamic rolling volatility, IV calibration (VRP model),
#   earnings avoidance/widen/reduce_size, configurable laddering,
#   bid-ask spread model, liquidity screens, risk overlays (VaR/ES),
#   partial fills, extended cost model, sector/name caps,
#   adaptive tenor/delta, VIX download, benchmark comparison,
#   comprehensive parameter sweep, portfolio-level stress tests,
#   full DataFrame outputs for analysis.

include(joinpath(@__DIR__, "Include.jl"));

const RUN_PARAMETER_SWEEP = false
const RUN_STRESS_TEST     = false

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: UNIVERSE CONSTRUCTION (from Wheel Strategy Code)
# Implements Portfolio Construction per Varner PDF Section 3.
# ══════════════════════════════════════════════════════════════════════════════

println("══════════════════════════════════════════════════════")
println("    Part 1: Universe Construction (Varner PDF §3)")
println("══════════════════════════════════════════════════════\n")

# ── 1a: Load volatility data ─────────────────────────────────────────────────
# SAGBM parameters precomputed from 2014–2024 S&P 500 data (Polygon.io via course).
# The .jld2 file comes from VLQuantitativeFinancePackage → MyTestingMarketDataSet().
# If not available, we use the precomputed SAGBM CSV as fallback.

sagbm_df = load_sagbm_parameters();

has_market_data = isfile(joinpath(_PATH_TO_DATA, "SP500-Daily-OHLC-1-3-2025-to-11-18-2025.jld2"));
if has_market_data
    println("  Loading JLD2 market data for volatility estimation...")
    original_dataset = MyTestingMarketDataSet() |> x -> x["dataset"];
    maximum_number_trading_days = original_dataset["NVDL"] |> nrow;
    dataset = let
        d = Dict{String, DataFrame}();
        for (ticker, data) ∈ original_dataset
            nrow(data) == maximum_number_trading_days && (d[ticker] = data)
        end; d
    end;
    list_of_tickers = keys(dataset) |> collect |> sort;

    growth_rate_array = log_growth_matrix(dataset, list_of_tickers,
        Δt = 1/252, risk_free_rate = 0.0);
    vols = std(growth_rate_array, dims=1) |> vec;
    vol_df = DataFrame(ticker = list_of_tickers, volatility = vols);

    println("  Log-growth matrix: $(size(growth_rate_array, 1)) days × $(size(growth_rate_array, 2)) tickers")
    println("  Volatility range: [$(round(minimum(vols), digits=4)), $(round(maximum(vols), digits=4))]")
    println("  Mean vol: $(round(mean(vols), digits=4)),  Median vol: $(round(median(vols), digits=4))\n")
else
    @warn "JLD2 market data not found — using SAGBM precomputed volatility."
    vol_df = select(sagbm_df, :ticker, :volatility);
end

# ── 1b: Load dividend yield from Finviz ──────────────────────────────────────

finviz_df = load_finviz_screener();
div_lookup = Dict{String, Float64}()
mktcap_lookup = Dict{String, Float64}()
volume_lookup = Dict{String, Float64}()
if nrow(finviz_df) > 0
    div_lookup = Dict(row.Ticker => row.div_yield for row in eachrow(finviz_df));
    for row in eachrow(finviz_df)
        mc = row[Symbol("Market Cap")]; vol = row[:Volume]
        mktcap_lookup[row.Ticker] = (mc isa Number && !ismissing(mc)) ? Float64(mc) : 0.0
        volume_lookup[row.Ticker] = (vol isa Number && !ismissing(vol)) ? Float64(vol) : 0.0
    end
end

# ── 1c: Universe selection — 25 Safe + 10 Aggressive (manual) ────────────────
# Selection criteria (Varner PDF §3):
#   Safe:       low vol, high div yield, deep options markets
#   Aggressive: top-quintile vol, deep options markets, sector diversity

safe_tickers = [
    "PEP",    # PepsiCo — Consumer Staples, ~3.7% div
    "KO",     # Coca-Cola — Consumer Staples, ~3.0% div
    "PG",     # Procter & Gamble — Consumer Staples, ~2.5% div
    "JNJ",    # Johnson & Johnson — Healthcare, ~3.2% div
    "CME",    # CME Group — Financials, ~3.6% div
    "CMCSA",  # Comcast — Communication, ~3.9% div
    "VZ",     # Verizon — Communication, ~6.5% div
    "T",      # AT&T — Communication, ~5.0% div
    "IBM",    # IBM — Technology, ~3.0% div
    "MO",     # Altria — Consumer Staples, ~8.0% div
    "PM",     # Philip Morris — Consumer Staples, ~4.5% div
    "MDLZ",   # Mondelez — Consumer Staples, ~3.5% div
    "EXC",    # Exelon — Utilities, ~3.4% div
    "KMB",    # Kimberly-Clark — Consumer Staples, ~5.0% div
    "PAYX",   # Paychex — Industrials, ~4.4% div
    "TROW",   # T. Rowe Price — Financials, ~5.7% div
    "PFG",    # Principal Financial — Financials, ~3.6% div
    "SO",     # Southern Company — Utilities, ~3.5% div
    "DUK",    # Duke Energy — Utilities, ~3.8% div
    "ED",     # Consolidated Edison — Utilities, ~3.5% div
    "LNT",    # Alliant Energy — Utilities, ~3.0% div
    "GIS",    # General Mills — Consumer Staples, ~3.5% div
    "CAG",    # Conagra Brands — Consumer Staples, ~5.0% div
    "REG",    # Regency Centers — Real Estate, ~3.8% div
    "CPB",    # Campbell Soup — Consumer Staples, ~6.2% div
];

aggressive_tickers = [
    "TSLA",   # Tesla — Consumer Discretionary, σ=0.49
    "NVDA",   # NVIDIA — Technology, σ=0.39
    "AMD",    # AMD — Technology, σ=0.47
    "AAPL",   # Apple — Technology, deepest options market globally
    "MSFT",   # Microsoft — Technology, deepest options market globally
    "AMZN",   # Amazon — Consumer Discretionary, deep options market
    "GOOG",   # Alphabet — Communication, σ=0.31
    "META",   # Meta Platforms — Communication, σ~0.30
    "NFLX",   # Netflix — Communication, σ=0.36
    "DVN",    # Devon Energy — Energy, σ=0.46 (sector diversity)
];

all_tickers = vcat(safe_tickers, aggressive_tickers);
sleeves = vcat(fill("Safe", length(safe_tickers)), fill("Aggressive", length(aggressive_tickers)));

n_safe = length(safe_tickers)
n_aggr = length(aggressive_tickers)

# ── 1d: Portfolio weights — 60/40 Safe/Aggressive ────────────────────────────

total_nav_initial = 600_000_000.0;
safe_alloc  = 0.60;
aggr_alloc  = 0.40;
max_per_name = 0.05;

safe_weight = safe_alloc / n_safe;
aggr_weight = aggr_alloc / n_aggr;
weights = vcat(fill(safe_weight, n_safe), fill(aggr_weight, n_aggr));

@assert safe_weight ≤ max_per_name "Safe per-name weight $(round(safe_weight*100,digits=2))% exceeds 5% cap"
@assert aggr_weight ≤ max_per_name "Aggr per-name weight $(round(aggr_weight*100,digits=2))% exceeds 5% cap"

# ── 1e: Sector map for diversification (PDF §3) ─────────────────────────────

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
    "PAYX" => "Industrials",
    "REG" => "Real Estate",
    "DVN" => "Energy",
)

# ── 1f: Display universe table ───────────────────────────────────────────────

universe_df = DataFrame(
    Ticker  = all_tickers,
    Sleeve  = sleeves,
    Sector  = [get(sector_map, t, "Unknown") for t in all_tickers],
    DivYield = [get(div_lookup, t, 0.0) for t in all_tickers],
    Weight  = weights .* 100.0,
    Notional = weights .* total_nav_initial,
)
sagbm_vol_lookup = nrow(sagbm_df) > 0 ? Dict(row.ticker => row.volatility for row in eachrow(sagbm_df)) : Dict{String,Float64}()
universe_df[!, :SAGBM_Vol] = [get(sagbm_vol_lookup, t, NaN) for t in all_tickers];

println("  Universe: $(nrow(universe_df)) names ($(n_safe) Safe + $(n_aggr) Aggressive)")
println("  Safe per-name: $(round(safe_weight*100, digits=2))%   Aggr per-name: $(round(aggr_weight*100, digits=2))%")
println("  Total NAV: \$$(Int(total_nav_initial))\n")
pretty_table(universe_df, column_labels=["Ticker", "Sleeve", "Sector", "Div(%)", "Wt(%)", "Notional(\$)", "SAGBM σ"])

sector_weights = combine(groupby(universe_df, :Sector), :Weight => sum => :TotalWeight);
sort!(sector_weights, :TotalWeight, rev=true);
println("\n  Sector Diversification:")
pretty_table(sector_weights, column_labels=["Sector", "Weight (%)"])

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

println("\n══════════════════════════════════════════════════════")
println("    Part 2: Data Loading & Preprocessing")
println("══════════════════════════════════════════════════════\n")

# ── 2a: SAGBM static volatility (fallback) ──────────────────────────────────

vol_map = Dict{String, Float64}()
for row in eachrow(sagbm_df)
    vol_map[row.ticker] = row.volatility
end
missing_vol = [t for t in all_tickers if !haskey(vol_map, t)]
if !isempty(missing_vol)
    @warn "Tickers missing from SAGBM (using default σ=0.25): $missing_vol"
end

# ── 2b: Download 2025 daily price & dividend data ────────────────────────────

start_date = Date(2025, 1, 2)
end_date   = Date(2025, 12, 31)

println("  Downloading price data...")
price_data = download_all_prices(all_tickers, start_date, end_date);
println("  Downloading dividend data...")
div_data = download_all_dividends(all_tickers, start_date, end_date);

# ── 2c: Compute rolling volatility & dividend yields ─────────────────────────

rolling_vol = compute_rolling_volatility(price_data; window=30);
div_yields = compute_dividend_yields(div_data, price_data);

# ── 2d: Calibrate implied volatility (⚠ SELF-DESIGNED VRP model) ─────────────

sleeves_map = Dict{String, String}(
    all_tickers[i] => sleeves[i] for i in 1:length(all_tickers)
)
rolling_iv = compute_rolling_iv(rolling_vol, sleeves_map);
println("  IV calibration: $(length(rolling_iv)) tickers calibrated (VRP model)")

# ── 2e: Download VIX for reference ──────────────────────────────────────────

vix_data = nothing
try
    vix_raw = download_all_prices(["^VIX"], start_date, end_date)
    if haskey(vix_raw, "^VIX") && nrow(vix_raw["^VIX"]) > 0
        global vix_data = vix_raw["^VIX"]
        println("  VIX data: $(nrow(vix_data)) days, range [$(round(minimum(vix_data.close), digits=1)), $(round(maximum(vix_data.close), digits=1))]")
    end
catch e
    @warn "Could not download VIX: $e"
end

# ── 2f: Load earnings calendar ───────────────────────────────────────────────

earnings_cal = load_earnings_calendar(all_tickers; year=2025);

# ── 2g: Trading days and day-1 prices ────────────────────────────────────────

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

# ── 2h: Output — Rolling Volatility Summary ─────────────────────────────────

println("\n── Rolling Volatility Summary (30-day window, last available date) ──")
vol_summary_df = DataFrame(
    Ticker=String[], Sleeve=String[], RV_Last=Float64[], IV_Last=Float64[],
    RV_Min=Float64[], RV_Max=Float64[], DivYield_q=Float64[]
)
for (i, tk) in enumerate(all_tickers)
    rv_dict = get(rolling_vol, tk, Dict{Date, Float64}())
    iv_dict = get(rolling_iv, tk, Dict{Date, Float64}())
    if !isempty(rv_dict)
        rv_vals = collect(values(rv_dict))
        last_date = maximum(keys(rv_dict))
        push!(vol_summary_df, (
            tk, sleeves[i],
            round(rv_dict[last_date]*100, digits=1),
            round(get(iv_dict, last_date, NaN)*100, digits=1),
            round(minimum(rv_vals)*100, digits=1),
            round(maximum(rv_vals)*100, digits=1),
            round(get(div_yields, tk, 0.0)*100, digits=2)
        ))
    end
end
sort!(vol_summary_df, :RV_Last, rev=true)
pretty_table(vol_summary_df, column_labels=["Ticker", "Sleeve", "RV(%)", "IV(%)", "RV Min(%)", "RV Max(%)", "Div q(%)"])

# ── 2i: Download benchmark (SPY) ────────────────────────────────────────────

benchmark_navs = nothing
try
    global spy_data = download_all_prices(["SPY"], start_date, end_date)
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
println("    Part 3: Running Backtest")
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

# ── 4a: Daily NAV DataFrame ─────────────────────────────────────────────────

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

println("\n── Daily NAV DataFrame (first 5 + last 5 days) ──")
pretty_table(vcat(first(daily_df[:, [:Date, :NAV, :Cash, :OptionMTM, :Delta, :DailyReturn]], 5),
                  last(daily_df[:, [:Date, :NAV, :Cash, :OptionMTM, :Delta, :DailyReturn]], 5)),
    column_labels=["Date", "NAV", "Cash", "Opt MTM", "Delta", "Daily Ret"])

CSV.write(joinpath(_PATH_TO_DATA, "daily_nav_2025.csv"), daily_df)
println("  → Saved to data/daily_nav_2025.csv")

# ── 4b: Per-Ticker Performance DataFrame ────────────────────────────────────

ticker_perf_df = DataFrame(
    Ticker=String[], Sleeve=String[], Sector=String[],
    BlockA_Shares=Int[], Day1_Price=Float64[], LastPrice=Float64[],
    Premium=Float64[], Dividends=Float64[], Costs=Float64[],
    Assigns=Int[], CallAways=Int[], Repairs=Int[], Trades=Int[],
    BlockA_PnL=Float64[]
)
for tk in sort(collect(keys(portfolio.states)))
    st = portfolio.states[tk]
    p1 = get(prices_day1, tk, NaN)
    last_p = NaN
    if haskey(price_data, tk)
        pdf = price_data[tk]
        lp = get_price_on_date(pdf, trading_days[end])
        lp !== nothing && (last_p = lp)
    end
    ba_pnl = st.block_a_shares * (last_p - p1)
    push!(ticker_perf_df, (
        tk, st.sleeve, get(sector_map, tk, "?"),
        st.block_a_shares, round(p1, digits=2), round(last_p, digits=2),
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

println("\n── Per-Ticker Performance ──")
pretty_table(ticker_perf_df,
    column_labels=["Ticker","Sleeve","Sector","BA Shares","Day1 \$","Last \$",
                    "Premium","Divs","Costs","Assigns","CallAways","Repairs","Trades","BA P&L"])

CSV.write(joinpath(_PATH_TO_DATA, "ticker_performance_2025.csv"), ticker_perf_df)
println("  → Saved to data/ticker_performance_2025.csv")

# ── 4c: Monthly Return Summary ──────────────────────────────────────────────

if nrow(daily_df) > 20
    monthly_df = DataFrame(Month=String[], NAV_Start=Float64[], NAV_End=Float64[],
                            Return=Float64[], MaxDD=Float64[], Premium=Float64[])
    ym_vec = [Dates.yearmonth(d) for d in daily_df.Date]
    months = unique(ym_vec)
    for ym in months
        mask = [y == ym for y in ym_vec]
        mdata = daily_df[mask, :]
        nrow(mdata) == 0 && continue
        nav_s = mdata.NAV[1]
        nav_e = mdata.NAV[end]
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
    println("\n── Monthly Return Summary ──")
    pretty_table(monthly_df,
        column_labels=["Month", "NAV Start(M\$)", "NAV End(M\$)", "Return(%)", "MaxDD(%)", "Premium(M\$)"])
end

# ── 4d: Sector Performance ──────────────────────────────────────────────────

sector_perf = combine(groupby(ticker_perf_df, :Sector),
    :Premium => sum => :TotalPremium,
    :Dividends => sum => :TotalDivs,
    :Costs => sum => :TotalCosts,
    :Assigns => sum => :TotalAssigns,
    :Trades => sum => :TotalTrades,
    :BlockA_PnL => sum => :TotalBA_PnL
)
sort!(sector_perf, :TotalPremium, rev=true)
println("\n── Sector Performance ──")
pretty_table(sector_perf,
    column_labels=["Sector", "Premium", "Dividends", "Costs", "Assigns", "Trades", "Block A P&L"])

# ── 4e: Charts ───────────────────────────────────────────────────────────────

if nrow(daily_df) > 5
    dates_plot = daily_df.Date
    navs_plot = daily_df.NAV
    plot_dir = _PATH_TO_DATA

    # ── Chart 1: NAV Curve with Benchmark ────────────────────────────────────

    p1 = plot(dates_plot, navs_plot ./ 1e6,
        title = "Wheel Strategy — 2025 Daily NAV",
        xlabel = "Date", ylabel = "NAV (\$ millions)",
        label = "Wheel Strategy", linewidth = 2, color = :steelblue,
        size = (900, 400), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
    hline!([initial_nav / 1e6], linestyle = :dash, color = :gray, label = "Initial NAV")
    if benchmark_navs !== nothing && length(benchmark_navs) >= nrow(daily_df)
        plot!(dates_plot, benchmark_navs[1:nrow(daily_df)] ./ 1e6,
              label = "SPY Buy-and-Hold", linewidth = 1.5, color = :orange, linestyle = :dash)
    end
    savefig(p1, joinpath(plot_dir, "nav_curve_2025.png"))
    display(p1)

    # ── Chart 2: Drawdown Curve ──────────────────────────────────────────────

    peak_nav = accumulate(max, navs_plot)
    drawdown_pct = (peak_nav .- navs_plot) ./ peak_nav .* 100.0

    p2 = plot(dates_plot, -drawdown_pct,
        title = "Drawdown from Peak",
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
    savefig(p2, joinpath(plot_dir, "drawdown_2025.png"))
    display(p2)

    # ── Chart 3: Cumulative Income (Premium + Dividends - Costs) ─────────────

    p3 = plot(dates_plot, daily_df.CumPremium ./ 1e6,
        title = "Cumulative Income Decomposition",
        xlabel = "Date", ylabel = "\$ millions",
        label = "Premium", linewidth = 2, color = :green,
        size = (900, 400), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
    plot!(dates_plot, daily_df.CumDividends ./ 1e6,
        label = "Dividends", linewidth = 2, color = :blue)
    plot!(dates_plot, daily_df.CumCosts ./ 1e6,
        label = "Trading Costs", linewidth = 2, color = :red, linestyle = :dash)
    net_income = (daily_df.CumPremium .+ daily_df.CumDividends .- daily_df.CumCosts) ./ 1e6
    plot!(dates_plot, net_income,
        label = "Net Income", linewidth = 2.5, color = :black)
    savefig(p3, joinpath(plot_dir, "income_decomposition_2025.png"))
    display(p3)

    # ── Chart 4: Portfolio Greeks Over Time ──────────────────────────────────

    p4a = plot(dates_plot, daily_df.Delta,
        title = "Portfolio Greeks — Delta",
        xlabel = "Date", ylabel = "Delta (shares equivalent)",
        label = "Portfolio Delta", linewidth = 1.5, color = :steelblue,
        size = (900, 300), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
    savefig(p4a, joinpath(plot_dir, "greeks_delta_2025.png"))
    display(p4a)

    p4b = plot(dates_plot, daily_df.Gamma,
        title = "Portfolio Greeks — Gamma",
        xlabel = "Date", ylabel = "Gamma",
        label = "Portfolio Gamma", linewidth = 1.5, color = :purple,
        size = (900, 300), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
    savefig(p4b, joinpath(plot_dir, "greeks_gamma_2025.png"))
    display(p4b)

    p4c = plot(dates_plot, daily_df.Vega,
        title = "Portfolio Greeks — Vega",
        xlabel = "Date", ylabel = "Vega",
        label = "Portfolio Vega", linewidth = 1.5, color = :darkorange,
        size = (900, 300), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
    savefig(p4c, joinpath(plot_dir, "greeks_vega_2025.png"))
    display(p4c)

    # ── Chart 5: NAV Composition (Cash vs Shares vs Options) ─────────────────

    p5 = areaplot(dates_plot,
        [daily_df.Cash ./ 1e6  daily_df.BlockA_Value ./ 1e6  daily_df.SharesValue ./ 1e6],
        title = "NAV Composition Over Time",
        xlabel = "Date", ylabel = "\$ millions",
        label = ["Cash" "Block A (Hold)" "Block B (Shares)"],
        color = [:lightgreen :steelblue :orange],
        size = (900, 400), dpi = 150, alpha = 0.7,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
    savefig(p5, joinpath(plot_dir, "nav_composition_2025.png"))
    display(p5)

    # ── Chart 6: Daily Returns Distribution ──────────────────────────────────

    dr = daily_df.DailyReturn[2:end] .* 100.0
    p6 = histogram(dr,
        title = "Daily Return Distribution",
        xlabel = "Daily Return (%)", ylabel = "Frequency",
        label = "Wheel Returns", bins = 50, color = :steelblue, alpha = 0.7,
        size = (900, 400), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
    vline!([mean(dr)], linewidth = 2, color = :red, label = "Mean=$(round(mean(dr), digits=3))%")
    vline!([0.0], linewidth = 1, color = :black, linestyle = :dash, label = nothing)
    if hasproperty(daily_df, :SPY_Return)
        spy_dr = daily_df.SPY_Return[2:end] .* 100.0
        histogram!(spy_dr, bins = 50, color = :orange, alpha = 0.4, label = "SPY Returns")
    end
    savefig(p6, joinpath(plot_dir, "return_distribution_2025.png"))
    display(p6)

    # ── Chart 7: Rolling Sharpe Ratio (60-day) ───────────────────────────────

    if nrow(daily_df) > 65
        window_sharpe = 60
        roll_sharpe = Float64[]
        all_dr = daily_df.DailyReturn
        for i in (window_sharpe+1):length(all_dr)
            w = all_dr[(i-window_sharpe+1):i]
            s = std(w) > 0 ? (mean(w) * 252) / (std(w) * sqrt(252)) : 0.0
            push!(roll_sharpe, s)
        end
        p7 = plot(dates_plot[(window_sharpe+1):end], roll_sharpe,
            title = "Rolling 60-Day Sharpe Ratio",
            xlabel = "Date", ylabel = "Sharpe Ratio",
            label = "Wheel Strategy", linewidth = 1.5, color = :steelblue,
            size = (900, 300), dpi = 150,
            left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
        hline!([0.0], linestyle = :dash, color = :gray, label = nothing)
        hline!([1.0], linestyle = :dot, color = :green, alpha = 0.5, label = "Sharpe=1.0")
        if hasproperty(daily_df, :SPY_Return)
            spy_sharpe = Float64[]
            spy_dr_all = daily_df.SPY_Return
            for i in (window_sharpe+1):length(spy_dr_all)
                w = spy_dr_all[(i-window_sharpe+1):i]
                s = std(w) > 0 ? (mean(w) * 252) / (std(w) * sqrt(252)) : 0.0
                push!(spy_sharpe, s)
            end
            plot!(dates_plot[(window_sharpe+1):end], spy_sharpe,
                  label = "SPY", linewidth = 1.0, color = :orange, linestyle = :dash)
        end
        savefig(p7, joinpath(plot_dir, "rolling_sharpe_2025.png"))
        display(p7)
    end

    # ── Chart 8: Per-Ticker Premium Bar Chart ────────────────────────────────

    top_n = min(15, nrow(ticker_perf_df))
    top_tickers = ticker_perf_df[1:top_n, :]
    colors_bar = [s == "Safe" ? :steelblue : :orange for s in top_tickers.Sleeve]
    p8 = bar(top_tickers.Ticker, top_tickers.Premium ./ 1e6,
        title = "Top $(top_n) Tickers by Premium Income",
        xlabel = "Ticker", ylabel = "Premium (\$ millions)",
        label = nothing, color = colors_bar, alpha = 0.8,
        size = (900, 400), dpi = 150, rotation = 45,
        left_margin = 10Plots.mm, bottom_margin = 12Plots.mm)
    savefig(p8, joinpath(plot_dir, "premium_by_ticker_2025.png"))
    display(p8)

    # ── Chart 9: Option MTM Over Time ────────────────────────────────────────

    p9 = plot(dates_plot, daily_df.OptionMTM ./ 1e6,
        title = "Short Option Mark-to-Market Liability",
        xlabel = "Date", ylabel = "\$ millions (negative = liability)",
        label = "Option MTM", linewidth = 1.5, color = :red,
        fill = (0, 0.15, :red),
        size = (900, 300), dpi = 150,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
    savefig(p9, joinpath(plot_dir, "option_mtm_2025.png"))
    display(p9)

    # ── Chart 10: Monthly Returns Heatmap-style Bar ──────────────────────────

    if @isdefined(monthly_df) && nrow(monthly_df) > 0
        month_colors = [r >= 0 ? :green : :red for r in monthly_df.Return]
        p10 = bar(monthly_df.Month, monthly_df.Return,
            title = "Monthly Returns (%)",
            xlabel = "Month", ylabel = "Return (%)",
            label = nothing, color = month_colors, alpha = 0.8,
            size = (900, 350), dpi = 150, rotation = 45,
            left_margin = 10Plots.mm, bottom_margin = 12Plots.mm)
        hline!([0.0], linestyle = :dash, color = :black, label = nothing)
        savefig(p10, joinpath(plot_dir, "monthly_returns_2025.png"))
        display(p10)
    end

    println("\n  → All charts saved to data/ folder:")
    println("    1. nav_curve_2025.png         — NAV vs SPY benchmark")
    println("    2. drawdown_2025.png          — Drawdown from peak")
    println("    3. income_decomposition_2025.png — Premium + Dividends - Costs")
    println("    4. greeks_delta/gamma/vega_2025.png — Portfolio Greeks")
    println("    5. nav_composition_2025.png   — Cash vs Block A vs Block B")
    println("    6. return_distribution_2025.png — Daily return histogram")
    println("    7. rolling_sharpe_2025.png    — 60-day rolling Sharpe")
    println("    8. premium_by_ticker_2025.png — Top tickers by premium")
    println("    9. option_mtm_2025.png        — Short option liability")
    println("   10. monthly_returns_2025.png   — Monthly return bars")
end

# ══════════════════════════════════════════════════════════════════════════════
# PART 5 (optional): PARAMETER SWEEP (Varner PDF §6)
# ══════════════════════════════════════════════════════════════════════════════

if RUN_PARAMETER_SWEEP
    println("\n══════════════════════════════════════════════════════")
    println("    Part 5: Parameter Sweep (Varner PDF §6)")
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
        ("Reduce size at earnings",
         WheelConfig(earnings_policy=:reduce_size, earnings_size_reduction=0.50), weights),
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

# ══════════════════════════════════════════════════════════════════════════════
# PART 6 (optional): PORTFOLIO-LEVEL STRESS TESTS (Varner PDF §7B)
# ══════════════════════════════════════════════════════════════════════════════

if RUN_STRESS_TEST
    println("\n══════════════════════════════════════════════════════")
    println("    Part 6: Portfolio-Level Stress Tests (PDF §7B)")
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

println("\n✓ Backtest complete.")
