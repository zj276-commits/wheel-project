# Backtest.jl — Wheel Strategy Backtest (any year)
#
# Change BACKTEST_YEAR below to run a different calendar year.
# All date ranges, cache directories, and output file names adapt automatically.

include(joinpath(@__DIR__, "Include.jl"));

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURE HERE                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

const BACKTEST_YEAR       = 2025          # ← change to 2024, 2023, etc.
const RUN_PARAMETER_SWEEP = false
const RUN_STRESS_TEST     = false
const RUN_ROBUST_MC       = false        # §7B Monte Carlo robust simulation
const RUN_PAPER_PORTFOLIO = false         # §7C Paper/shadow portfolio validation

# ── Derived constants ────────────────────────────────────────────────────────
const YR        = string(BACKTEST_YEAR)
const start_date = Date(BACKTEST_YEAR, 1, 2)
const end_date   = Date(BACKTEST_YEAR, 12, 31)

# PART 1: UNIVERSE CONSTRUCTION (Varner PDF §3)

println("\n--- Wheel Strategy -- $(YR) Backtest ---")
println("--- Part 1: Universe Construction (Varner PDF §3) ---\n")

# ── 1a: Load volatility data ────────────────────────────────────────────────

sagbm_df = load_sagbm_parameters();
vol_df = select(sagbm_df, :ticker, :volatility);
println("  SAGBM volatility loaded: $(nrow(vol_df)) tickers")

# ── 1b: Load dividend yield from Finviz ─────────────────────────────────────

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

# ── 1c: Universe selection — 25 Safe + 10 Aggressive (manual) ───────────────

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

# ── 1d: Portfolio weights — 60/40 Safe/Aggressive ───────────────────────────

total_nav_initial = 600_000_000.0;
safe_alloc  = 0.60;
aggr_alloc  = 0.40;
max_per_name = 0.05;

safe_weight = safe_alloc / n_safe;
aggr_weight = aggr_alloc / n_aggr;
weights = vcat(fill(safe_weight, n_safe), fill(aggr_weight, n_aggr));

@assert safe_weight <= max_per_name "Safe per-name weight $(round(safe_weight*100,digits=2))% exceeds 5% cap"
@assert aggr_weight <= max_per_name "Aggr per-name weight $(round(aggr_weight*100,digits=2))% exceeds 5% cap"

# ── 1e: Sector map for diversification (PDF §3) ────────────────────────────

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

# ── 1f: Display universe table ──────────────────────────────────────────────

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

# PART 2: DATA LOADING & PREPROCESSING

println("\n--- Part 2: Data Loading & Preprocessing ($(YR)) ---\n")

# ── 2a: SAGBM static volatility (fallback) ─────────────────────────────────

vol_map = Dict{String, Float64}()
for row in eachrow(sagbm_df)
    vol_map[row.ticker] = row.volatility
end

missing_vol = [t for t in all_tickers if !haskey(vol_map, t)]
if !isempty(missing_vol)
    println("  ⚠ Tickers missing from SAGBM: $missing_vol — will compute from downloaded prices")
end

# ── 2b: Download daily price & dividend data ────────────────────────────────

println("  Downloading $(YR) price data...")
price_data = download_all_prices(all_tickers, start_date, end_date; cache_year=BACKTEST_YEAR);
println("  Downloading $(YR) dividend data...")
div_data = download_all_dividends(all_tickers, start_date, end_date; cache_year=BACKTEST_YEAR);

# ── 2c: Trading days and day-1 prices ───────────────────────────────────────

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

# ── 2d: Fill missing vol_map entries from actual downloaded price data ────────
for tk in all_tickers
    haskey(vol_map, tk) && continue
    haskey(price_data, tk) || continue
    df = price_data[tk]
    nrow(df) < 31 && continue
    prices = df.adj_close
    lr = log.(prices[2:end] ./ prices[1:end-1])
    computed_vol = std(lr) * sqrt(252)
    vol_map[tk] = computed_vol
    println("  [computed from prices] $tk → σ = $(round(computed_vol, digits=4))")
end
still_missing = [t for t in all_tickers if !haskey(vol_map, t)]
if !isempty(still_missing)
    error("Cannot proceed: tickers with no volatility data and no price data: $still_missing")
end

# ── 2e: Compute rolling volatility & dividend yields ────────────────────────

rolling_vol = compute_rolling_volatility(price_data; window=30);

# ── 2f: Load Heston calibration (from calibrate_heston.jl) ────────────────

sleeves_map = Dict{String, String}(
    all_tickers[i] => sleeves[i] for i in 1:length(all_tickers)
)

heston_ts = Dict{String, Dict{Date, HestonCalibration}}()
if isfile(HESTON_CAL_PATH)
    heston_ts = load_heston_params(HESTON_CAL_PATH)
    missing_cal = [t for t in all_tickers if !haskey(heston_ts, t)]
    if !isempty(missing_cal)
        @warn "Tickers without Heston calibration: $missing_cal"
    end
else
    println("  heston_params.csv not found — auto-calibrating from JumpHMM + price data...")
    heston_ts = auto_calibrate_heston(price_data, all_tickers; N=5, nu=5.0, tune_jumps=false)
end

# ── 2g: Implied volatility — Heston stochastic volatility model ───────────

println("  Building Heston IV map (per-stock RV-adjusted θ_base)...")
rv_regime_map = compute_stock_rv_regime(rolling_vol)
rolling_iv = build_heston_iv_map(price_data, trading_days;
                                   r=0.045, heston_ts=heston_ts, rolling_vol=rolling_vol)
println("  Heston IV map: $(length(rolling_iv)) tickers total")

# ── 2h: Load earnings calendar ──────────────────────────────────────────────

earnings_cal = load_earnings_calendar(all_tickers; year=BACKTEST_YEAR);

# ── 2i: Output — Rolling Volatility Summary ────────────────────────────────

println("\n-- Rolling Volatility Summary (30-day window, last available date) --")
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
        last_price = get_price_on_date(price_data[tk], last_date)
        ddf = get(div_data, tk, DataFrame(ex_date=Date[], amount=Float64[]))
        q_trail = last_price !== nothing ? trailing_dividend_yield(ddf, last_price, last_date) : 0.0
        push!(vol_summary_df, (
            tk, sleeves[i],
            round(rv_dict[last_date]*100, digits=1),
            round(get(iv_dict, last_date, NaN)*100, digits=1),
            round(minimum(rv_vals)*100, digits=1),
            round(maximum(rv_vals)*100, digits=1),
            round(q_trail*100, digits=2)
        ))
    end
end
sort!(vol_summary_df, :RV_Last, rev=true)
pretty_table(vol_summary_df, column_labels=["Ticker", "Sleeve", "RV(%)", "IV(%)", "RV Min(%)", "RV Max(%)", "Div q(%)"])

# ── 2i: Download benchmark (SPY) ───────────────────────────────────────────

benchmark_navs = nothing
try
    global spy_data = download_all_prices(["SPY"], start_date, end_date; cache_year=BACKTEST_YEAR)
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

# ── 2j: Survivorship Bias & Corporate Action Warnings ──────────────────────

println("\n⚠ SURVIVORSHIP BIAS WARNING:")
println("  Universe is hand-picked from current S&P 500 constituents.")
println("  Delisted, acquired, or bankrupt names not included.")
println("  True out-of-sample testing requires a point-in-time constituent list.\n")

split_events = detect_stock_splits(price_data)
if !isempty(split_events)
    println("⚠ CORPORATE ACTION DETECTED:")
    for (tk, events) in split_events
        for ev in events
            println("  $tk -- probable split/corp action on $(ev.date), ratio ~ $(ev.ratio)")
        end
    end
    println("  Option strikes and lot sizes may need adjustment around these dates.\n")
else
    println("  No stock splits detected in $(length(all_tickers)) tickers.\n")
end

# ── Filter out tickers with no price data ─────────────────────────────────────
missing_tickers = [t for t in all_tickers if !haskey(prices_day1, t)]
if !isempty(missing_tickers)
    @warn "Excluding tickers with no price data: $missing_tickers"
    valid_mask = [haskey(prices_day1, t) for t in all_tickers]
    all_tickers = all_tickers[valid_mask]
    sleeves     = sleeves[valid_mask]
    weights     = weights[valid_mask]
    weights    ./= sum(weights)
    sleeves_map = Dict(all_tickers[i] => sleeves[i] for i in 1:length(all_tickers))
end

# PART 3: RUN BACKTEST

println("\n--- Part 3: Running $(YR) Backtest ---\n")

initial_nav = total_nav_initial
config = default_config()

portfolio = initialize_portfolio(all_tickers, sleeves, weights, initial_nav,
                                  prices_day1, config);

println("  Initial cash: \$$(round(Int, portfolio.cash))")
println("  Running $(length(trading_days)) trading days...\n")

run_backtest!(portfolio, price_data, div_data, vol_map, trading_days;
              earnings_cal=earnings_cal, rolling_vol=rolling_vol,
              sector_map=sector_map, div_yields=div_yields,
              rolling_iv=rolling_iv, heston_ts=heston_ts);

# PART 4: RESULTS & REPORTING

generate_report(portfolio; benchmark_navs=benchmark_navs, benchmark_label="SPY")

# ── 4a: Daily NAV DataFrame ────────────────────────────────────────────────

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

println("\n-- Daily NAV DataFrame (first 5 + last 5 days) --")
pretty_table(vcat(first(daily_df[:, [:Date, :NAV, :Cash, :OptionMTM, :Delta, :DailyReturn]], 5),
                  last(daily_df[:, [:Date, :NAV, :Cash, :OptionMTM, :Delta, :DailyReturn]], 5)),
column_labels=["Date", "NAV", "Cash", "Opt MTM", "Delta", "Daily Ret"])

CSV.write(joinpath(_PATH_TO_DATA, "daily_nav_$(YR).csv"), daily_df)
println("  -> Saved to data/daily_nav_$(YR).csv")

# ── 4b: Per-Ticker Performance DataFrame ───────────────────────────────────

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
        lp = get_price_on_date(price_data[tk], trading_days[end])
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

println("\n-- Per-Ticker Performance --")
pretty_table(ticker_perf_df,    column_labels=["Ticker","Sleeve","Sector","BA Shares","Day1 \$","Last \$",
                    "Premium","Divs","Costs","Assigns","CallAways","Repairs","Trades","BA P&L"])

CSV.write(joinpath(_PATH_TO_DATA, "ticker_performance_$(YR).csv"), ticker_perf_df)
println("  -> Saved to data/ticker_performance_$(YR).csv")

# ── 4c: Monthly Return Summary ─────────────────────────────────────────────

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
    println("\n-- Monthly Return Summary ($(YR)) --")
    pretty_table(monthly_df,        column_labels=["Month", "NAV Start(M\$)", "NAV End(M\$)", "Return(%)", "MaxDD(%)", "Premium(M\$)"])
end

# ── 4d: Sector Performance ─────────────────────────────────────────────────

sector_perf = combine(groupby(ticker_perf_df, :Sector),
    :Premium => sum => :TotalPremium,
    :Dividends => sum => :TotalDivs,
    :Costs => sum => :TotalCosts,
    :Assigns => sum => :TotalAssigns,
    :Trades => sum => :TotalTrades,
    :BlockA_PnL => sum => :TotalBA_PnL
)
sort!(sector_perf, :TotalPremium, rev=true)
println("\n-- Sector Performance --")
pretty_table(sector_perf,    column_labels=["Sector", "Premium", "Dividends", "Costs", "Assigns", "Trades", "Block A P&L"])

# ── 4e: Charts — Academic style (Alswaidan-Varner paper) ─────────────────────
# White background, thin gridlines, sans-serif titles, muted palette,
# tight margins — matching figures in HMM-Modeling-Equity.pdf.

const _CHART_DEFAULTS = (
    background_color = :white,
    foreground_color = :black,
    grid = true,
    gridalpha = 0.3,
    gridlinewidth = 0.5,
    framestyle = :box,
    tickfontsize = 8,
    guidefontsize = 10,
    titlefontsize = 11,
    legendfontsize = 8,
    dpi = 200,
    size = (960, 380),
    left_margin = 12Plots.mm,
    bottom_margin = 10Plots.mm,
    top_margin = 5Plots.mm,
    right_margin = 5Plots.mm,
)

if nrow(daily_df) > 5
    dates_plot = daily_df.Date
    navs_plot = daily_df.NAV
    plot_dir = _PATH_TO_DATA

    p1 = plot(dates_plot, navs_plot ./ 1e6;
        _CHART_DEFAULTS...,
        title = "Figure 1: Portfolio NAV — Wheel Strategy vs SPY ($(YR))",
        xlabel = "Date", ylabel = "NAV (\$ millions)",
        label = "Wheel Strategy", linewidth = 2, color = RGB(0.2, 0.4, 0.7))
    hline!([initial_nav / 1e6], linestyle = :dash, color = :gray60, label = "Initial NAV", linewidth = 1)
    if benchmark_navs !== nothing && length(benchmark_navs) >= nrow(daily_df)
        plot!(dates_plot, benchmark_navs[1:nrow(daily_df)] ./ 1e6,
              label = "SPY (Buy-and-Hold)", linewidth = 1.5, color = RGB(0.85, 0.45, 0.15), linestyle = :dash)
    end
    savefig(p1, joinpath(plot_dir, "nav_curve_$(YR).png"))
    display(p1)

    peak_nav = accumulate(max, navs_plot)
    drawdown_pct = (peak_nav .- navs_plot) ./ peak_nav .* 100.0

    p2 = plot(dates_plot, -drawdown_pct;
        _CHART_DEFAULTS..., size = (960, 300),
        title = "Figure 2: Drawdown from Peak ($(YR))",
        xlabel = "Date", ylabel = "Drawdown (%)",
        label = "Wheel Strategy", linewidth = 1.5, color = RGB(0.8, 0.15, 0.15),
        fill = (0, 0.15, RGB(0.8, 0.15, 0.15)))
    if benchmark_navs !== nothing && length(benchmark_navs) >= nrow(daily_df)
        bm = benchmark_navs[1:nrow(daily_df)]
        bm_peak = accumulate(max, bm)
        bm_dd = (bm_peak .- bm) ./ bm_peak .* 100.0
        plot!(dates_plot, -bm_dd, label = "SPY", linewidth = 1.0, color = RGB(0.85, 0.45, 0.15), linestyle = :dash)
    end
    savefig(p2, joinpath(plot_dir, "drawdown_$(YR).png"))
    display(p2)

    p3 = plot(dates_plot, daily_df.CumPremium ./ 1e6;
        _CHART_DEFAULTS...,
        title = "Figure 3: Cumulative Income Decomposition ($(YR))",
        xlabel = "Date", ylabel = "\$ millions",
        label = "Premium", linewidth = 2, color = RGB(0.2, 0.6, 0.3))
    plot!(dates_plot, daily_df.CumDividends ./ 1e6,
        label = "Dividends", linewidth = 2, color = RGB(0.2, 0.4, 0.7))
    plot!(dates_plot, daily_df.CumCosts ./ 1e6,
        label = "Trading Costs", linewidth = 1.5, color = RGB(0.8, 0.15, 0.15), linestyle = :dash)
    net_income = (daily_df.CumPremium .+ daily_df.CumDividends .- daily_df.CumCosts) ./ 1e6
    plot!(dates_plot, net_income,
        label = "Net Income", linewidth = 2.5, color = :black)
    savefig(p3, joinpath(plot_dir, "income_decomposition_$(YR).png"))
    display(p3)

    p4a = plot(dates_plot, daily_df.Delta;
        _CHART_DEFAULTS..., size = (960, 300),
        title = "Figure 4a: Portfolio Delta ($(YR))",
        xlabel = "Date", ylabel = "Delta (normalized, 1.0 = fully long)",
        label = "Portfolio Δ", linewidth = 1.5, color = RGB(0.2, 0.4, 0.7))
    hline!([0.0], linestyle = :dot, color = :gray50, label = nothing, linewidth = 0.8)
    savefig(p4a, joinpath(plot_dir, "greeks_delta_$(YR).png"))
    display(p4a)

    p4b = plot(dates_plot, daily_df.Gamma;
        _CHART_DEFAULTS..., size = (960, 300),
        title = "Figure 4b: Portfolio Gamma ($(YR))",
        xlabel = "Date", ylabel = "Gamma (per share)",
        label = "Portfolio Γ", linewidth = 1.5, color = RGB(0.55, 0.25, 0.65))
    hline!([0.0], linestyle = :dot, color = :gray50, label = nothing, linewidth = 0.8)
    savefig(p4b, joinpath(plot_dir, "greeks_gamma_$(YR).png"))
    display(p4b)

    p4c = plot(dates_plot, daily_df.Vega;
        _CHART_DEFAULTS..., size = (960, 300),
        title = "Figure 4c: Portfolio Vega ($(YR))",
        xlabel = "Date", ylabel = "Vega (per share)",
        label = "Portfolio ν", linewidth = 1.5, color = RGB(0.85, 0.45, 0.15))
    hline!([0.0], linestyle = :dot, color = :gray50, label = nothing, linewidth = 0.8)
    savefig(p4c, joinpath(plot_dir, "greeks_vega_$(YR).png"))
    display(p4c)

    p5 = areaplot(dates_plot,
        [daily_df.Cash ./ 1e6  daily_df.BlockA_Value ./ 1e6  daily_df.SharesValue ./ 1e6];
        _CHART_DEFAULTS...,
        title = "Figure 5: NAV Composition ($(YR))",
        xlabel = "Date", ylabel = "\$ millions",
        label = ["Cash" "Block A (Hold)" "Block B (Shares)"],
        color = [RGB(0.75, 0.88, 0.75) RGB(0.2, 0.4, 0.7) RGB(0.85, 0.45, 0.15)],
        alpha = 0.7)
    savefig(p5, joinpath(plot_dir, "nav_composition_$(YR).png"))
    display(p5)

    dr = daily_df.DailyReturn[2:end] .* 100.0
    p6 = histogram(dr;
        _CHART_DEFAULTS...,
        title = "Figure 6: Daily Return Distribution ($(YR))",
        xlabel = "Daily Return (%)", ylabel = "Frequency",
        label = "Wheel Returns", bins = 60, color = RGB(0.2, 0.4, 0.7), alpha = 0.7)
    vline!([mean(dr)], linewidth = 2, color = RGB(0.8, 0.15, 0.15),
           label = "Mean = $(round(mean(dr), digits=3))%")
    vline!([0.0], linewidth = 1, color = :black, linestyle = :dash, label = nothing)
    if hasproperty(daily_df, :SPY_Return)
        spy_dr = daily_df.SPY_Return[2:end] .* 100.0
        histogram!(spy_dr, bins = 60, color = RGB(0.85, 0.45, 0.15), alpha = 0.35, label = "SPY Returns")
    end
    savefig(p6, joinpath(plot_dir, "return_distribution_$(YR).png"))
    display(p6)

    if nrow(daily_df) > 65
        window_sharpe = 60
        roll_sharpe = Float64[]
        all_dr = daily_df.DailyReturn
        for i in (window_sharpe+1):length(all_dr)
            w = all_dr[(i-window_sharpe+1):i]
            s = std(w) > 0 ? (mean(w) * 252) / (std(w) * sqrt(252)) : 0.0
            push!(roll_sharpe, s)
        end
        p7 = plot(dates_plot[(window_sharpe+1):end], roll_sharpe;
            _CHART_DEFAULTS..., size = (960, 300),
            title = "Figure 7: Rolling 60-Day Sharpe Ratio ($(YR))",
            xlabel = "Date", ylabel = "Sharpe Ratio",
            label = "Wheel Strategy", linewidth = 1.5, color = RGB(0.2, 0.4, 0.7))
        hline!([0.0], linestyle = :dash, color = :gray60, label = nothing, linewidth = 0.8)
        hline!([1.0], linestyle = :dot, color = RGB(0.2, 0.6, 0.3), alpha = 0.6, label = "Sharpe = 1.0")
        if hasproperty(daily_df, :SPY_Return)
            spy_sharpe = Float64[]
            spy_dr_all = daily_df.SPY_Return
            for i in (window_sharpe+1):length(spy_dr_all)
                w = spy_dr_all[(i-window_sharpe+1):i]
                s = std(w) > 0 ? (mean(w) * 252) / (std(w) * sqrt(252)) : 0.0
                push!(spy_sharpe, s)
            end
            plot!(dates_plot[(window_sharpe+1):end], spy_sharpe,
                  label = "SPY", linewidth = 1.0, color = RGB(0.85, 0.45, 0.15), linestyle = :dash)
        end
        savefig(p7, joinpath(plot_dir, "rolling_sharpe_$(YR).png"))
        display(p7)
    end

    top_n = min(15, nrow(ticker_perf_df))
    top_tickers = ticker_perf_df[1:top_n, :]
    colors_bar = [s == "Safe" ? RGB(0.2, 0.4, 0.7) : RGB(0.85, 0.45, 0.15) for s in top_tickers.Sleeve]
    p8 = bar(top_tickers.Ticker, top_tickers.Premium ./ 1e6;
        _CHART_DEFAULTS...,
        title = "Figure 8: Premium Income by Ticker ($(YR))",
        xlabel = "Ticker", ylabel = "Premium (\$ millions)",
        label = nothing, color = colors_bar, alpha = 0.85, rotation = 45,
        bottom_margin = 14Plots.mm)
    savefig(p8, joinpath(plot_dir, "premium_by_ticker_$(YR).png"))
    display(p8)

    p9 = plot(dates_plot, daily_df.OptionMTM ./ 1e6;
        _CHART_DEFAULTS..., size = (960, 300),
        title = "Figure 9: Short Option Mark-to-Market ($(YR))",
        xlabel = "Date", ylabel = "\$ millions (negative = liability)",
        label = "Option MTM", linewidth = 1.5, color = RGB(0.8, 0.15, 0.15),
        fill = (0, 0.12, RGB(0.8, 0.15, 0.15)))
    savefig(p9, joinpath(plot_dir, "option_mtm_$(YR).png"))
    display(p9)

    if @isdefined(monthly_df) && nrow(monthly_df) > 0
        month_colors = [r >= 0 ? RGB(0.2, 0.6, 0.3) : RGB(0.8, 0.15, 0.15) for r in monthly_df.Return]
        p10 = bar(monthly_df.Month, monthly_df.Return;
            _CHART_DEFAULTS..., size = (960, 340),
            title = "Figure 10: Monthly Returns ($(YR))",
            xlabel = "Month", ylabel = "Return (%)",
            label = nothing, color = month_colors, alpha = 0.85, rotation = 45,
            bottom_margin = 14Plots.mm)
        hline!([0.0], linestyle = :dash, color = :black, label = nothing, linewidth = 0.8)
        savefig(p10, joinpath(plot_dir, "monthly_returns_$(YR).png"))
        display(p10)
    end

    println("\n  -> All charts saved to data/ folder (academic style)")
end

# PART 5 (optional): PARAMETER SWEEP (Varner PDF §6)

if RUN_PARAMETER_SWEEP
    println("\n--- Part 5: Parameter Sweep (Varner PDF §6) ---\n")

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
                      rolling_iv=rolling_iv, heston_ts=heston_ts)

        recs = pf.daily_records
        isempty(recs) && continue
        fin = recs[end].nav
        ret = (fin - initial_nav) / initial_nav * 100.0
        navs = [r.nav for r in recs]
        local dr = diff(log.(navs))
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

    pretty_table(sweep_results,        column_labels=["Configuration", "Final NAV (M\$)", "Return %",
                        "Sharpe", "Sortino", "MaxDD %", "Premium (M\$)",
                        "Assigns", "CallAways", "Trades"])
end

# PART 6 (optional): PORTFOLIO-LEVEL STRESS TESTS (Varner PDF §7B)

if RUN_STRESS_TEST
    run_portfolio_stress_tests(;
        price_data, div_data, vol_map,
        all_tickers, sleeves, weights, sleeves_map,
        initial_nav, prices_day1, config,
        earnings_cal, sector_map,
        safe_tickers, aggressive_tickers,
        chart_defaults=_CHART_DEFAULTS,
        heston_ts=heston_ts, rolling_vol=rolling_vol)
end

# PART 7 (optional): MONTE CARLO ROBUST SIMULATION (Varner PDF §7B)

if RUN_ROBUST_MC
    run_robust_mc_simulation(;
        price_data, div_data, vol_map,
        all_tickers, sleeves, weights, sleeves_map,
        initial_nav, prices_day1, config,
        earnings_cal, sector_map,
        trading_days, daily_df,
        safe_tickers, aggressive_tickers,
        chart_defaults=_CHART_DEFAULTS, yr=YR,
        heston_ts=heston_ts, rolling_vol=rolling_vol,
        n_mc_runs=20)
end

# PART 8 (optional): PAPER/SHADOW PORTFOLIO VALIDATION (Varner PDF §7C)

if RUN_PAPER_PORTFOLIO
    run_paper_portfolio_validation(;
        price_data, div_data, vol_map,
        all_tickers, sleeves, weights,
        initial_nav, config,
        earnings_cal, sector_map,
        rolling_vol, rolling_iv,
        trading_days, daily_df,
        portfolio,
        chart_defaults=_CHART_DEFAULTS, yr=YR,
        heston_ts=heston_ts)
end

println("\n Done. $(YR) Backtest complete.")
