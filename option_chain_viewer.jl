"""
Interactive Option Chain Viewer — Terminal-based

Displays a real-time option chain table for any selected ticker,
showing Strike, Moneyness, Bid, Mid, Ask, IV, Delta, Gamma, Vega
computed via the Heston stochastic volatility model.

Usage: julia option_chain_viewer.jl
Then type a ticker symbol at the prompt.
"""

include("Include.jl")

const TICKERS = [
    "PEP","KO","PG","JNJ","CME","CMCSA","VZ","T","IBM","MO",
    "PM","MDLZ","EXC","KMB","PAYX","TROW","PFG","SO","DUK","ED",
    "LNT","GIS","CAG","REG","CPB",
    "TSLA","NVDA","AMD","AAPL","MSFT","AMZN","GOOG","META","NFLX","DVN"
]

const SLEEVES = Dict(
    [t => "Safe" for t in TICKERS[1:25]]...,
    [t => "Aggressive" for t in TICKERS[26:end]]...
)

const DELTA_TARGETS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
const R = 0.045

const _VIEWER_HESTON_TS = if isfile(HESTON_CAL_PATH)
    load_heston_params(HESTON_CAL_PATH)
else
    println("  ⚠ No Heston calibration found. Run `julia calibrate_heston.jl` first.")
    Dict{String, Dict{Date, HestonCalibration}}()
end

function load_latest_price(ticker::String)::Float64
    try
        df = download_price_data(ticker, Date(2025, 1, 1), Date(2025, 12, 31); cache_year=2025)
        nrow(df) == 0 && return NaN
        return df.adj_close[end]
    catch
        return NaN
    end
end

function compute_rolling_vol_single(ticker::String; window::Int=30)::Float64
    try
        df = download_price_data(ticker, Date(2025, 1, 1), Date(2025, 12, 31); cache_year=2025)
        nrow(df) < window + 1 && return NaN
        prices = df.adj_close
        log_ret = log.(prices[2:end] ./ prices[1:end-1])
        recent = log_ret[max(1, end-window+1):end]
        return std(recent) * sqrt(252)
    catch
        return NaN
    end
end

function get_stock_rv_regime(ticker::String)::Float64
    try
        df = download_price_data(ticker, Date(2024, 1, 1), Date(2025, 12, 31); cache_year=2025)
        nrow(df) < 60 && return 1.0
        prices = df.adj_close
        window = 30
        rv_series = Float64[]
        for i in (window+1):length(prices)
            lr = log.(prices[(i-window+1):i] ./ prices[(i-window):(i-1)])
            push!(rv_series, std(lr) * sqrt(252))
        end
        isempty(rv_series) && return 1.0
        med = median(rv_series)
        latest = rv_series[end]
        return clamp(latest / max(med, 0.01), 0.5, 2.0)
    catch
        return 1.0
    end
end

function get_dividend_yield(ticker::String)::Float64
    csv_path = joinpath(_PATH_TO_DATA, "dividends_2025", "$ticker.csv")
    if !isfile(csv_path)
        csv_path = joinpath(_PATH_TO_DATA, "dividends_2024", "$ticker.csv")
    end
    !isfile(csv_path) && return 0.0
    df = CSV.read(csv_path, DataFrame)
    nrow(df) == 0 && return 0.0
    S = load_latest_price(ticker)
    isnan(S) && return 0.0
    total_div = sum(df.amount)
    return total_div / S
end

function fmt2(x)
    x isa Number ? string(round(x, digits=2)) : string(x)
end

function display_chain(ticker::String, tenor_days::Int)
    S = load_latest_price(ticker)
    if isnan(S)
        println("  No price data for $ticker")
        return
    end

    sleeve = get(SLEEVES, ticker, "Safe")
    σ_rv = compute_rolling_vol_single(ticker)
    if isnan(σ_rv)
        println("  Insufficient price history for $ticker (need 31+ days for rolling vol)")
        return
    end
    rv_regime = get_stock_rv_regime(ticker)
    q = get_dividend_yield(ticker)
    T = tenor_days / 365.0

    raw_params = lookup_heston_params(_VIEWER_HESTON_TS, ticker, Dates.today())
    if raw_params === nothing
        println("  No Heston calibration for $ticker — run `julia calibrate_heston.jl` first")
        return
    end
    params = rv_adjusted_params(raw_params, rv_regime)

    atm_iv = heston_iv_for_option(S, S, T, R, params; q=q, option_type=:put)

    strikes = Float64[]
    for δ in DELTA_TARGETS
        K = strike_from_delta(S, T, R, atm_iv, δ, :put; q=q)
        push!(strikes, round(K, digits=2))
    end
    extras = [round(S * m, digits=2) for m in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]]
    all_strikes = sort(unique(vcat(strikes, extras)))

    header = ["Strike", "OTM%", "P.Bid", "P.Mid", "P.Ask", "P.IV", "P.Delta",
              "C.Bid", "C.Mid", "C.Ask", "C.IV", "C.Delta", "Gamma", "Vega"]

    data = Matrix{Any}(undef, length(all_strikes), 14)

    for (row, K) in enumerate(all_strikes)
        moneyness = (S - K) / S * 100.0

        iv_put = heston_iv_for_option(S, K, T, R, params; q=q, option_type=:put)
        iv_call = heston_iv_for_option(S, K, T, R, params; q=q, option_type=:call)
        put_price = option_price(S, K, T, R, iv_put, :put; q=q)
        call_price = option_price(S, K, T, R, iv_call, :call; q=q)

        spread_pct = 0.02 + 0.01 * (1.0 - min(abs(moneyness) / 20.0, 1.0))
        put_bid = max(put_price * (1.0 - spread_pct), 0.0)
        put_ask = put_price * (1.0 + spread_pct)
        call_bid = max(call_price * (1.0 - spread_pct), 0.0)
        call_ask = call_price * (1.0 + spread_pct)

        g_put = option_greeks(S, K, T, R, iv_put, :put; N=50, q=q)
        g_call = option_greeks(S, K, T, R, iv_call, :call; N=50, q=q)

        data[row, 1]  = fmt2(K)
        data[row, 2]  = "$(round(moneyness, digits=1))%"
        data[row, 3]  = fmt2(put_bid)
        data[row, 4]  = fmt2(put_price)
        data[row, 5]  = fmt2(put_ask)
        data[row, 6]  = "$(round(iv_put*100, digits=1))%"
        data[row, 7]  = round(g_put.delta, digits=4)
        data[row, 8]  = fmt2(call_bid)
        data[row, 9]  = fmt2(call_price)
        data[row, 10] = fmt2(call_ask)
        data[row, 11] = "$(round(iv_call*100, digits=1))%"
        data[row, 12] = round(g_call.delta, digits=4)
        data[row, 13] = round(g_put.gamma, digits=6)
        data[row, 14] = round(g_put.vega, digits=4)
    end

    println()
    println("="^130)
    println("  $ticker Option Chain | Spot: \$$(round(S, digits=2)) | Tenor: $(tenor_days)d | Sleeve: $sleeve")
    println("  Heston: θ=$(round(raw_params.θ_base, digits=4))→$(round(params.θ_base, digits=4)) (×$(round(rv_regime, digits=2))) β=$(round.(params.β, digits=3)) κ=$(params.κ) σ_v=$(params.σ_v)")
    println("  RV: $(round(σ_rv*100, digits=1))% | RV regime: $(round(rv_regime, digits=2))x | Div Yield: $(round(q*100, digits=2))%")
    println("="^130)

    widths = [9, 7, 8, 8, 8, 7, 8, 8, 8, 8, 7, 8, 10, 8]
    function pad(s, w)
        str = string(s)
        return " "^max(0, w - length(str)) * str
    end

    hdr_line = Base.join([pad(header[i], widths[i]) for i in 1:14], " | ")
    println(hdr_line)
    println("-"^130)

    for row in 1:size(data, 1)
        line = Base.join([pad(data[row, i], widths[i]) for i in 1:14], " | ")
        println(line)
    end
    println()
end

function main_loop()
    println("\n" * "="^60)
    println("  Wheel Strategy -- Interactive Option Chain Viewer")
    println("  Powered by Heston Stochastic Volatility Model")
    println("="^60)
    println("\nAvailable tickers:")
    println("  Safe:       $(Base.join(TICKERS[1:25], ", "))")
    println("  Aggressive: $(Base.join(TICKERS[26:end], ", "))")
    println("\nCommands:")
    println("  <TICKER>        -- show 30-day chain (e.g., AAPL)")
    println("  <TICKER> <DAYS> -- show chain for specific tenor (e.g., AAPL 14)")
    println("  all             -- show all tickers ATM summary")
    println("  quit            -- exit\n")

    while true
        print("chain> ")
        input = Base.strip(readline())
        isempty(input) && continue
        input == "quit" && break
        input == "exit" && break

        try
            parts = Base.split(input)
            ticker = Base.uppercase(parts[1])

            if ticker == "ALL"
                println("\n  ATM IV Summary (30-day, Heston model):")
                for tk in TICKERS
                    S = load_latest_price(tk)
                    isnan(S) && continue
                    σ_rv = compute_rolling_vol_single(tk)
                    isnan(σ_rv) && continue
                    sl = get(SLEEVES, tk, "Safe")
                    raw_p = lookup_heston_params(_VIEWER_HESTON_TS, tk, Dates.today())
                    if raw_p === nothing
                        println("    $tk: ⚠ no calibration"); continue
                    end
                    tk_regime = get_stock_rv_regime(tk)
                    adj_p = rv_adjusted_params(raw_p, tk_regime)
                    atm_iv = heston_iv_for_option(S, S, 30.0/365.0, R, adj_p; option_type=:put)
                    println("    $tk: Spot=\$$(round(S, digits=2))  RV=$(round(σ_rv*100, digits=1))%  HestonIV=$(round(atm_iv*100, digits=1))%  regime=$(round(tk_regime, digits=2))x  [$sl]")
                end
                println()
                continue
            end

            if !(ticker in TICKERS)
                println("  Unknown ticker: $ticker. Type 'all' to see available tickers.")
                continue
            end

            tenor = length(parts) >= 2 ? parse(Int, parts[2]) : 30
            display_chain(ticker, tenor)
        catch e
            println("  Error: $e")
        end
    end

    println("Goodbye!")
end

main_loop()
