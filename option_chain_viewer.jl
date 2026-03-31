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

function load_latest_price(ticker::String)::Float64
    csv_path = joinpath(_PATH_TO_DATA, "prices_2025", "$ticker.csv")
    if !isfile(csv_path)
        csv_path = joinpath(_PATH_TO_DATA, "prices_2024", "$ticker.csv")
    end
    !isfile(csv_path) && return NaN
    df = CSV.read(csv_path, DataFrame)
    nrow(df) == 0 && return NaN
    return df.adj_close[end]
end

function compute_rolling_vol_single(ticker::String; window::Int=30)::Float64
    csv_path = joinpath(_PATH_TO_DATA, "prices_2025", "$ticker.csv")
    if !isfile(csv_path)
        csv_path = joinpath(_PATH_TO_DATA, "prices_2024", "$ticker.csv")
    end
    !isfile(csv_path) && return 0.25
    df = CSV.read(csv_path, DataFrame)
    nrow(df) < window + 1 && return 0.25
    prices = df.adj_close
    log_ret = log.(prices[2:end] ./ prices[1:end-1])
    recent = log_ret[max(1, end-window+1):end]
    return std(recent) * sqrt(252)
end

function get_latest_vix()::Float64
    csv_path = joinpath(_PATH_TO_DATA, "prices_2025", "^VIX.csv")
    !isfile(csv_path) && return 20.0
    df = CSV.read(csv_path, DataFrame)
    nrow(df) == 0 && return 20.0
    return df.close[end]
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
    vix = get_latest_vix()
    q = get_dividend_yield(ticker)
    T = tenor_days / 365.0

    params = calibrate_heston_from_vix(vix, σ_rv; sleeve=sleeve)

    strikes = Float64[]
    for δ in DELTA_TARGETS
        K = strike_from_delta(S, T, R, sqrt(params.v0), δ, :put; q=q)
        push!(strikes, round(K, digits=2))
    end
    extras = [round(S * m, digits=2) for m in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]]
    all_strikes = sort(unique(vcat(strikes, extras)))

    header = ["Strike", "OTM%", "P.Bid", "P.Mid", "P.Ask", "P.IV", "P.Delta",
              "C.Bid", "C.Mid", "C.Ask", "C.IV", "C.Delta", "Gamma", "Vega"]

    data = Matrix{Any}(undef, length(all_strikes), 14)

    for (row, K) in enumerate(all_strikes)
        moneyness = (S - K) / S * 100.0

        put_price = heston_put_price(S, K, T, R, params; q=q)
        call_price = heston_call_price(S, K, T, R, params; q=q)
        iv_put = heston_implied_vol(S, K, T, R, params; q=q, option_type=:put)
        iv_call = heston_implied_vol(S, K, T, R, params; q=q, option_type=:call)

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
    println("  Heston: v0=$(round(params.v0, digits=4)) kappa=$(params.kappa) theta=$(round(params.theta, digits=4)) xi=$(params.xi) rho=$(params.rho)")
    println("  RV: $(round(σ_rv*100, digits=1))% | VIX: $(round(vix, digits=1)) | Div Yield: $(round(q*100, digits=2))%")
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
                vix = get_latest_vix()
                for tk in TICKERS
                    S = load_latest_price(tk)
                    isnan(S) && continue
                    σ_rv = compute_rolling_vol_single(tk)
                    sl = get(SLEEVES, tk, "Safe")
                    params = calibrate_heston_from_vix(vix, σ_rv; sleeve=sl)
                    atm_iv = heston_implied_vol(S, S, 30.0/365.0, R, params; option_type=:put)
                    println("    $tk: Spot=\$$(round(S, digits=2))  RV=$(round(σ_rv*100, digits=1))%  HestonIV=$(round(atm_iv*100, digits=1))%  [$sl]")
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
