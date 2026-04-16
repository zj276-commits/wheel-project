"""
calibrate_heston.jl — Modified Heston parameter calibration

Reads option_prices.csv (real market bid/ask), and for each ticker on every
Nth trading day, fits (θ_base, β₁..β₄, γ, κ, σ_v) by minimizing SSE between
model-predicted IV (= √(θ·ψ)) and market-extracted BS implied volatility.

The model:
    IV(K, DTE) = √(θ_base · (1+γ·mood) · ψ(β, DTE, K/S))
where ψ = exp(β₁·ln(DTE) + β₂·ln(K/S) + β₃·ln(DTE)·ln(K/S) + β₄·(ln(K/S))²)

Output: data/heston_params.csv
    ticker, date, theta_base, beta1, beta2, beta3, beta4, gamma, kappa, sigma_v

Usage:
    julia calibrate_heston.jl            # calibrate every 5 days
    julia calibrate_heston.jl 10         # calibrate every 10 days

The backtest loads this CSV at startup — no hardcoded Heston parameters needed.
"""

include(joinpath(@__DIR__, "Include.jl"))
using Random

const OPTION_FILE = joinpath(_PATH_TO_DATA, "option_prices.csv")
const OUTPUT_FILE = joinpath(_PATH_TO_DATA, "heston_params.csv")
const R = 0.045

const TARGET_TICKERS = Set([
    "PEP","KO","PG","JNJ","CME","CMCSA","VZ","T","IBM","MO",
    "PM","MDLZ","EXC","KMB","PAYX","TROW","PFG","SO","DUK","ED",
    "LNT","GIS","CAG","REG","CPB",
    "TSLA","NVDA","AMD","AAPL","MSFT","AMZN","GOOG","META","NFLX","DVN"
])

cal_interval = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5

println("═══════════════════════════════════════════════════")
println("  Modified Heston Parameter Calibration")
println("  Model: IV = √(θ_base · ψ(β, DTE, K/S))")
println("  Input:    $(OPTION_FILE)")
println("  Output:   $(OUTPUT_FILE)")
println("  Interval: every $(cal_interval) trading days")
println("═══════════════════════════════════════════════════\n")

if !isfile(OPTION_FILE)
    error("option_prices.csv not found. Run preprocess_option_data.jl first.")
end

# ── Step 1: Stream option file, group by (ticker, date) ─────────────────────

println("  Step 1: Reading option prices...")

struct OptRecord
    K::Float64
    T::Float64
    mid::Float64
    otype::Symbol
    S_approx::Float64
end

opt_by_ticker_date = Dict{String, Dict{String, Vector{OptRecord}}}()

open(OPTION_FILE, "r") do f
    readline(f)
    lines = 0
    while !eof(f)
        line = readline(f)
        lines += 1
        lines % 1_000_000 == 0 && println("    $(lines ÷ 1_000_000)M lines...")

        fields = split(line, ",")
        length(fields) < 14 && continue

        tk = String(Base.strip(fields[1]))
        tk in TARGET_TICKERS || continue

        date_str = String(Base.strip(fields[2]))
        exdate_str = String(Base.strip(fields[3]))
        cp = String(Base.strip(fields[4]))
        strike_raw = tryparse(Float64, Base.strip(fields[5]))
        bid_raw = tryparse(Float64, Base.strip(fields[6]))
        offer_raw = tryparse(Float64, Base.strip(fields[7]))

        (strike_raw === nothing || bid_raw === nothing || offer_raw === nothing) && continue
        bid_raw <= 0.0 && continue
        offer_raw <= bid_raw && continue

        K = strike_raw / 1000.0
        mid = (bid_raw + offer_raw) / 2.0
        mid <= 0.0 && continue

        y1 = tryparse(Int, date_str[1:4]); y1 === nothing && continue
        m1 = tryparse(Int, date_str[6:7]); m1 === nothing && continue
        d1 = tryparse(Int, date_str[9:10]); d1 === nothing && continue
        y2 = tryparse(Int, exdate_str[1:4]); y2 === nothing && continue
        m2 = tryparse(Int, exdate_str[6:7]); m2 === nothing && continue
        d2 = tryparse(Int, exdate_str[9:10]); d2 === nothing && continue
        dte = Dates.value(Date(y2,m2,d2) - Date(y1,m1,d1))
        (dte < 7 || dte > 180) && continue
        T = dte / 365.0

        otype = cp == "C" ? :call : :put

        if !haskey(opt_by_ticker_date, tk)
            opt_by_ticker_date[tk] = Dict{String, Vector{OptRecord}}()
        end
        if !haskey(opt_by_ticker_date[tk], date_str)
            opt_by_ticker_date[tk][date_str] = OptRecord[]
        end

        push!(opt_by_ticker_date[tk][date_str], OptRecord(K, T, mid, otype, NaN))
    end
    println("    Total lines: $(lines)")
end

println("  Tickers loaded: $(length(opt_by_ticker_date))")

# ── Step 2: For each ticker, rolling calibration ─────────────────────────────

println("\n  Step 2: Rolling calibration (modified Heston model)...\n")

results = NamedTuple{(:ticker,:date,:theta_base,:beta1,:beta2,:beta3,:beta4,:gamma,:kappa,:sigma_v),
                      Tuple{String,String,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}}[]

for tk in sort(collect(keys(opt_by_ticker_date)))
    date_strs = sort(collect(keys(opt_by_ticker_date[tk])))
    n_dates = length(date_strs)

    cal_dates_idx = 1:cal_interval:n_dates
    if isempty(cal_dates_idx)
        println("    $tk: only $(n_dates) dates, skipping")
        continue
    end

    n_cal = 0
    for idx in cal_dates_idx
        ds = date_strs[idx]
        records = opt_by_ticker_date[tk][ds]
        length(records) < 5 && continue

        mids = [r.mid for r in records]
        Ks = [r.K for r in records]
        median_K = median(Ks)

        S_est = median_K

        sample = if length(records) > 40
            records[sort(randperm(length(records))[1:40])]
        else
            records
        end

        opt_data = [(K=r.K, T=r.T, market_price=r.mid, option_type=r.otype) for r in sample]

        try
            #// IVData.jl: Calibrate the Heston model parameters from the option data
            best = calibrate_heston_from_options(S_est, R, opt_data) 
            push!(results, (ticker=tk, date=ds,
                            theta_base=best.θ_base,
                            beta1=best.β[1], beta2=best.β[2],
                            beta3=best.β[3], beta4=best.β[4],
                            gamma=best.γ, kappa=best.κ, sigma_v=best.σ_v))
            n_cal += 1
        catch e
            continue
        end
    end

    if n_cal > 0
        last = results[end]
        println("    $tk: $(n_cal) calibrations over $(n_dates) dates | θ=$(round(last.theta_base, digits=4)) β=[$(round(last.beta1,digits=3)),$(round(last.beta2,digits=3)),$(round(last.beta3,digits=3)),$(round(last.beta4,digits=3))]")
    else
        println("    $tk: ⚠ no valid calibrations")
    end
end

# ── Step 3: Save ─────────────────────────────────────────────────────────────

println("\n  Step 3: Saving results...")

open(OUTPUT_FILE, "w") do f
    write(f, "ticker,date,theta_base,beta1,beta2,beta3,beta4,gamma,kappa,sigma_v\n")
    for r in results
        write(f, "$(r.ticker),$(r.date),$(r.theta_base),$(r.beta1),$(r.beta2),$(r.beta3),$(r.beta4),$(r.gamma),$(r.kappa),$(r.sigma_v)\n")
    end
end

n_tickers = length(unique(r.ticker for r in results))
println("\n  Done! $(length(results)) calibration points for $(n_tickers) tickers")
println("  Saved to: $(OUTPUT_FILE)")
