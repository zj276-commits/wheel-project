# calibrate_from_market.jl
# Calibrate Heston IV model using real market option prices (OptionMetrics data)
#
# Pipeline:
#   1. Load historical prices (2014-2024 training + 2025 testing)
#   2. Fit JumpHMM per ticker → decode HMM states per day
#   3. Stream the 8 GB option CSV → filter for our 35 tickers, puts, valid IV
#   4. Build CalibrationData per ticker (market IV + HMM states + moods)
#   5. Call calibrate() (Nelder-Mead) → (HestonParameters, ThetaHybrid) per ticker
#   6. For each 2025 trading day, map HMM state → θ_base and save heston_params.csv

include(joinpath(@__DIR__, "Include.jl"))

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const OPTION_CSV   = raw"c:\Users\aaron\Desktop\Cornell Documents\2014-2026 option price.csv"
const OUTPUT_PATH  = joinpath(_PATH_TO_DATA, "heston_params.csv")
const N_HMM        = 5
const BACKTEST_YEAR = 2025

const TICKERS = [
    "PEP","KO","PG","JNJ","CME","CMCSA","VZ","T","IBM","MO",
    "PM","MDLZ","EXC","KMB","PAYX","TROW","PFG","SO","DUK","ED",
    "LNT","GIS","CAG","REG","CPB",
    "TSLA","NVDA","AMD","AAPL","MSFT","AMZN","GOOG","META","NFLX","DVN",
]

# OptionMetrics uses historical tickers; map old → current
const TICKER_MAP = Dict{String,String}(
    "FB" => "META",
    "GOOGL" => "GOOG",
)

const TICKER_SET = Set(TICKERS)
const REVERSE_TICKER_MAP = Dict(v => k for (k, v) in TICKER_MAP)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Load price data
# ═══════════════════════════════════════════════════════════════════════════════

println("\n═══ Phase 1: Loading price data ═══\n")

println("  Loading training data (2014-2024)...")
training_prices = download_all_prices(TICKERS, Date(2014,1,1), Date(2024,12,31); cache_year=2024)

println("\n  Loading testing data (2025)...")
testing_prices = download_all_prices(TICKERS, Date(2025,1,2), Date(2025,12,31); cache_year=2025)

merged_prices = Dict{String, DataFrame}()
for tk in TICKERS
    dfs = DataFrame[]
    haskey(training_prices, tk) && push!(dfs, training_prices[tk])
    haskey(testing_prices, tk) && push!(dfs, testing_prices[tk])
    if !isempty(dfs)
        combined = vcat(dfs...)
        sort!(combined, :date)
        unique!(combined, :date)
        merged_prices[tk] = combined
    end
end
println("\n  Merged prices: $(length(merged_prices)) tickers")
for (tk, df) in merged_prices
    println("    $tk: $(nrow(df)) days [$(minimum(df.date)) — $(maximum(df.date))]")
end

price_lookup = Dict{String, Dict{Date, Float64}}()
for (tk, df) in merged_prices
    price_lookup[tk] = Dict(row.date => Float64(row.adj_close) for row in eachrow(df))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Fit JumpHMM per ticker
# ═══════════════════════════════════════════════════════════════════════════════

println("\n═══ Phase 2: Fitting JumpHMM (N=$N_HMM) ═══\n")

hmm_models = Dict{String, Any}()
hmm_state_lookup = Dict{String, Dict{Date, Int}}()

for tk in TICKERS
    !haskey(merged_prices, tk) && continue
    df = merged_prices[tk]
    nrow(df) < 61 && (@warn "Skipping $tk: only $(nrow(df)) price days"; continue)

    prices = Float64.(df.adj_close)
    model = fit_jumphmm(prices; N=N_HMM, nu=5.0)
    hmm_models[tk] = model

    G = excess_growth_rates(prices; rf=model.rf, dt=model.dt)
    decoded = assign_states(model.partition, G)

    state_dict = Dict{Date, Int}()
    dates = df.date
    for i in 1:length(decoded)
        state_dict[dates[i + 1]] = decoded[i]
    end
    hmm_state_lookup[tk] = state_dict
    println("  $tk: $(length(decoded)) state assignments")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Mood is fixed at 0.0
# ═══════════════════════════════════════════════════════════════════════════════
#
# The Wheel backtest uses lookup_heston_params which sets mood=0.0.
# So we must calibrate with mood=0.0 for consistency.
# Otherwise θ_states are biased low by the factor (1+γ·avg_mood).

println("\n═══ Phase 3: Mood fixed at 0.0 (consistent with backtest) ═══\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Stream option CSV (8 GB) and extract calibration observations
# ═══════════════════════════════════════════════════════════════════════════════

println("\n═══ Phase 4: Streaming option CSV ═══\n")
println("  File: $OPTION_CSV")

struct OptionObs
    strike::Float64
    dte::Int
    iv::Float64
    spot::Float64
    state::Int
    mood::Float64
end

const CACHE_PATH = joinpath(_PATH_TO_DATA, "option_obs_cache.jld2")

ticker_obs = Dict{String, Vector{OptionObs}}()

if isfile(CACHE_PATH)
    println("  Loading cached option observations from $CACHE_PATH ...")
    cached = load(CACHE_PATH)
    for tk in keys(cached)
        tk_data = cached[tk]
        ticker_obs[tk] = [OptionObs(tk_data[:strike][i], tk_data[:dte][i],
                                     tk_data[:iv][i], tk_data[:spot][i],
                                     tk_data[:state][i], 0.0)  # mood = 0.0
                          for i in 1:length(tk_data[:strike])]
    end
    println("  Loaded $(sum(length(v) for v in values(ticker_obs))) observations for $(length(ticker_obs)) tickers")
else
    println("  Scanning $OPTION_CSV (first run — will cache for future use)...")

    global n_read = 0
    global n_kept = 0

    open(OPTION_CSV, "r") do io
        readline(io)  # skip header

        while !eof(io)
            line = readline(io)
            global n_read += 1

            if n_read % 10_000_000 == 0
                println("    ... $(n_read ÷ 1_000_000)M lines, $(n_kept) kept")
            end

            fields = split(line, ',')
            length(fields) < 28 && continue

            raw_ticker = String(fields[28])
            ticker = get(TICKER_MAP, raw_ticker, raw_ticker)
            ticker ∈ TICKER_SET || continue

            String(fields[7]) == "P" || continue

            iv_str = String(fields[13])
            isempty(iv_str) && continue
            iv = tryparse(Float64, iv_str)
            iv === nothing && continue
            (iv <= 0.01 || iv >= 5.0) && continue

            d = tryparse(Date, String(fields[2]))
            exd = tryparse(Date, String(fields[5]))
            (d === nothing || exd === nothing) && continue

            dte = Dates.value(exd - d)
            (dte < 7 || dte > 180) && continue

            strike_raw = tryparse(Float64, String(fields[8]))
            strike_raw === nothing && continue
            strike = strike_raw / 1000.0

            !haskey(price_lookup, ticker) && continue
            spot = get(price_lookup[ticker], d, 0.0)
            spot <= 0.0 && continue

            moneyness = strike / spot
            (moneyness < 0.80 || moneyness > 1.10) && continue

            !haskey(hmm_state_lookup, ticker) && continue
            state = get(hmm_state_lookup[ticker], d, 0)
            state == 0 && continue

            if !haskey(ticker_obs, ticker)
                ticker_obs[ticker] = OptionObs[]
            end
            push!(ticker_obs[ticker], OptionObs(strike, dte, iv, spot, state, 0.0))
            global n_kept += 1
        end
    end

    println("\n  Done: $(n_read) lines scanned, $(n_kept) observations kept")
    for tk in sort(collect(keys(ticker_obs)))
        println("    $tk: $(length(ticker_obs[tk])) observations")
    end

    println("  Caching observations to $CACHE_PATH ...")
    cache_dict = Dict{String, Any}()
    for (tk, obs) in ticker_obs
        cache_dict[tk] = Dict(
            :strike => [o.strike for o in obs],
            :dte    => [o.dte for o in obs],
            :iv     => [o.iv for o in obs],
            :spot   => [o.spot for o in obs],
            :state  => [o.state for o in obs],
        )
    end
    save(CACHE_PATH, cache_dict)
    println("  Cached.")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5A — Shared β calibration on reference tickers (Varner 2025 approach)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Paper methodology: calibrate β₁...β₅ on a pooled dataset from high-liquidity
# tickers, with per-ticker θ_states as nuisance parameters. The shared β
# captures the universal smile/term-structure shape; only the scalar θ level
# varies across tickers (Table 3 in Varner 2025).
#
# κ=5.0, σ_v=0.3, γ=0.0 are FIXED (not optimized), since:
#   - κ/σ_v are unidentifiable from the IV = √θ equilibrium formula
#   - γ is set to 0.0 for consistency with the backtest (mood=0.0)

println("\n═══ Phase 5A: Shared β calibration (paper methodology) ═══\n")

const FIXED_κ  = 5.0
const FIXED_σv = 0.3
const FIXED_γ  = 0.0

# Select reference tickers: top by observation count (most liquid options)
obs_counts = [(tk, length(ticker_obs[tk])) for tk in keys(ticker_obs)]
sort!(obs_counts, by=x -> x[2], rev=true)
ref_tickers = [x[1] for x in obs_counts[1:min(8, length(obs_counts))]]
println("  Reference tickers: $(ref_tickers)")

# Subsample each reference ticker to cap total observations
const MAX_OBS_PER_REF = 15_000
ref_obs = Dict{String, Vector{OptionObs}}()
for tk in ref_tickers
    obs = ticker_obs[tk]
    if length(obs) > MAX_OBS_PER_REF
        step = length(obs) ÷ MAX_OBS_PER_REF
        obs = obs[1:step:end]
    end
    ref_obs[tk] = obs
    println("    $tk: $(length(obs)) observations (subsampled)")
end

# Build pooled calibration data with ticker index
# Parameter vector layout:
#   [1:5]            = β₁...β₅ (shared)
#   [6:5+N*n_ref]    = log(θ_states) for each ref ticker

n_ref = length(ref_tickers)
n_β = 5
n_params_A = n_β + N_HMM * n_ref

# Flatten observations with ticker mapping
pooled_strikes  = Float64[]
pooled_dtes     = Int[]
pooled_ivs      = Float64[]
pooled_spots    = Float64[]
pooled_states   = Int[]
pooled_tk_idx   = Int[]

for (idx, tk) in enumerate(ref_tickers)
    for o in ref_obs[tk]
        push!(pooled_strikes, o.strike)
        push!(pooled_dtes, o.dte)
        push!(pooled_ivs, o.iv)
        push!(pooled_spots, o.spot)
        push!(pooled_states, o.state)
        push!(pooled_tk_idx, idx)
    end
end

n_pooled = length(pooled_ivs)
println("  Total pooled observations: $n_pooled")

# Initialize θ per (ticker, state) from average IV² in each group
θ_inits = fill(0.04, N_HMM, n_ref)
for i in 1:n_pooled
    tk_idx = pooled_tk_idx[i]
    s = pooled_states[i]
    if 1 <= s <= N_HMM
        θ_inits[s, tk_idx] = pooled_ivs[i]^2
    end
end

x0_A = Vector{Float64}(undef, n_params_A)
x0_A[1] = -0.3   # β₁ initial (paper: -0.622)
x0_A[2] = -2.0   # β₂ initial (paper: -8.448)
x0_A[3] =  0.3   # β₃ initial (paper: 1.617)
x0_A[4] =  0.0   # β₄ initial (paper: -1.242)
x0_A[5] =  0.1   # β₅ initial (paper: 0.150)
for r in 1:n_ref
    for s in 1:N_HMM
        x0_A[n_β + (r-1)*N_HMM + s] = log(max(θ_inits[s, r], 1e-8))
    end
end

function objective_shared(x)
    β = x[1:n_β]

    total_error = 0.0
    for i in 1:n_pooled
        tk_idx = pooled_tk_idx[i]
        s = pooled_states[i]
        (s < 1 || s > N_HMM) && continue

        θ_s = exp(x[n_β + (tk_idx-1)*N_HMM + s])
        DTE = Float64(pooled_dtes[i])
        moneyness = pooled_strikes[i] / pooled_spots[i]

        ψ_val = ψ(β, DTE, moneyness)
        σ_model = sqrt(max(θ_s * ψ_val, 1e-10))
        total_error += (σ_model - pooled_ivs[i])^2
    end

    # Soft bounds on β (per paper Table 4 ranges)
    penalty = 0.0
    penalty += max(0.0, abs(β[1]) - 1.0)^2 * 10.0      # β₁ ∈ [-1, 1]
    penalty += max(0.0, abs(β[2]) - 12.0)^2 * 10.0      # β₂ ∈ [-12, 12]
    penalty += max(0.0, -β[3])^2 * 10.0                  # β₃ ≥ 0 (paper: skew flattens)
    penalty += max(0.0, abs(β[4]) - 3.0)^2 * 10.0        # β₄ ∈ [-3, 3]
    penalty += max(0.0, abs(β[5]) - 0.5)^2 * 10.0        # β₅ ∈ [-0.5, 0.5]

    return total_error / n_pooled + penalty
end

println("  Optimizing $(n_params_A) parameters (5 β + $(N_HMM)×$(n_ref) θ_states)...")
result_A = optimize(objective_shared, x0_A, NelderMead(),
                    Optim.Options(iterations=20000, show_trace=false))

x_A = Optim.minimizer(result_A)
shared_β = x_A[1:n_β]

println("\n  ═══ Shared β (Varner 2025 eq. 11) ═══")
println("    β₁ (DTE decay)      = $(round(shared_β[1], digits=4))")
println("    β₂ (skew)           = $(round(shared_β[2], digits=4))")
println("    β₃ (DTE×skew)       = $(round(shared_β[3], digits=4))")
println("    β₄ (smile curvature)= $(round(shared_β[4], digits=4))")
println("    β₅ (DTE curvature)  = $(round(shared_β[5], digits=4))")
println("    Final loss: $(round(Optim.minimum(result_A), digits=6))")

# Print reference-ticker θ from Stage A
for (r, tk) in enumerate(ref_tickers)
    θ_vals = [exp(x_A[n_β + (r-1)*N_HMM + s]) for s in 1:N_HMM]
    iv_vals = [round(sqrt(max(θ,0))*100, digits=1) for θ in θ_vals]
    println("    $tk θ_states IV: $iv_vals %")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5B — Per-ticker θ calibration (fixing shared β)
# ═══════════════════════════════════════════════════════════════════════════════

println("\n═══ Phase 5B: Per-ticker θ calibration (β fixed from 5A) ═══\n")

calibrated = Dict{String, Tuple{HestonParameters, ThetaHybrid}}()
heston_fixed = HestonParameters(FIXED_κ, FIXED_σv)

# Store reference-ticker results from Stage A
for (r, tk) in enumerate(ref_tickers)
    θ_states = [exp(x_A[n_β + (r-1)*N_HMM + s]) for s in 1:N_HMM]
    calibrated[tk] = (heston_fixed, ThetaHybrid(θ_states, shared_β, FIXED_γ))
end

# Calibrate remaining tickers: only optimize θ_states with fixed β
for tk in TICKERS
    haskey(calibrated, tk) && continue
    !haskey(ticker_obs, tk) && continue
    obs = ticker_obs[tk]
    length(obs) < 50 && (@warn "Skipping $tk: only $(length(obs)) observations"; continue)

    if length(obs) > 30_000
        step = length(obs) ÷ 30_000
        obs = obs[1:step:end]
    end

    tk_strikes = [o.strike for o in obs]
    tk_dtes    = [o.dte for o in obs]
    tk_ivs     = [o.iv for o in obs]
    tk_spots   = [o.spot for o in obs]
    tk_states  = [o.state for o in obs]
    n_tk = length(obs)

    θ_init_tk = fill(0.04, N_HMM)
    for i in 1:n_tk
        s = tk_states[i]
        if 1 <= s <= N_HMM
            θ_init_tk[s] = tk_ivs[i]^2
        end
    end

    x0_tk = log.(max.(θ_init_tk, 1e-8))

    function objective_theta(x)
        total_error = 0.0
        for i in 1:n_tk
            s = tk_states[i]
            (s < 1 || s > N_HMM) && continue
            θ_s = exp(x[s])
            DTE = Float64(tk_dtes[i])
            moneyness = tk_strikes[i] / tk_spots[i]
            ψ_val = ψ(shared_β, DTE, moneyness)
            σ_model = sqrt(max(θ_s * ψ_val, 1e-10))
            total_error += (σ_model - tk_ivs[i])^2
        end
        return total_error / n_tk
    end

    try
        result_tk = optimize(objective_theta, x0_tk, NelderMead(),
                             Optim.Options(iterations=5000, show_trace=false))
        θ_opt = exp.(Optim.minimizer(result_tk))
        calibrated[tk] = (heston_fixed, ThetaHybrid(θ_opt, shared_β, FIXED_γ))
        iv_vals = [round(sqrt(max(θ,0))*100, digits=1) for θ in θ_opt]
        println("  ✓ $tk ($(n_tk) obs): IV per state = $iv_vals %, loss=$(round(Optim.minimum(result_tk), digits=6))")
    catch e
        @warn "Calibration failed for $tk" exception=(e, catch_backtrace())
    end
end

# Fallback: tickers with no option data → auto-calibrate θ from RV
missing_tickers = [tk for tk in TICKERS if !haskey(calibrated, tk) && haskey(merged_prices, tk)]
if !isempty(missing_tickers)
    println("\n  Auto-calibrating $(length(missing_tickers)) tickers without option data:")
    for tk in missing_tickers
        df = merged_prices[tk]
        nrow(df) < 61 && continue
        prices = Float64.(df.adj_close)
        model = get(hmm_models, tk, fit_jumphmm(prices; N=N_HMM, nu=5.0))
        θ_states = auto_calibrate_theta_states(model, prices)
        calibrated[tk] = (heston_fixed, ThetaHybrid(θ_states, shared_β, FIXED_γ))
        iv_vals = [round(sqrt(max(θ,0))*100, digits=1) for θ in θ_states]
        println("    $tk: auto-calibrated, IV per state = $iv_vals %")
    end
end

println("\n  Calibrated: $(length(calibrated)) / $(length(TICKERS)) tickers")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6 — Generate heston_params.csv for the backtest year
# ═══════════════════════════════════════════════════════════════════════════════

println("\n═══ Phase 6: Writing $OUTPUT_PATH ═══\n")

rows = NamedTuple[]
for tk in TICKERS
    !haskey(calibrated, tk) && continue
    !haskey(hmm_state_lookup, tk) && continue

    heston, theta_hybrid = calibrated[tk]
    state_dict = hmm_state_lookup[tk]

    test_dates = if haskey(testing_prices, tk)
        sort(testing_prices[tk].date)
    else
        Date[]
    end

    β = theta_hybrid.β
    β5_val = length(β) >= 5 ? β[5] : 0.0

    for d in test_dates
        s = get(state_dict, d, 1)
        s = clamp(s, 1, N_HMM)
        θ_base = theta_hybrid.θ_states[s]

        push!(rows, (
            ticker     = tk,
            date       = d,
            theta_base = θ_base,
            beta1      = β[1],
            beta2      = β[2],
            beta3      = β[3],
            beta4      = β[4],
            beta5      = β5_val,
            gamma      = theta_hybrid.γ,
            kappa      = heston.κ,
            sigma_v    = heston.σ_v,
        ))
    end
end

output_df = DataFrame(rows)
CSV.write(OUTPUT_PATH, output_df)

println("  Saved $(nrow(output_df)) rows ($(length(unique(output_df.ticker))) tickers) to $OUTPUT_PATH")

# Summary
println("\n═══ Calibration Summary ═══\n")
println("  Shared β: $(round.(shared_β, digits=4))")
println("  Fixed: κ=$(FIXED_κ), σ_v=$(FIXED_σv), γ=$(FIXED_γ)")
println()
for tk in TICKERS
    !haskey(calibrated, tk) && continue
    _, theta_hybrid = calibrated[tk]
    iv_vals = [round(sqrt(max(θ,0))*100, digits=1) for θ in theta_hybrid.θ_states]
    n_obs = haskey(ticker_obs, tk) ? length(ticker_obs[tk]) : 0
    println("  $tk: $(n_obs) market obs, IV per state = $iv_vals %")
end

println("\n✓ Done. Run Backtest.jl to use these calibrated parameters.")
