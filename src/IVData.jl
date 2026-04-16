"""
Modified Heston Implied Volatility Model

Pipeline: JumpHMM synthetic prices → Heston dv variance process → IV → CRR binomial tree → American option prices

The key innovation is that the Heston mean-reversion target θ is a hybrid function
of the JumpHMM state, days to expiration, moneyness, and aggregate market mood:

    θ(t) = θ_{s_t} · (1 + γ·M_t) · ψ(DTE, K/S_t)

where ψ = exp(β₁·ln(DTE) + β₂·ln(K/S) + β₃·ln(DTE)·ln(K/S) + β₄·(ln(K/S))²)

Heston variance process:
    dv = κ(θ(t) - v)dt + σ_v √v dW_v

with reflecting boundary at v = 0 and time-varying θ(t).

Reference: Varner (2025) "Modified Heston Implied Volatility Model"
"""

using Random

# ══════════════════════════════════════════════════════════════════════════════════
# Types (from HestonIV/Types.jl)
# ══════════════════════════════════════════════════════════════════════════════════

"""
    HestonParameters

Parameters for the Heston stochastic variance process:
    dv = κ(θ(t) - v)dt + σ_v √v dW_v

- `κ`: mean-reversion speed
- `σ_v`: vol-of-vol

Note: v₀ is not stored here — it is initialized per (ticker, strike, DTE) as
v₀ = θ(s₀, DTE, K/S₀, M₀), so the process starts at the mean-reversion target
for the current regime. This gives each contract its own initial IV automatically.
"""
struct HestonParameters
    κ::Float64
    σ_v::Float64
end

"""
    ThetaHybrid

Hybrid θ-function: θ(t) = θ_{s_t} · (1 + γ·M_t) · ψ(DTE, K/S_t)

- `θ_states`: vector of θ values, one per HMM state
- `β`: parameters for ψ (Varner 2025 eq. 11):
    [β₁, β₂, β₃, β₄, β₅] → ψ = exp(β₁·ln(τ) + β₂·ln(m) + β₃·ln(τ)·ln(m) + β₄·(ln(m))² + β₅·(ln(τ))²)
    β₁: linear DTE decay
    β₂: skew (asymmetry)
    β₃: DTE × skew interaction
    β₄: smile curvature
    β₅: DTE curvature (U-shaped ATM term structure)
- `γ`: market mood sensitivity
"""
struct ThetaHybrid
    θ_states::Vector{Float64}
    β::Vector{Float64}  # [β₁, β₂, β₃, β₄, β₅]
    γ::Float64
end

"""
    OptionContract

Specification of an option contract.
"""
struct OptionContract
    K::Float64          # strike price
    DTE::Int            # days to expiration
    option_type::Symbol # :call or :put
    style::Symbol       # :american or :european
end

# ══════════════════════════════════════════════════════════════════════════════════
# ψ function and compute_theta (from HestonIV/ThetaFunction.jl)
# ══════════════════════════════════════════════════════════════════════════════════

"""
    ψ(β, DTE, moneyness) → Float64

Term-structure, skew, and smile adjustment (Varner 2025 eq. 11):
    ψ = exp(β₁·ln(τ) + β₂·ln(m) + β₃·ln(τ)·ln(m) + β₄·(ln(m))² + β₅·(ln(τ))²)

where τ = max(DTE, 1) and m = K/S.

- β₁: linear DTE decay (negative = short-term IV elevated)
- β₂: skew (negative = put skew, OTM puts have higher IV)
- β₃: DTE × skew interaction (positive = skew flattens at longer maturities)
- β₄: smile curvature
- β₅: DTE curvature (positive = U-shaped ATM term structure)
"""
function ψ(β::Vector{Float64}, DTE::Float64, moneyness::Float64)::Float64
    log_τ = log(max(DTE, 1.0))
    log_m = log(moneyness)
    β₄ = length(β) >= 4 ? β[4] : 0.0
    β₅ = length(β) >= 5 ? β[5] : 0.0
    return exp(β[1] * log_τ + β[2] * log_m + β[3] * log_τ * log_m + β₄ * log_m^2 + β₅ * log_τ^2)
end

"""
    compute_theta(θ_hybrid, s_t, DTE, moneyness, mood) → Float64

Compute the time-varying mean-reversion target:
    θ(t) = θ_{s_t} · (1 + γ·M_t) · ψ(DTE, K/S_t)

# Arguments
- `θ_hybrid::ThetaHybrid`: the hybrid θ-function parameters
- `s_t::Int`: current HMM state index (1-based)
- `DTE::Float64`: days to expiration
- `moneyness::Float64`: K/S_t ratio
- `mood::Float64`: aggregate market mood ∈ [0, 1]
"""
function compute_theta(θ_hybrid::ThetaHybrid, s_t::Int, DTE::Float64,
                       moneyness::Float64, mood::Float64)::Float64
    θ_base = θ_hybrid.θ_states[s_t]
    mood_factor = 1.0 + θ_hybrid.γ * mood
    ψ_factor = ψ(θ_hybrid.β, DTE, moneyness)
    return θ_base * mood_factor * ψ_factor
end

"""
    compute_mood(states, n_states, n_tail) → Float64

Compute aggregate market mood as the fraction of tickers currently in tail states.

# Arguments
- `states::Vector{Int}`: current HMM state index for each ticker
- `n_states::Int`: total number of HMM states
- `n_tail::Int`: number of states at each tail considered "extreme"

Returns a value in [0, 1] where 0 = no tickers in tail states, 1 = all tickers in tails.
"""
function compute_mood(states::Vector{Int}, n_states::Int, n_tail::Int)::Float64
    n_tickers = length(states)
    n_tickers == 0 && return 0.0
    count = 0
    for s in states
        if s <= n_tail || s > n_states - n_tail
            count += 1
        end
    end
    return count / n_tickers
end

"""
    compute_mood_path(state_matrix, n_states, n_tail) → Vector{Float64}

Compute market mood at each timestep from a matrix of HMM states.

# Arguments
- `state_matrix`: n_tickers × n_steps matrix of HMM state indices
- `n_states::Int`: total number of HMM states
- `n_tail::Int`: number of tail states at each end
"""
function compute_mood_path(state_matrix::Matrix{Int}, n_states::Int, n_tail::Int)::Vector{Float64}
    n_steps = size(state_matrix, 2)
    mood = Vector{Float64}(undef, n_steps)
    for t in 1:n_steps
        mood[t] = compute_mood(view(state_matrix, :, t), n_states, n_tail)
    end
    return mood
end

"""
    auto_calibrate_theta_states(model, prices; rf, dt) → Vector{Float64}

Compute θ_states from the empirical variance of returns in each HMM state.

Decodes the historical price series into HMM states using the market model's partition,
then computes mean(G²) for each state. States with no observations fall back to the
unconditional variance.

# Arguments
- `model`: a fitted JumpHiddenMarkovModel (e.g., from fit_jumphmm)
- `prices::AbstractVector{Float64}`: historical close prices

# Keyword Arguments
- `rf::Float64`: risk-free rate (default: model.rf)
- `dt::Float64`: time step in years (default: model.dt)

# Returns
Vector of length N_states, where θ_states[s] = annualized realized variance when market
is in state s (suitable for IV = √θ). Converted from growth-rate units via multiplication
by dt.
"""
function auto_calibrate_theta_states(model, prices::AbstractVector{Float64};
                                     rf::Union{Float64,Nothing}=nothing,
                                     dt::Union{Float64,Nothing}=nothing)::Vector{Float64}
    rf_val = rf !== nothing ? rf : model.rf
    dt_val = dt !== nothing ? dt : model.dt

    G = excess_growth_rates(prices; rf=rf_val, dt=dt_val)
    states = assign_states(model.partition, G)

    N_states = model.partition.N
    unconditional_var = sum(G .^ 2) / length(G) * dt_val

    θ_states = Vector{Float64}(undef, N_states)
    for s in 1:N_states
        mask = states .== s
        n_obs = sum(mask)
        if n_obs > 0
            θ_states[s] = sum(G[mask] .^ 2) / n_obs * dt_val
        else
            θ_states[s] = unconditional_var
        end
    end

    return θ_states
end

"""
    auto_calibrate_heston(price_data, tickers; N, nu, tune_jumps, default_β, default_γ, default_κ, default_σv)

Fit JumpHMM per ticker, auto-calibrate θ_states from realized variance per HMM state,
decode states for each trading day, and produce a heston_ts dictionary.

This implements the new model's auto-calibration pipeline:
  1. fit_jumphmm(prices) → HMM model with N states
  2. auto_calibrate_theta_states(model, prices) → θ per state
  3. assign_states(partition, G) → decoded state per day
  4. θ_base[date] = θ_states[state[date]]

Returns Dict{String, Dict{Date, HestonCalibration}} ready for the Wheel engine.
"""
function auto_calibrate_heston(price_data::Dict{String, DataFrame},
                                tickers::Vector{String};
                                N::Int=100, nu::Float64=5.0,
                                tune_jumps::Bool=false,
                                default_β::Vector{Float64}=Float64[0.0, 0.0, 0.0, 0.0, 0.0],
                                default_γ::Float64=0.0,
                                default_κ::Float64=5.0,
                                default_σv::Float64=0.3)::Dict{String, Dict{Date, HestonCalibration}}
    heston_ts = Dict{String, Dict{Date, HestonCalibration}}()

    for tk in tickers
        !haskey(price_data, tk) && continue
        df = price_data[tk]
        nrow(df) < 61 && continue

        try
            prices = Float64.(df.adj_close)
            model = fit_jumphmm(prices; N=N, nu=nu)
            if tune_jumps
                model = tune_jumphmm(model, prices)
            end

            θ_states = auto_calibrate_theta_states(model, prices)

            G = excess_growth_rates(prices; rf=model.rf, dt=model.dt)
            decoded_states = assign_states(model.partition, G)

            tk_cal = Dict{Date, HestonCalibration}()
            dates = df.date
            for i in 1:length(decoded_states)
                d = dates[i + 1]
                s = decoded_states[i]
                θ_base = θ_states[s]
                β5_val = length(default_β) >= 5 ? default_β[5] : 0.0
                tk_cal[d] = HestonCalibration(θ_base, default_β[1], default_β[2],
                                               default_β[3], default_β[4], β5_val,
                                               default_γ, default_κ, default_σv)
            end
            heston_ts[tk] = tk_cal
            println("    $tk: $(length(tk_cal)) dates, θ range [$(round(minimum(θ_states), digits=4)), $(round(maximum(θ_states), digits=4))], IV range [$(round(sqrt(minimum(θ_states))*100, digits=1))%, $(round(sqrt(maximum(θ_states))*100, digits=1))%]")
        catch e
            @warn "Auto-calibration failed for $tk: $e"
        end
    end

    println("  -> Auto-calibrated Heston: $(length(heston_ts)) tickers")
    return heston_ts
end

# ══════════════════════════════════════════════════════════════════════════════════
# Heston Variance Process (from HestonIV/HestonVariance.jl)
# ══════════════════════════════════════════════════════════════════════════════════

"""
    simulate_variance(params, θ_func, hmm_states, S_path, contract, mood_path; Δt, rng) → Vector{Float64}

Simulate the modified Heston variance process along a single JumpHMM price path.

The initial variance v₀ is set to θ(t=0) — the mean-reversion target at the
first timestep — so each (ticker, strike, DTE) triple starts at its own
equilibrium IV. This gives the initial IV surface smile, skew, and term
structure automatically.

The variance process is discretized via Euler-Maruyama with a reflecting boundary:
    v_{t+1} = |v_t + κ(θ_t - v_t)Δt + σ_v·√(max(v_t,0))·√Δt·Z|

# Arguments
- `params::HestonParameters`: κ, σ_v
- `θ_func::ThetaHybrid`: hybrid θ-function parameters
- `hmm_states::Vector{Int}`: HMM state sequence from JumpHMM simulation
- `S_path::Vector{Float64}`: underlying price path
- `contract::OptionContract`: option contract (provides K and DTE)
- `mood_path::Vector{Float64}`: market mood at each timestep
- `Δt::Float64`: time step in years (default 1/252)
- `rng`: random number generator

# Returns
Vector of variance values v_t at each timestep. Take √v_t to get implied volatility.
"""
function simulate_variance(params::HestonParameters, θ_func::ThetaHybrid,
                           hmm_states::Vector{Int}, S_path::Vector{Float64},
                           contract::OptionContract, mood_path::Vector{Float64};
                           Δt::Float64=1.0/252.0,
                           rng::AbstractRNG=Random.default_rng())::Vector{Float64}
    n_steps = length(hmm_states)
    v = Vector{Float64}(undef, n_steps)

    remaining_DTE_0 = Float64(contract.DTE)
    moneyness_0 = contract.K / S_path[1]
    v[1] = compute_theta(θ_func, hmm_states[1], remaining_DTE_0,
                         moneyness_0, mood_path[1])

    sqrt_Δt = sqrt(Δt)

    for t in 1:(n_steps - 1)
        remaining_DTE = max(contract.DTE - t + 1, 1)
        moneyness = contract.K / S_path[t]

        θ_t = compute_theta(θ_func, hmm_states[t], Float64(remaining_DTE),
                            moneyness, mood_path[t])

        v_curr = max(v[t], 0.0)
        dv = params.κ * (θ_t - v_curr) * Δt + params.σ_v * sqrt(v_curr) * sqrt_Δt * randn(rng)
        v[t + 1] = abs(v_curr + dv)
    end

    return v
end

"""
    simulate_variance_ensemble(params, θ_func, hmm_states_matrix, S_paths_matrix,
                                contract, mood_path; n_var_paths, Δt, rng) → Matrix{Float64}

Simulate multiple variance paths for each synthetic price path.

For scenario analysis, each JumpHMM price path can generate multiple variance
realizations (since dW_v is independent of the price path noise).

# Arguments
- `hmm_states_matrix`: n_paths × n_steps matrix of HMM states
- `S_paths_matrix`: n_paths × n_steps matrix of underlying prices
- `mood_path`: n_steps vector of market mood (shared across paths for a given simulation)
- `n_var_paths::Int`: number of variance paths per price path

# Returns
(n_paths * n_var_paths) × n_steps matrix of variance values.
"""
function simulate_variance_ensemble(params::HestonParameters, θ_func::ThetaHybrid,
                                    hmm_states_matrix::Matrix{Int},
                                    S_paths_matrix::Matrix{Float64},
                                    contract::OptionContract,
                                    mood_path::Vector{Float64};
                                    n_var_paths::Int=1,
                                    Δt::Float64=1.0/252.0,
                                    rng::AbstractRNG=Random.default_rng())::Matrix{Float64}
    n_price_paths, n_steps = size(S_paths_matrix)
    total_paths = n_price_paths * n_var_paths
    V = Matrix{Float64}(undef, total_paths, n_steps)

    idx = 1
    for i in 1:n_price_paths
        states_i = view(hmm_states_matrix, i, :) |> collect
        S_i = view(S_paths_matrix, i, :) |> collect
        for _ in 1:n_var_paths
            V[idx, :] .= simulate_variance(params, θ_func, states_i, S_i,
                                           contract, mood_path; Δt=Δt, rng=rng)
            idx += 1
        end
    end

    return V
end

"""Convert variance to implied volatility: σ_imp = √v."""
variance_to_iv(v::Float64)::Float64 = sqrt(max(v, 0.0))

"""Convert a variance path to an IV path."""
variance_path_to_iv(v_path::Vector{Float64})::Vector{Float64} = [variance_to_iv(v) for v in v_path]

# ══════════════════════════════════════════════════════════════════════════════════
# Calibration (from HestonIV/Calibration.jl)
# ══════════════════════════════════════════════════════════════════════════════════

"""
    CalibrationData

Preprocessed calibration dataset: one row per (date, strike, DTE) observation.
"""
struct CalibrationData
    dates::Vector{Int}           # timestep index into the price series
    strikes::Vector{Float64}     # strike prices
    dtes::Vector{Int}            # days to expiration
    market_ivs::Vector{Float64}  # observed implied volatilities
    spot_prices::Vector{Float64} # underlying price at each observation
    hmm_states::Vector{Int}      # decoded HMM state at each observation
    moods::Vector{Float64}       # market mood at each observation
end

"""
    calibrate(cal_data, N_states; kwargs...) → (HestonParameters, ThetaHybrid)

Calibrate the Heston + ThetaHybrid parameters to minimize IV prediction error.

Since v₀ = θ(t=0) (the process starts at equilibrium), the calibration objective
simplifies: the model-predicted IV for each observation is just √θ(s, DTE, K/S, M).
The κ parameter controls how quickly the process would mean-revert if perturbed,
and σ_v controls the stochastic spread around the target.

# Arguments
- `cal_data::CalibrationData`: preprocessed calibration data
- `N_states::Int`: number of HMM states

# Keyword Arguments
- `κ_init::Float64`: initial κ (default 5.0)
- `σv_init::Float64`: initial σ_v (default 0.3)
- `γ_init::Float64`: initial mood sensitivity (default 0.5)
- `method`: Optim method (default NelderMead())
- `maxiter::Int`: maximum iterations (default 5000)

# Returns
Tuple of (HestonParameters, ThetaHybrid) that minimize Σ(σ_model - IV_market)²
"""
function calibrate(cal_data::CalibrationData, N_states::Int;
                   κ_init::Float64=5.0,
                   σv_init::Float64=0.3,
                   γ_init::Float64=0.5,
                   method=NelderMead(),
                   maxiter::Int=5000)

    n_obs = length(cal_data.market_ivs)

    # Parameter vector layout (Varner 2025 eq. 11):
    # [1:5]       = β₁, β₂, β₃, β₄, β₅ (ψ parameters)
    # [6:5+N]     = log(θ_states)   (ensures θ > 0 for each state)
    # κ, σ_v, γ are fixed (not optimized — κ/σ_v are unidentifiable from √θ)

    θ_init = fill(0.04, N_states)
    for i in 1:n_obs
        s = cal_data.hmm_states[i]
        if 1 <= s <= N_states
            θ_init[s] = cal_data.market_ivs[i]^2
        end
    end

    n_params = 5 + N_states
    x0 = Vector{Float64}(undef, n_params)
    x0[1] = 0.0   # β₁ (DTE decay)
    x0[2] = 0.0   # β₂ (skew)
    x0[3] = 0.0   # β₃ (interaction)
    x0[4] = 0.0   # β₄ (smile curvature)
    x0[5] = 0.0   # β₅ (DTE curvature)
    x0[6:end] .= log.(max.(θ_init, 1e-8))

    function objective(x)
        β = x[1:5]
        θ_states = exp.(x[6:end])
        γ = γ_init

        θ_func = ThetaHybrid(θ_states, β, γ)

        total_error = 0.0
        for i in 1:n_obs
            s = cal_data.hmm_states[i]
            if s < 1 || s > N_states
                continue
            end
            DTE = Float64(cal_data.dtes[i])
            moneyness = cal_data.strikes[i] / cal_data.spot_prices[i]
            mood = cal_data.moods[i]

            θ_t = compute_theta(θ_func, s, DTE, moneyness, mood)
            σ_model = sqrt(max(θ_t, 1e-10))

            total_error += (σ_model - cal_data.market_ivs[i])^2
        end

        # Soft bounds on β to prevent extreme parameters (per paper Table 4)
        penalty = 0.0
        penalty += max(0.0, abs(β[1]) - 1.0)^2 * 10.0
        penalty += max(0.0, abs(β[2]) - 12.0)^2 * 10.0
        penalty += max(0.0, β[3] < 0.0 ? -β[3] : 0.0)^2 * 10.0  # β₃ ≥ 0
        penalty += max(0.0, abs(β[4]) - 3.0)^2 * 10.0
        penalty += max(0.0, abs(β[5]) - 0.5)^2 * 10.0

        return total_error / n_obs + penalty
    end

    result = optimize(objective, x0, method,
                      Optim.Options(iterations=maxiter, show_trace=false))

    x_opt = Optim.minimizer(result)

    heston = HestonParameters(κ_init, σv_init)
    θ_func = ThetaHybrid(exp.(x_opt[6:end]), x_opt[1:5], γ_init)

    return heston, θ_func
end

# ══════════════════════════════════════════════════════════════════════════════════
# Wheel Integration Layer
# ══════════════════════════════════════════════════════════════════════════════════

"""
    HestonParams

Runtime parameters for the Wheel engine, combining ThetaHybrid + HestonParameters
into a single struct for convenience. Single-state: θ_base = θ_states[1], hmm_state=1.

- `θ_base`: base variance level (= θ_states[1] for single-state calibration)
- `β`: [β₁, β₂, β₃, β₄] — ψ shape parameters
- `γ`: mood/regime sensitivity
- `κ`: mean-reversion speed of variance process
- `σ_v`: vol-of-vol
- `mood`: current market mood ∈ [0,1], default 0.0
"""
struct HestonParams
    θ_base::Float64
    β::Vector{Float64}
    γ::Float64
    κ::Float64
    σ_v::Float64
    mood::Float64
end

"""
    HestonCalibration

Stored calibration result per (ticker, date).
Flat fields for CSV serialization.
"""
struct HestonCalibration
    θ_base::Float64
    β1::Float64
    β2::Float64
    β3::Float64
    β4::Float64
    β5::Float64
    γ::Float64
    κ::Float64
    σ_v::Float64
end

const HESTON_CAL_PATH = joinpath(_PATH_TO_DATA, "heston_params.csv")

# ── IV Computation ────────────────────────────────────────────────────────────

"""
    heston_iv_for_option(S, K, T, r, params; q, option_type) → Float64

Compute per-contract implied volatility using the modified Heston model.
Internally constructs ThetaHybrid and delegates to compute_theta:

    IV = √(compute_theta(ThetaHybrid([θ_base], β, γ), 1, DTE, K/S, mood))

This gives each (K, DTE) pair its own IV automatically, producing
the volatility smile, skew, and term structure from the ψ function.
"""
function heston_iv_for_option(S::Float64, K::Float64, T::Float64, r::Float64,
                               params::HestonParams;
                               q::Float64=0.0, option_type::Symbol=:put)::Float64
    DTE = max(T * 365.0, 1.0)
    moneyness = K / S
    θ_func = ThetaHybrid([params.θ_base], params.β, params.γ)
    θ_t = compute_theta(θ_func, 1, DTE, moneyness, params.mood)
    return sqrt(max(θ_t, 1e-10))
end

# ── Heston Variance Process (Wheel wrapper) ──────────────────────────────────

"""
    simulate_heston_variance(params, hmm_states, S_path, K, DTE, mood_path; ...) → Vector{Float64}

Wrapper around simulate_variance (from new code) for the Wheel engine's flat HestonParams.
Constructs HestonParameters, ThetaHybrid, OptionContract internally.
"""
function simulate_heston_variance(params::HestonParams,
                                   hmm_states::Vector{Int},
                                   S_path::Vector{Float64},
                                   K::Float64, DTE::Int,
                                   mood_path::Vector{Float64};
                                   Δt::Float64=1.0/252.0,
                                   rng::AbstractRNG=Random.default_rng())::Vector{Float64}
    heston = HestonParameters(params.κ, params.σ_v)
    θ_func = ThetaHybrid([params.θ_base], params.β, params.γ)
    contract = OptionContract(K, DTE, :put, :american)
    return simulate_variance(heston, θ_func, hmm_states, S_path, contract, mood_path;
                             Δt=Δt, rng=rng)
end

# ── Calibration (Nelder-Mead, matching new code's approach) ──────────────────

"""
    calibrate_heston_from_options(S, r, option_data; q) → HestonParams

Calibrate from market option prices: extract BS IVs, then fit via Nelder-Mead.
"""
function calibrate_heston_from_options(S::Float64, r::Float64,
                                        option_data::Vector;
                                        q::Float64=0.0)::HestonParams
    length(option_data) < 3 && error("Need ≥3 option observations, got $(length(option_data))")

    market_ivs = Float64[]
    strikes = Float64[]
    dtes = Float64[]
    for opt in option_data
        T = opt.T
        K = opt.K
        iv = _extract_bs_iv(S, K, T, r, opt.market_price, opt.option_type; q=q)
        if iv > 0.001 && iv < 3.0
            push!(market_ivs, iv)
            push!(strikes, K)
            push!(dtes, max(T * 365.0, 1.0))
        end
    end

    length(market_ivs) < 3 && error("Need ≥3 valid IV observations after extraction")

    return calibrate_heston_nelder_mead(market_ivs, strikes, dtes, S)
end

"""
    calibrate_heston_nelder_mead(market_ivs, strikes, dtes, S; mood) → HestonParams

Calibrate θ_base, β₁..β₄, γ, κ, σ_v by minimizing Σ(√(compute_theta(...)) - IV_market)²
using Nelder-Mead. Matches the new code's calibrate() for single-state (N_states=1).
"""
function calibrate_heston_nelder_mead(market_ivs::Vector{Float64},
                                       strikes::Vector{Float64},
                                       dtes::Vector{Float64},
                                       S::Float64;
                                       mood::Float64=0.0)::HestonParams
    moneyness = strikes ./ S
    n_obs = length(market_ivs)
    mean_iv2 = mean(market_ivs .^ 2)

    # Parameter vector: [log(θ_base), β₁, β₂, β₃, β₄, β₅]
    # κ, σ_v, γ are fixed (not identifiable from √θ)
    x0 = [log(max(mean_iv2, 1e-8)), 0.0, 0.0, 0.0, 0.0, 0.0]

    function objective(x)
        θ_base = exp(x[1])
        β = x[2:6]
        θ_func = ThetaHybrid([θ_base], β, mood)
        total = 0.0
        for i in 1:n_obs
            θ_t = compute_theta(θ_func, 1, dtes[i], moneyness[i], mood)
            σ_model = sqrt(max(θ_t, 1e-10))
            total += (σ_model - market_ivs[i])^2
        end
        return total / n_obs
    end

    result = optimize(objective, x0, NelderMead(),
                      Optim.Options(iterations=5000, show_trace=false))
    x_opt = Optim.minimizer(result)

    return HestonParams(exp(x_opt[1]), x_opt[2:6], 0.0, 5.0, 0.3, mood)
end

"""Extract Black-Scholes implied volatility from a market price via bisection."""
function _extract_bs_iv(S::Float64, K::Float64, T::Float64, r::Float64,
                         market_price::Float64, option_type::Symbol;
                         q::Float64=0.0)::Float64
    T <= 0.0 && return 0.0
    market_price <= 0.0 && return 0.0

    σ_lo, σ_hi = 0.001, 5.0
    for _ in 1:80
        σ_mid = (σ_lo + σ_hi) / 2.0
        bs_price = _bs_price(S, K, T, r, σ_mid, option_type; q=q)
        if bs_price > market_price
            σ_hi = σ_mid
        else
            σ_lo = σ_mid
        end
        (σ_hi - σ_lo) < 1e-7 && break
    end
    return (σ_lo + σ_hi) / 2.0
end

function _bs_price(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64,
                    option_type::Symbol; q::Float64=0.0)::Float64
    T <= 0.0 && return max(option_type == :call ? S - K : K - S, 0.0)
    d1 = (log(S / K) + (r - q + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    nd = Normal(0.0, 1.0)
    if option_type == :call
        return S * exp(-q * T) * cdf(nd, d1) - K * exp(-r * T) * cdf(nd, d2)
    else
        return K * exp(-r * T) * cdf(nd, -d2) - S * exp(-q * T) * cdf(nd, -d1)
    end
end

# ── Load / Lookup ─────────────────────────────────────────────────────────────

"""
    load_heston_params(path) → Dict{String, Dict{Date, HestonCalibration}}

Load Heston calibration time-series from CSV.
Expected columns: ticker, date, theta_base, beta1..beta5, gamma, kappa, sigma_v
"""
function load_heston_params(path::String)::Dict{String, Dict{Date, HestonCalibration}}
    df = CSV.read(path, DataFrame)
    result = Dict{String, Dict{Date, HestonCalibration}}()

    has_β5 = hasproperty(df, :beta5)

    for row in eachrow(df)
        tk = row.ticker
        d = Date(row.date)

        β5_val = has_β5 ? row.beta5 : 0.0
        cal = HestonCalibration(row.theta_base, row.beta1, row.beta2, row.beta3, row.beta4,
                                 β5_val, row.gamma, row.kappa, row.sigma_v)

        if !haskey(result, tk)
            result[tk] = Dict{Date, HestonCalibration}()
        end
        result[tk][d] = cal
    end

    n_tickers = length(result)
    n_dates = isempty(result) ? 0 : maximum(length(v) for v in values(result))
    println("  -> Loaded Heston params: $(n_tickers) tickers × up to $(n_dates) calibration dates")
    return result
end

"""
    lookup_heston_params(heston_ts, ticker, date) → Union{HestonParams, Nothing}

Find the most recent calibration on or before `date` for `ticker`.
Returns a ready-to-use HestonParams with mood=0.0.
"""
function lookup_heston_params(heston_ts::Dict{String, Dict{Date, HestonCalibration}},
                                ticker::String, date::Date)::Union{HestonParams, Nothing}
    !haskey(heston_ts, ticker) && return nothing
    cal_dates = sort(collect(keys(heston_ts[ticker])))
    isempty(cal_dates) && return nothing
    idx = searchsortedlast(cal_dates, date)
    idx == 0 && return nothing
    cal = heston_ts[ticker][cal_dates[idx]]
    return HestonParams(cal.θ_base, [cal.β1, cal.β2, cal.β3, cal.β4, cal.β5],
                         cal.γ, cal.κ, cal.σ_v, 0.0)
end

# ── Regime Adjustment ─────────────────────────────────────────────────────────

"""
    compute_stock_rv_regime(rolling_vol) → Dict{String, Dict{Date, Float64}}

Compute a per-stock daily regime multiplier:
    regime = σ_rv(date) / expanding_median(σ_rv) ∈ [0.5, 2.0]
"""
function compute_stock_rv_regime(rolling_vol::Dict{String, Dict{Date, Float64}})::Dict{String, Dict{Date, Float64}}
    result = Dict{String, Dict{Date, Float64}}()

    for (tk, vol_dict) in rolling_vol
        isempty(vol_dict) && continue
        sorted_dates = sort(collect(keys(vol_dict)))
        rv_vals = [vol_dict[d] for d in sorted_dates]

        tk_regime = Dict{Date, Float64}()
        for (j, d) in enumerate(sorted_dates)
            expanding_med = median(rv_vals[1:j])
            regime = clamp(rv_vals[j] / max(expanding_med, 0.01), 0.5, 2.0)
            tk_regime[d] = regime
        end
        result[tk] = tk_regime
    end

    return result
end

"""
    rv_adjusted_params(params, regime_scale) → HestonParams

Scale θ_base by the per-stock RV regime multiplier.
regime > 1.0 → stock more volatile than its norm → higher IV
regime < 1.0 → stock calmer than usual → lower IV
"""
function rv_adjusted_params(params::HestonParams, regime_scale::Float64)::HestonParams
    return HestonParams(params.θ_base * regime_scale, params.β, params.γ,
                         params.κ, params.σ_v, params.mood)
end

"""
    vix_adjusted_params(params, regime_scale) → HestonParams

Alias for rv_adjusted_params — scales θ_base by a market-wide regime factor.
"""
function vix_adjusted_params(params::HestonParams, regime_scale::Float64)::HestonParams
    return rv_adjusted_params(params, regime_scale)
end

# ── IV Surface & Map ──────────────────────────────────────────────────────────

"""
    generate_iv_surface(S, r, params; tenors, deltas, q) → DataFrame

Generate a full IV surface for a given stock price and Heston parameters.
"""
function generate_iv_surface(S::Float64, r::Float64, params::HestonParams;
                              tenors::Vector{Int}=[7, 14, 30, 60, 90],
                              deltas::Vector{Float64}=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
                              q::Float64=0.0)::DataFrame
    rows = NamedTuple{(:tenor_days, :delta, :strike, :iv_put, :iv_call, :put_price, :call_price),
                       Tuple{Int, Float64, Float64, Float64, Float64, Float64, Float64}}[]

    atm_iv = heston_iv_for_option(S, S, 30.0/365.0, r, params)

    for tenor in tenors
        T = tenor / 365.0
        for δ in deltas
            K = strike_from_delta(S, T, r, atm_iv, δ, :put; q=q)
            iv_p = heston_iv_for_option(S, K, T, r, params; q=q, option_type=:put)
            iv_c = heston_iv_for_option(S, K, T, r, params; q=q, option_type=:call)
            p_price = option_price(S, K, T, r, iv_p, :put; q=q)
            c_price = option_price(S, K, T, r, iv_c, :call; q=q)
            push!(rows, (tenor_days=tenor, delta=δ, strike=round(K, digits=2),
                         iv_put=round(iv_p, digits=4), iv_call=round(iv_c, digits=4),
                         put_price=round(p_price, digits=2), call_price=round(c_price, digits=2)))
        end
    end

    return DataFrame(rows)
end

"""
    build_heston_iv_map(price_data, trading_days; r, heston_ts, rolling_vol) → Dict

Build rolling ATM IV map using pre-calibrated modified Heston parameters,
with θ_base daily-adjusted by each stock's own RV regime.
"""
function build_heston_iv_map(price_data::Dict{String, DataFrame},
                               trading_days::Vector{Date};
                               r::Float64=0.045,
                               heston_ts::Dict{String, Dict{Date, HestonCalibration}}=Dict{String, Dict{Date, HestonCalibration}}(),
                               rolling_vol::Dict{String, Dict{Date, Float64}}=Dict{String, Dict{Date, Float64}}()
                               )::Dict{String, Dict{Date, Float64}}
    rv_regime = if !isempty(rolling_vol)
        regime = compute_stock_rv_regime(rolling_vol)
        n_stocks = length(regime)
        if n_stocks > 0
            println("  Per-stock RV regime: $(n_stocks) tickers")
        end
        regime
    else
        Dict{String, Dict{Date, Float64}}()
    end

    result = Dict{String, Dict{Date, Float64}}()

    for (tk, tk_df) in price_data
        !haskey(heston_ts, tk) && continue
        iv_dict = Dict{Date, Float64}()
        tk_rv = get(rv_regime, tk, Dict{Date, Float64}())

        for d in trading_days
            params = lookup_heston_params(heston_ts, tk, d)
            params === nothing && continue

            regime_scale = get(tk_rv, d, 1.0)
            adjusted = rv_adjusted_params(params, regime_scale)

            S = get_price_on_date(tk_df, d)
            if S !== nothing && S > 0.0
                iv_dict[d] = heston_iv_for_option(Float64(S), Float64(S),
                                                   30.0/365.0, r, adjusted; option_type=:put)
            else
                iv_dict[d] = sqrt(max(adjusted.θ_base, 0.0))
            end
        end

        if !isempty(iv_dict)
            result[tk] = iv_dict
        end
    end

    n_total = length(price_data)
    n_done = length(result)
    if n_done < n_total
        missing_tks = [tk for tk in keys(price_data) if !haskey(result, tk)]
        @warn "Tickers without Heston calibration (no IV generated): $missing_tks"
    end
    rv_status = isempty(rv_regime) ? "no RV adjustment" : "per-stock RV-adjusted θ_base"
    println("  Heston IV map: $(n_done)/$(n_total) tickers ($(rv_status))")
    return result
end
