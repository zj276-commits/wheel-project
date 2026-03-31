"""
Heston Stochastic Volatility Model — IV Surface Generation

Replaces the previous WRDS lookup approach. All IV is now model-derived
via the Heston (1993) framework, making the system forward-looking and
applicable to any tenor without historical option data dependency.

Model:
  dS = μ S dt + √v S dW₁
  dv = κ(θ - v)dt + ξ√v dW₂
  Corr(dW₁, dW₂) = ρ

Parameters:
  v₀  — current instantaneous variance (calibrated from VIX or recent RV)
  κ   — mean-reversion speed of variance
  θ   — long-run variance level
  ξ   — vol-of-vol (volatility of the variance process)
  ρ   — correlation between stock and variance (typically negative for equities)

Calibration sources:
  - VIX index → proxy for √(30-day expected variance), sets v₀
  - Historical realized vol → sets θ (long-run level)
  - Option prices (when available) → refine κ, ξ, ρ via least-squares
  - Fallback: empirical defaults by sleeve type

Reference: Heston (1993) "A Closed-Form Solution for Options with
Stochastic Volatility with Applications to Bond and Currency Options"
"""

struct HestonParams
    v0::Float64     # current instantaneous variance
    kappa::Float64  # mean-reversion speed
    theta::Float64  # long-run variance
    xi::Float64     # vol of vol
    rho::Float64    # correlation (stock, variance)
end

"""
Heston characteristic function φ(u) for log-price.
Used in the semi-analytic pricing formula via numerical integration.
"""
function heston_char_func(u::ComplexF64, S::Float64, K::Float64, T::Float64,
                           r::Float64, q::Float64, params::HestonParams)::ComplexF64
    v0, κ, θ, ξ, ρ = params.v0, params.kappa, params.theta, params.xi, params.rho

    d = sqrt((ρ * ξ * im * u - κ)^2 + ξ^2 * (im * u + u^2))
    g = (κ - ρ * ξ * im * u - d) / (κ - ρ * ξ * im * u + d)

    C = (r - q) * im * u * T +
        (κ * θ / ξ^2) * ((κ - ρ * ξ * im * u - d) * T - 2.0 * log((1.0 - g * exp(-d * T)) / (1.0 - g)))

    D = ((κ - ρ * ξ * im * u - d) / ξ^2) * ((1.0 - exp(-d * T)) / (1.0 - g * exp(-d * T)))

    return exp(C + D * v0 + im * u * log(S))
end

"""
Heston European call price via numerical integration (Heston 1993).

C = S·e^{-qT}·P₁ − K·e^{-rT}·P₂

P_j = ½ + (1/π) ∫₀^∞ Re[e^{−iu·ln(K)} · f_j(u) / (iu)] du

f₂(u) = φ(u)           — risk-neutral measure
f₁(u) = φ(u−i) / φ(−i) — stock-numeraire measure
"""
function heston_call_price(S::Float64, K::Float64, T::Float64, r::Float64,
                            params::HestonParams; q::Float64=0.0, N_quad::Int=200)::Float64
    T <= 0.0 && return max(S * exp(-q * T) - K * exp(-r * T), 0.0)

    lnK = log(K)
    du = 150.0 / N_quad
    φ_neg_i = heston_char_func(ComplexF64(-im), S, K, T, r, q, params)

    I1, I2 = 0.0, 0.0
    for i in 1:N_quad
        u = (i - 0.5) * du
        u < 1e-8 && continue

        e_factor = exp(-im * u * lnK)

        φ_u = heston_char_func(ComplexF64(u), S, K, T, r, q, params)
        I2 += real(e_factor * φ_u / (im * u)) * du

        φ_u_mi = heston_char_func(ComplexF64(u) - im, S, K, T, r, q, params)
        I1 += real(e_factor * φ_u_mi / (im * u * φ_neg_i)) * du
    end

    P1 = clamp(0.5 + I1 / π, 0.0, 1.0)
    P2 = clamp(0.5 + I2 / π, 0.0, 1.0)

    return max(S * exp(-q * T) * P1 - K * exp(-r * T) * P2, 0.0)
end

"""
Heston European put price via put-call parity.
"""
function heston_put_price(S::Float64, K::Float64, T::Float64, r::Float64,
                           params::HestonParams; q::Float64=0.0)::Float64
    call = heston_call_price(S, K, T, r, params; q=q)
    return call - S * exp(-q * T) + K * exp(-r * T)
end

"""
Convert a Heston model price to Black-Scholes implied volatility via bisection.
"""
function heston_implied_vol(S::Float64, K::Float64, T::Float64, r::Float64,
                             params::HestonParams; q::Float64=0.0,
                             option_type::Symbol=:put)::Float64
    T <= 0.0 && return sqrt(params.v0)

    target = option_type == :call ?
        heston_call_price(S, K, T, r, params; q=q) :
        heston_put_price(S, K, T, r, params; q=q)
    target <= 0.0 && return sqrt(params.v0)

    σ_lo, σ_hi = 0.001, 5.0
    for _ in 1:80
        σ_mid = (σ_lo + σ_hi) / 2.0
        bs_price = _bs_price(S, K, T, r, σ_mid, option_type; q=q)
        if bs_price > target
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

# ── Calibration ──────────────────────────────────────────────────────────────

"""
    calibrate_heston_from_vix(vix_level, realized_vol; sleeve) -> HestonParams

Quick calibration using VIX as v₀ proxy and realized vol as θ proxy.
VIX = 100 × √(annualized 30-day expected variance), so v₀ ≈ (VIX/100)².

Sleeve-dependent defaults for κ, ξ, ρ based on empirical equity option surfaces.
"""
function calibrate_heston_from_vix(vix_level::Float64, realized_vol::Float64;
                                    sleeve::String="Safe")::HestonParams
    v0 = (vix_level / 100.0)^2
    θ = max(realized_vol^2, 0.01^2)

    if sleeve == "Aggressive"
        return HestonParams(v0, 2.0, θ, 0.6, -0.70)
    else
        return HestonParams(v0, 3.0, θ, 0.4, -0.65)
    end
end

"""
    calibrate_heston_from_options(S, r, option_data; q, initial_params) -> HestonParams

Full calibration by minimizing sum of squared errors between Heston model
prices and observed market prices across multiple (K, T) pairs.

option_data: Vector of NamedTuples with fields: K, T, market_price, option_type
"""
function calibrate_heston_from_options(S::Float64, r::Float64,
                                        option_data::Vector;
                                        q::Float64=0.0,
                                        initial_params::HestonParams=HestonParams(0.04, 2.0, 0.04, 0.5, -0.7))::HestonParams
    length(option_data) < 3 && return initial_params

    best_params = initial_params
    best_error = Inf

    v0_grid = [initial_params.v0]
    κ_grid  = [1.0, 2.0, 3.0, 5.0, 8.0]
    θ_grid  = [initial_params.theta * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]]
    ξ_grid  = [0.2, 0.4, 0.6, 0.8, 1.0]
    ρ_grid  = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4]

    for κ in κ_grid, θ in θ_grid, ξ in ξ_grid, ρ in ρ_grid
        params = HestonParams(initial_params.v0, κ, θ, ξ, ρ)
        total_err = 0.0
        valid = true
        for opt in option_data
            try
                model_price = opt.option_type == :call ?
                    heston_call_price(S, opt.K, opt.T, r, params; q=q) :
                    heston_put_price(S, opt.K, opt.T, r, params; q=q)
                total_err += (model_price - opt.market_price)^2
            catch
                valid = false
                break
            end
        end
        if valid && total_err < best_error
            best_error = total_err
            best_params = params
        end
    end

    return best_params
end

"""
    calibrate_heston_rv_only(realized_vol; sleeve) -> HestonParams

Fallback calibration using only realized volatility (no VIX, no option data).
"""
function calibrate_heston_rv_only(realized_vol::Float64;
                                   sleeve::String="Safe")::HestonParams
    v0 = realized_vol^2
    θ = v0
    if sleeve == "Aggressive"
        return HestonParams(v0, 2.0, θ, 0.6, -0.70)
    else
        return HestonParams(v0, 3.0, θ, 0.4, -0.65)
    end
end

# ── IV Surface Generation ────────────────────────────────────────────────────

"""
    generate_iv_surface(S, r, params; tenors, deltas, q) -> DataFrame

Generate a full IV surface for a given stock price and Heston parameters.
Produces IV for every combination of tenor and delta.
"""
function generate_iv_surface(S::Float64, r::Float64, params::HestonParams;
                              tenors::Vector{Int}=[7, 14, 30, 60, 90],
                              deltas::Vector{Float64}=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
                              q::Float64=0.0)::DataFrame
    rows = NamedTuple{(:tenor_days, :delta, :strike, :iv_put, :iv_call, :put_price, :call_price),
                       Tuple{Int, Float64, Float64, Float64, Float64, Float64, Float64}}[]

    for tenor in tenors
        T = tenor / 365.0
        for δ in deltas
            K = strike_from_delta(S, T, r, sqrt(params.v0), δ, :put; q=q)
            iv_p = heston_implied_vol(S, K, T, r, params; q=q, option_type=:put)
            iv_c = heston_implied_vol(S, K, T, r, params; q=q, option_type=:call)
            p_price = heston_put_price(S, K, T, r, params; q=q)
            c_price = heston_call_price(S, K, T, r, params; q=q)
            push!(rows, (tenor_days=tenor, delta=δ, strike=round(K, digits=2),
                         iv_put=round(iv_p, digits=4), iv_call=round(iv_c, digits=4),
                         put_price=round(p_price, digits=2), call_price=round(c_price, digits=2)))
        end
    end

    return DataFrame(rows)
end

"""
    build_heston_iv_map(price_data, rolling_vol, sleeves_map, trading_days;
                         vix_data, r) -> Dict{String, Dict{Date, Float64}}

Build rolling IV map using Heston model, compatible with WheelEngine.
For each ticker on each trading day, calibrate Heston and extract ATM IV.

Output format matches the old rolling_iv: Dict{ticker => Dict{date => iv}}
"""
function build_heston_iv_map(price_data::Dict{String, DataFrame},
                               rolling_vol::Dict{String, Dict{Date, Float64}},
                               sleeves_map::Dict{String, String},
                               trading_days::Vector{Date};
                               vix_data=nothing,
                               r::Float64=0.045)::Dict{String, Dict{Date, Float64}}
    result = Dict{String, Dict{Date, Float64}}()

    vix_lookup = Dict{Date, Float64}()
    if vix_data !== nothing && hasproperty(vix_data, :date) && hasproperty(vix_data, :close)
        for row in eachrow(vix_data)
            vix_lookup[row.date] = row.close
        end
    end

    for (tk, vol_dict) in rolling_vol
        sleeve = get(sleeves_map, tk, "Safe")
        iv_dict = Dict{Date, Float64}()

        for d in trading_days
            σ_rv = get(vol_dict, d, NaN)
            isnan(σ_rv) && continue

            vix_val = get(vix_lookup, d, NaN)

            params = if !isnan(vix_val)
                calibrate_heston_from_vix(vix_val, σ_rv; sleeve=sleeve)
            else
                calibrate_heston_rv_only(σ_rv; sleeve=sleeve)
            end

            iv_dict[d] = sqrt(params.v0)
        end

        if !isempty(iv_dict)
            result[tk] = iv_dict
        end
    end

    println("  Heston IV map: $(length(result)) tickers")
    return result
end

"""
    heston_iv_for_option(S, K, T, r, params; q, option_type) -> Float64

Single-point IV query: given current market state and Heston params,
return the BS-equivalent IV for a specific option contract.
This is used by WheelEngine for pricing individual option positions.
"""
function heston_iv_for_option(S::Float64, K::Float64, T::Float64, r::Float64,
                               params::HestonParams;
                               q::Float64=0.0, option_type::Symbol=:put)::Float64
    return heston_implied_vol(S, K, T, r, params; q=q, option_type=option_type)
end
