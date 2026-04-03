"""
Heston Stochastic Volatility Model — IV Surface Generation

All IV is model-derived via the Heston (1993) framework.

Model:
  dS = μ S dt + √v S dW₁
  dv = κ(θ - v)dt + ξ√v dW₂
  Corr(dW₁, dW₂) = ρ

Parameters:
  v₀  — current instantaneous variance (daily-updated from VIX signal)
  κ   — mean-reversion speed of variance (from rolling calibration)
  θ   — long-run variance level (from rolling calibration)
  ξ   — vol-of-vol (from rolling calibration)
  ρ   — correlation between stock and variance (from rolling calibration)

Calibration architecture:
  - κ, θ, ξ, ρ: slow-moving parameters from `calibrate_heston.jl` (every N days)
  - v₀: updated daily using VIX (VXX proxy) signal to reflect current market fear
    v₀_live = v₀_calibrated × vix_regime,  where
    vix_regime = current_VXX_RV / expanding_median_VXX_RV  ∈ [0.5, 2.0]

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
    HestonCalibration

Per-ticker, per-date Heston parameters calibrated from real market option data.
All five parameters (v₀, κ, θ, ξ, ρ) come directly from fitting to option prices.
No hardcoded defaults, no VRP.
"""
struct HestonCalibration
    v0::Float64
    kappa::Float64
    theta::Float64
    xi::Float64
    rho::Float64
end

const HESTON_CAL_PATH = joinpath(_PATH_TO_DATA, "heston_params.csv")

"""
    calibrate_heston_from_options(S, r, option_data; q) -> HestonParams

Calibrate all 5 Heston parameters by minimizing SSE between Heston model
prices and real market mid-prices via grid search.

Input: a set of (K, T, market_price, option_type) observations from a single day.
"""
function calibrate_heston_from_options(S::Float64, r::Float64,
                                        option_data::Vector;
                                        q::Float64=0.0)::HestonParams
    length(option_data) < 3 && error("Need ≥3 option observations, got $(length(option_data))")

    atm_opts = filter(o -> abs(o.K - S) / S < 0.05, option_data)
    v0_est = if !isempty(atm_opts)
        avg_mid = mean(o.market_price for o in atm_opts)
        avg_T = mean(o.T for o in atm_opts)
        max((avg_mid / S / 0.4)^2 / max(avg_T, 0.01), 0.005^2)
    else
        (0.25)^2
    end
    v0_est = clamp(v0_est, 0.005^2, 2.0^2)

    best_params = HestonParams(v0_est, 2.0, v0_est, 0.5, -0.7)
    best_error = Inf

    v0_grid  = [v0_est * f for f in [0.5, 0.75, 1.0, 1.5, 2.0]]
    kap_grid = [0.5, 1.0, 2.0, 4.0, 8.0]
    tht_grid = [v0_est * f for f in [0.5, 1.0, 1.5]]
    xi_grid  = [0.2, 0.5, 0.8]
    rho_grid = [-0.9, -0.7, -0.5, -0.3]

    for v0 in v0_grid, kap in kap_grid, tht in tht_grid, xi in xi_grid, rho in rho_grid
        params = HestonParams(v0, kap, tht, xi, rho)
        total_err = 0.0
        valid = true
        for opt in option_data
            try
                model_price = opt.option_type == :call ?
                    heston_call_price(S, opt.K, opt.T, r, params; q=q) :
                    heston_put_price(S, opt.K, opt.T, r, params; q=q)
                total_err += (model_price - opt.market_price)^2
            catch
                valid = false; break
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
    load_heston_params(path) -> Dict{String, Dict{Date, HestonCalibration}}

Load rolling Heston calibration time-series produced by `calibrate_heston.jl`.
Returns: ticker → (date → HestonCalibration)
"""
function load_heston_params(path::String)::Dict{String, Dict{Date, HestonCalibration}}
    df = CSV.read(path, DataFrame)
    result = Dict{String, Dict{Date, HestonCalibration}}()
    for row in eachrow(df)
        tk = row.ticker
        d = Date(row.date)
        cal = HestonCalibration(row.v0, row.kappa, row.theta, row.xi, row.rho)
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
    lookup_heston_params(heston_ts, ticker, date) -> HestonParams

Find the most recent calibration on or before `date` for `ticker`.
"""
function lookup_heston_params(heston_ts::Dict{String, Dict{Date, HestonCalibration}},
                                ticker::String, date::Date)::Union{HestonParams, Nothing}
    !haskey(heston_ts, ticker) && return nothing
    cal_dates = sort(collect(keys(heston_ts[ticker])))
    isempty(cal_dates) && return nothing
    idx = searchsortedlast(cal_dates, date)
    idx == 0 && return nothing
    cal = heston_ts[ticker][cal_dates[idx]]
    return HestonParams(cal.v0, cal.kappa, cal.theta, cal.xi, cal.rho)
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
    compute_vix_regime_series(vix_df; window=20) -> Dict{Date, Float64}

Compute a daily VIX regime multiplier from VXX (iPath VIX ETN) price data.

For each trading day, computes VXX's trailing `window`-day realized volatility
and divides by the expanding historical median. The result is clamped to [0.5, 2.0].

  regime > 1.0 → market more fearful than usual → scale v₀ up
  regime < 1.0 → market calmer than usual → scale v₀ down
  regime ≈ 1.0 → normal conditions
"""
function compute_vix_regime_series(vix_df::DataFrame; window::Int=20)::Dict{Date, Float64}
    nrow(vix_df) < window + 1 && return Dict{Date, Float64}()

    prices = vix_df.close
    dates = vix_df.date

    rv_series = Float64[]
    rv_dates = Date[]
    for i in (window+1):length(prices)
        lr = log.(prices[(i-window+1):i] ./ prices[(i-window):(i-1)])
        push!(rv_series, std(lr) * sqrt(252))
        push!(rv_dates, dates[i])
    end

    isempty(rv_series) && return Dict{Date, Float64}()

    result = Dict{Date, Float64}()
    for (j, d) in enumerate(rv_dates)
        expanding_med = median(rv_series[1:j])
        regime = clamp(rv_series[j] / max(expanding_med, 0.01), 0.5, 2.0)
        result[d] = regime
    end

    return result
end

"""
    vix_adjusted_params(params, vix_regime) -> HestonParams

Return a copy of `params` with v₀ scaled by the VIX regime multiplier.
κ, θ, ξ, ρ remain unchanged (slow-moving, from periodic calibration).
"""
function vix_adjusted_params(params::HestonParams, vix_regime::Float64)::HestonParams
    return HestonParams(params.v0 * vix_regime, params.kappa, params.theta, params.xi, params.rho)
end

"""
    build_heston_iv_map(price_data, trading_days;
                         r, heston_ts, vix_data) -> Dict{String, Dict{Date, Float64}}

Build rolling IV map using pre-calibrated Heston parameters from
`calibrate_heston.jl`, with v₀ daily-adjusted by the VIX regime signal.

κ, θ, ξ, ρ come from `heston_params.csv` (calibrated every N trading days).
v₀ is scaled each day by the VIX regime multiplier derived from VXX data:
    v₀_live = v₀_calibrated × vix_regime(date)
"""
function build_heston_iv_map(price_data::Dict{String, DataFrame},
                               trading_days::Vector{Date};
                               r::Float64=0.045,
                               heston_ts::Dict{String, Dict{Date, HestonCalibration}}=Dict{String, Dict{Date, HestonCalibration}}(),
                               vix_data::Union{Nothing, DataFrame}=nothing
                               )::Dict{String, Dict{Date, Float64}}
    vix_regime = if vix_data !== nothing && nrow(vix_data) > 20
        series = compute_vix_regime_series(vix_data)
        if !isempty(series)
            println("  VIX regime series: $(length(series)) days, range [$(round(minimum(values(series)), digits=2)), $(round(maximum(values(series)), digits=2))]")
        end
        series
    else
        Dict{Date, Float64}()
    end

    result = Dict{String, Dict{Date, Float64}}()

    for (tk, tk_df) in price_data
        !haskey(heston_ts, tk) && continue
        iv_dict = Dict{Date, Float64}()

        for d in trading_days
            params = lookup_heston_params(heston_ts, tk, d)
            params === nothing && continue

            regime_scale = get(vix_regime, d, 1.0)
            adjusted = vix_adjusted_params(params, regime_scale)

            S = get_price_on_date(tk_df, d)
            if S !== nothing && S > 0.0
                iv_dict[d] = heston_implied_vol(Float64(S), Float64(S), 30.0/365.0, r, adjusted; option_type=:put)
            else
                iv_dict[d] = sqrt(adjusted.v0)
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
    vix_status = isempty(vix_regime) ? "no VIX adjustment" : "VIX-adjusted v₀"
    println("  Heston IV map: $(n_done)/$(n_total) tickers ($(vix_status))")
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
