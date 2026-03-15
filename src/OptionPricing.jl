"""
American option pricing via CRR (Cox-Ross-Rubinstein) binomial lattice.

All pricing uses the American exercise model exclusively. The Wheel strategy
trades American-style equity options where early exercise is possible and must
be accounted for — particularly for short puts on dividend-paying stocks.

Reference (CRR): CHEME-5660 Week 10 — "CRR Factors and American Derivatives Pricing"
  Cox, Ross, Rubinstein (1979).
  u = exp(σ√Δt), d = exp(-σ√Δt) = 1/u.
  Risk-neutral probability: p = (exp((r-q)Δt) - d) / (u - d).
  American value at each node: max(intrinsic, discounted continuation).

Reference (Delta): CHEME-5660 Week 12b — "Delta and Vertical Call Spread"
  Δ = (V_u - V_d) / (S·u - S·d).

Enhancements (TODO items 4, 9, 1):
  - Discrete dividend yield q in CRR lattice (proportional yield model)
  - Greeks: Gamma, Theta, Vega via finite differences on CRR
  - Implied volatility estimation via bisection on CRR price
  - strike_from_delta: self-designed bisection (no course reference)
"""

# ── CRR Binomial Lattice ─────────────────────────────────────────────────────

"""
    crr_price(S, K, T, r, σ, option_type; N=50, q=0.0) -> Float64

American option price via CRR binomial lattice with N time steps.

CRR parameters (CHEME-5660 L10a):
  Δt = T / N
  u  = exp(σ · √Δt)
  d  = exp(-σ · √Δt) = 1/u
  p  = (e^{(r-q)Δt} - d) / (u - d)   — risk-neutral up probability with dividend yield q

Continuous dividend yield q adjusts the risk-neutral drift so that
early exercise around ex-dates is correctly valued for American options.
"""
function crr_price(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64,
                   option_type::Symbol; N::Int=50, q::Float64=0.0)::Float64
    T <= 0.0 && return option_type == :call ? max(S - K, 0.0) : max(K - S, 0.0)
    σ <= 1e-10 && return option_type == :call ? max(S - K * exp(-r * T), 0.0) : max(K - S, 0.0)
    N = max(N, 1)

    dt = T / N
    u = exp(σ * sqrt(dt))
    d = 1.0 / u
    p = clamp((exp((r - q) * dt) - d) / (u - d), 0.001, 0.999)
    inv_disc = exp(-r * dt)

    V = Vector{Float64}(undef, N + 1)
    for j in 0:N
        Sj = S * u^j * d^(N - j)
        V[j+1] = option_type == :call ? max(Sj - K, 0.0) : max(K - Sj, 0.0)
    end

    for i in (N-1):-1:0
        for j in 0:i
            Sj = S * u^j * d^(i - j)
            continuation = inv_disc * (p * V[j+2] + (1.0 - p) * V[j+1])
            intrinsic = option_type == :call ? max(Sj - K, 0.0) : max(K - Sj, 0.0)
            V[j+1] = max(continuation, intrinsic)
        end
    end

    return V[1]
end

"""
    crr_delta(S, K, T, r, σ, option_type; N=50, q=0.0) -> Float64

American option delta from the first level of the CRR lattice.

Reference (CHEME-5660 L12b):
  Δ = (V_u - V_d) / (S·u - S·d)
"""
function crr_delta(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64,
                   option_type::Symbol; N::Int=50, q::Float64=0.0)::Float64
    if T <= 0.0 || σ <= 1e-10
        return option_type == :call ? (S > K ? 1.0 : 0.0) : (S < K ? -1.0 : 0.0)
    end
    N = max(N, 2)

    dt = T / N
    u = exp(σ * sqrt(dt))
    d = 1.0 / u

    S_u = S * u
    S_d = S * d
    V_u = crr_price(S_u, K, T - dt, r, σ, option_type; N=N-1, q=q)
    V_d = crr_price(S_d, K, T - dt, r, σ, option_type; N=N-1, q=q)

    return (V_u - V_d) / (S_u - S_d)
end

# ── Greeks via Finite Differences (CHEME-5660 Week 11) ────────────────────────

"""
    crr_gamma(S, K, T, r, σ, option_type; N=50, q=0.0) -> Float64

Gamma (∂²V/∂S²) via central finite differences on CRR price.
Reference: CHEME-5660 Week 11 — Greeks and risk management.
"""
function crr_gamma(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64,
                   option_type::Symbol; N::Int=50, q::Float64=0.0)::Float64
    (T <= 0.0 || σ <= 1e-10) && return 0.0
    ε = S * 0.01
    V_up = crr_price(S + ε, K, T, r, σ, option_type; N=N, q=q)
    V_mid = crr_price(S, K, T, r, σ, option_type; N=N, q=q)
    V_dn = crr_price(S - ε, K, T, r, σ, option_type; N=N, q=q)
    return (V_up - 2.0 * V_mid + V_dn) / (ε^2)
end

"""
    crr_theta(S, K, T, r, σ, option_type; N=50, q=0.0) -> Float64

Theta (∂V/∂t) per calendar day via forward difference on CRR price.
Returns the change in option value for one day's passage of time.
Reference: CHEME-5660 Week 11 — Greeks and risk management.
"""
function crr_theta(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64,
                   option_type::Symbol; N::Int=50, q::Float64=0.0)::Float64
    dt_day = 1.0 / 365.0
    T <= dt_day && return 0.0
    V_now = crr_price(S, K, T, r, σ, option_type; N=N, q=q)
    V_next = crr_price(S, K, T - dt_day, r, σ, option_type; N=max(N - 1, 1), q=q)
    return V_next - V_now
end

"""
    crr_vega(S, K, T, r, σ, option_type; N=50, q=0.0) -> Float64

Vega (∂V/∂σ) via central finite differences on CRR price.
Returns the change in option value per unit change in implied volatility.
Reference: CHEME-5660 Week 11 — Greeks and risk management.
"""
function crr_vega(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64,
                  option_type::Symbol; N::Int=50, q::Float64=0.0)::Float64
    (T <= 0.0 || σ <= 1e-10) && return 0.0
    Δσ = 0.01
    V_up = crr_price(S, K, T, r, σ + Δσ, option_type; N=N, q=q)
    V_dn = crr_price(S, K, T, r, max(σ - Δσ, 0.001), option_type; N=N, q=q)
    return (V_up - V_dn) / (2.0 * Δσ)
end

"""
    option_greeks(S, K, T, r, σ, option_type; N=50, q=0.0) -> NamedTuple

Compute all Greeks for an American option in a single call.
Returns (delta, gamma, theta, vega).
"""
function option_greeks(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64,
                       option_type::Symbol; N::Int=50, q::Float64=0.0)
    return (
        delta = crr_delta(S, K, T, r, σ, option_type; N=N, q=q),
        gamma = crr_gamma(S, K, T, r, σ, option_type; N=N, q=q),
        theta = crr_theta(S, K, T, r, σ, option_type; N=N, q=q),
        vega  = crr_vega(S, K, T, r, σ, option_type; N=N, q=q),
    )
end

# ── Strike from Delta (Bisection) ────────────────────────────────────────────

"""
    strike_from_delta(S, T, r, σ, target_delta, option_type; N=30, q=0.0) -> Float64

Find the strike K that produces a target |delta| under CRR American pricing.
Self-designed bisection (no direct course reference).
"""
function strike_from_delta(S::Float64, T::Float64, r::Float64, σ::Float64,
                           target_delta::Float64, option_type::Symbol;
                           N::Int=30, q::Float64=0.0)::Float64
    (T <= 1e-10 || σ <= 1e-10) && return round(S, digits=2)

    K_lo = S * 0.50
    K_hi = S * 2.00

    for _ in 1:30
        K_mid = (K_lo + K_hi) / 2.0
        d = crr_delta(S, K_mid, T, r, σ, option_type; N=N, q=q)

        if option_type == :call
            if d > target_delta
                K_lo = K_mid
            else
                K_hi = K_mid
            end
        else
            if abs(d) > target_delta
                K_hi = K_mid
            else
                K_lo = K_mid
            end
        end

        (K_hi - K_lo) < 0.005 && break
    end

    return round((K_lo + K_hi) / 2.0, digits=2)
end

# ── Implied Volatility Estimation ─────────────────────────────────────────────

"""
    estimate_implied_vol(S, K, T, r, market_price, option_type; q=0.0, N=50) -> Float64

Estimate implied volatility by bisection on CRR price.
Motivated by Varner PDF Section 7B: "calibrated to each name's IV surface."
Course reference: VLQuantitativeFinancePackage provides estimate_implied_volatility().
This is a self-designed CRR-based alternative for American options.
"""
function estimate_implied_vol(S::Float64, K::Float64, T::Float64, r::Float64,
                               market_price::Float64, option_type::Symbol;
                               q::Float64=0.0, N::Int=50)::Float64
    market_price <= 0.0 && return 0.01
    T <= 0.0 && return 0.01

    σ_lo, σ_hi = 0.01, 3.0
    for _ in 1:60
        σ_mid = (σ_lo + σ_hi) / 2.0
        model_price = crr_price(S, K, T, r, σ_mid, option_type; N=N, q=q)
        if model_price > market_price
            σ_hi = σ_mid
        else
            σ_lo = σ_mid
        end
        (σ_hi - σ_lo) < 1e-6 && break
    end
    return (σ_lo + σ_hi) / 2.0
end

# ── Unified Pricing API ──────────────────────────────────────────────────────

"""
    option_price(S, K, T, r, σ, option_type; steps=50, q=0.0) -> Float64

American option price via CRR lattice. This is the only pricing function used
by the Wheel backtest engine — all option valuations go through CRR to
correctly account for early exercise and dividend yield.
"""
function option_price(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64,
                      option_type::Symbol; steps::Int=50, q::Float64=0.0)::Float64
    return crr_price(S, K, T, r, σ, option_type; N=steps, q=q)
end
