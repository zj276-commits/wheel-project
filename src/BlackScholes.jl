"""
Black-Scholes option pricing and Greeks for the Wheel strategy backtest.
Uses Distributions.jl for the standard normal CDF/quantile.
"""

const _STD_NORMAL = Normal(0.0, 1.0)

function _N(x::Float64)::Float64
    return cdf(_STD_NORMAL, x)
end

function _Ninv(p::Float64)::Float64
    return quantile(_STD_NORMAL, p)
end

"""
    bs_d1(S, K, T, r, σ) -> Float64

Compute d1 in the Black-Scholes formula.
"""
function bs_d1(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64)::Float64
    return (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
end

"""
    bs_call_price(S, K, T, r, σ) -> Float64

European call option price. S=spot, K=strike, T=time to expiry (years),
r=risk-free rate, σ=annualized volatility.
"""
function bs_call_price(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64)::Float64
    if T <= 0.0
        return max(S - K, 0.0)
    end
    d1 = bs_d1(S, K, T, r, σ)
    d2 = d1 - σ * sqrt(T)
    return S * _N(d1) - K * exp(-r * T) * _N(d2)
end

"""
    bs_put_price(S, K, T, r, σ) -> Float64

European put option price via put-call parity.
"""
function bs_put_price(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64)::Float64
    if T <= 0.0
        return max(K - S, 0.0)
    end
    d1 = bs_d1(S, K, T, r, σ)
    d2 = d1 - σ * sqrt(T)
    return K * exp(-r * T) * _N(-d2) - S * _N(-d1)
end

"""
    bs_delta(S, K, T, r, σ, option_type) -> Float64

Option delta. For :call returns N(d1), for :put returns N(d1) - 1.
"""
function bs_delta(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64,
                  option_type::Symbol)::Float64
    if T <= 0.0
        if option_type == :call
            return S > K ? 1.0 : 0.0
        else
            return S < K ? -1.0 : 0.0
        end
    end
    d1 = bs_d1(S, K, T, r, σ)
    if option_type == :call
        return _N(d1)
    else
        return _N(d1) - 1.0
    end
end

"""
    strike_from_delta(S, T, r, σ, target_delta, option_type) -> Float64

Compute the strike price that produces a given Black-Scholes delta.
For puts, target_delta should be the absolute value (e.g. 0.25 for a -0.25 delta put).
"""
function strike_from_delta(S::Float64, T::Float64, r::Float64, σ::Float64,
                           target_delta::Float64, option_type::Symbol)::Float64
    sqrtT = σ * sqrt(T)
    if option_type == :call
        d1 = _Ninv(target_delta)
    else
        d1 = _Ninv(1.0 - target_delta)
    end
    K = S * exp(-d1 * sqrtT + (r + 0.5 * σ^2) * T)
    return round(K, digits=2)
end

"""
    option_price(S, K, T, r, σ, option_type) -> Float64

Unified pricing function dispatching to call or put.
"""
function option_price(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64,
                      option_type::Symbol)::Float64
    if option_type == :call
        return bs_call_price(S, K, T, r, σ)
    else
        return bs_put_price(S, K, T, r, σ)
    end
end
