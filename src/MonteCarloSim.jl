"""
Monte Carlo simulation for Wheel strategy stress testing.

Implements GBM path simulation and stress scenarios per Varner PDF Section 7B:
  "Price & volatility: regime-switching GBM or simple stochastic-vol model."
  "Stress: name-specific −30% gap; market-wide volatility spike; liquidity thinning."

GBM formula (Reference: CHEME-5660 Week 5b):
  S_{t+Δt} = S_t · exp[(μ - σ²/2)·Δt + σ·√Δt · Z],   Z ~ N(0,1)

Enhancements (TODO items 10, 11, 16):
  - Correlated multi-asset GBM via Cholesky decomposition (CHEME-5660 Week 6)
  - 2-state HMM regime detection via EM algorithm (CHEME-5660 Week 13)
  - Fixed gap scenario propagation bug
"""

"""
    simulate_gbm(S₀, μ, σ, T; Δt=1/252, n_paths=1000) -> Matrix{Float64}

Generate price paths via Geometric Brownian Motion (CHEME-5660 Week 5b).
Returns matrix of size (n_steps+1, n_paths). Row 1 = S₀ for all paths.
"""
function simulate_gbm(S₀::Float64, μ::Float64, σ::Float64, T::Float64;
                       Δt::Float64=1.0/252.0, n_paths::Int=1000)::Matrix{Float64}
    n_steps = ceil(Int, T / Δt)
    paths = Matrix{Float64}(undef, n_steps + 1, n_paths)
    paths[1, :] .= S₀

    drift = (μ - 0.5 * σ^2) * Δt
    diffusion = σ * sqrt(Δt)

    for p in 1:n_paths
        for t in 2:(n_steps + 1)
            Z = randn()
            paths[t, p] = paths[t-1, p] * exp(drift + diffusion * Z)
        end
    end
    return paths
end

"""
    simulate_regime_gbm(...) -> Matrix{Float64}

Two-state Markov regime-switching GBM.
Self-designed, motivated by Varner PDF Section 7B.
"""
function simulate_regime_gbm(S₀::Float64,
                              μ_normal::Float64, σ_normal::Float64,
                              μ_stressed::Float64, σ_stressed::Float64,
                              p_to_stressed::Float64, p_to_normal::Float64,
                              T::Float64;
                              Δt::Float64=1.0/252.0, n_paths::Int=1000)::Matrix{Float64}
    n_steps = ceil(Int, T / Δt)
    paths = Matrix{Float64}(undef, n_steps + 1, n_paths)
    paths[1, :] .= S₀

    for p in 1:n_paths
        stressed = false
        for t in 2:(n_steps + 1)
            if stressed
                stressed = rand() > p_to_normal
            else
                stressed = rand() < p_to_stressed
            end

            μ = stressed ? μ_stressed : μ_normal
            σ = stressed ? σ_stressed : σ_normal
            drift = (μ - 0.5 * σ^2) * Δt
            diffusion = σ * sqrt(Δt)
            Z = randn()
            paths[t, p] = paths[t-1, p] * exp(drift + diffusion * Z)
        end
    end
    return paths
end

# ── Correlated Multi-Asset GBM (TODO item 10) ────────────────────────────────

"""
    simulate_correlated_gbm(S₀, μ, σ, ρ, T; Δt=1/252, n_paths=1000) -> Array{Float64, 3}

Generate correlated price paths for multiple assets via Cholesky decomposition.
Reference: CHEME-5660 Week 6 — Multiple-Asset GBM.

Arguments:
  S₀ — initial prices (n_assets,)
  μ  — annualized drifts (n_assets,)
  σ  — annualized volatilities (n_assets,)
  ρ  — return correlation matrix (n_assets × n_assets)

Returns array of size (n_steps+1, n_assets, n_paths).
Correlated normals: Z_corr = L · Z_indep, where L = cholesky(ρ).L
"""
function simulate_correlated_gbm(S₀::Vector{Float64}, μ::Vector{Float64},
                                   σ::Vector{Float64}, ρ::Matrix{Float64},
                                   T::Float64;
                                   Δt::Float64=1.0/252.0, n_paths::Int=1000)::Array{Float64, 3}
    n_assets = length(S₀)
    n_steps = ceil(Int, T / Δt)

    ρ_pd = ρ + 1e-8 * I
    L = try
        cholesky(Hermitian(ρ_pd)).L
    catch
        Matrix{Float64}(I, n_assets, n_assets)
    end

    paths = Array{Float64, 3}(undef, n_steps + 1, n_assets, n_paths)
    for j in 1:n_assets
        paths[1, j, :] .= S₀[j]
    end

    for p in 1:n_paths
        for t in 2:(n_steps + 1)
            Z_indep = randn(n_assets)
            Z_corr = L * Z_indep
            for j in 1:n_assets
                drift = (μ[j] - 0.5 * σ[j]^2) * Δt
                diffusion = σ[j] * sqrt(Δt) * Z_corr[j]
                paths[t, j, p] = paths[t-1, j, p] * exp(drift + diffusion)
            end
        end
    end
    return paths
end

# ── 2-State HMM Regime Detection (TODO item 11) ──────────────────────────────

"""
    fit_two_state_hmm(returns; max_iter=100, tol=1e-6) -> NamedTuple

Fit a 2-state Gaussian Hidden Markov Model to return data using EM (Baum-Welch).
Reference: CHEME-5660 Week 13 — Markov Models and HMM.

State 1: "normal" regime (typically higher μ, lower σ)
State 2: "stressed" regime (typically lower μ, higher σ)

Returns: (μ₁, σ₁, μ₂, σ₂, p_12, p_21, log_likelihood)
"""
function fit_two_state_hmm(returns::Vector{Float64}; max_iter::Int=100, tol::Float64=1e-6)
    T = length(returns)
    T < 10 && error("Need at least 10 observations for HMM fitting")

    sorted_r = sort(returns)
    mid = div(T, 2)
    μ = [mean(sorted_r[1:mid]), mean(sorted_r[mid+1:end])]
    σ² = [max(var(sorted_r[1:mid]), 1e-10), max(var(sorted_r[mid+1:end]), 1e-10)]
    A = [0.95 0.05; 0.10 0.90]
    π₀ = [0.5, 0.5]

    _gauss(x, m, v) = exp(-0.5 * (x - m)^2 / v) / sqrt(2π * v)

    prev_ll = -Inf

    for iter in 1:max_iter
        # E-step: forward
        α = Matrix{Float64}(undef, T, 2)
        for k in 1:2
            α[1, k] = π₀[k] * _gauss(returns[1], μ[k], σ²[k])
        end
        s = sum(α[1, :]); s > 0 && (α[1, :] ./= s)
        scale_factors = [s]

        for t in 2:T
            for k in 1:2
                α[t, k] = sum(α[t-1, j] * A[j, k] for j in 1:2) * _gauss(returns[t], μ[k], σ²[k])
            end
            s = sum(α[t, :]); s > 0 && (α[t, :] ./= s)
            push!(scale_factors, s)
        end

        # E-step: backward
        β = Matrix{Float64}(undef, T, 2)
        β[T, :] .= 1.0
        for t in (T-1):-1:1
            for j in 1:2
                β[t, j] = sum(A[j, k] * _gauss(returns[t+1], μ[k], σ²[k]) * β[t+1, k] for k in 1:2)
            end
            s = scale_factors[t+1]; s > 0 && (β[t, :] ./= s)
        end

        # Responsibilities
        γ = α .* β
        for t in 1:T
            s = sum(γ[t, :]); s > 0 && (γ[t, :] ./= s)
        end

        # Transition counts
        ξ = zeros(2, 2)
        for t in 1:(T-1)
            denom = 0.0
            for j in 1:2, k in 1:2
                denom += α[t, j] * A[j, k] * _gauss(returns[t+1], μ[k], σ²[k]) * β[t+1, k]
            end
            denom = max(denom, 1e-300)
            for j in 1:2, k in 1:2
                ξ[j, k] += α[t, j] * A[j, k] * _gauss(returns[t+1], μ[k], σ²[k]) * β[t+1, k] / denom
            end
        end

        # M-step
        for k in 1:2
            wk = sum(γ[:, k])
            wk = max(wk, 1e-10)
            μ[k] = sum(γ[t, k] * returns[t] for t in 1:T) / wk
            σ²[k] = max(sum(γ[t, k] * (returns[t] - μ[k])^2 for t in 1:T) / wk, 1e-10)
        end
        for j in 1:2
            row_sum = max(sum(ξ[j, :]), 1e-10)
            A[j, :] .= ξ[j, :] ./ row_sum
        end
        π₀ = γ[1, :]

        ll = sum(log(max(sf, 1e-300)) for sf in scale_factors)
        abs(ll - prev_ll) < tol && break
        prev_ll = ll
    end

    idx_normal = σ²[1] < σ²[2] ? 1 : 2
    idx_stressed = 3 - idx_normal

    return (
        μ_normal    = μ[idx_normal],
        σ_normal    = sqrt(σ²[idx_normal]),
        μ_stressed  = μ[idx_stressed],
        σ_stressed  = sqrt(σ²[idx_stressed]),
        p_to_stressed = A[idx_normal, idx_stressed],
        p_to_normal   = A[idx_stressed, idx_normal],
        log_likelihood = prev_ll,
    )
end

"""
    classify_regime(recent_returns, hmm_params; window=20) -> Symbol

Classify current market regime (:normal or :stressed) using fitted HMM parameters.
Computes log-likelihood of recent returns under each regime's Gaussian.
"""
function classify_regime(recent_returns::Vector{Float64}, hmm_params; window::Int=20)::Symbol
    n = min(length(recent_returns), window)
    n < 5 && return :normal
    r = recent_returns[end-n+1:end]

    ll_normal = sum(-0.5 * ((x - hmm_params.μ_normal) / hmm_params.σ_normal)^2 -
                    log(hmm_params.σ_normal) for x in r)
    ll_stressed = sum(-0.5 * ((x - hmm_params.μ_stressed) / hmm_params.σ_stressed)^2 -
                      log(hmm_params.σ_stressed) for x in r)

    return ll_stressed > ll_normal ? :stressed : :normal
end

# ── Stress Scenarios ──────────────────────────────────────────────────────────

struct StressScenario
    name::String
    μ_override::Float64
    σ_multiplier::Float64
    gap_pct::Float64
    gap_day::Int
end

const DEFAULT_STRESS_SCENARIOS = [
    StressScenario("Normal",          0.0,  1.0,  0.0,  0),
    StressScenario("Vol Spike",       0.0,  2.0,  0.0,  0),
    StressScenario("Bear Market",    -0.20, 1.5,  0.0,  0),
    StressScenario("Flash Crash",     0.0,  2.0, -0.10, 30),
    StressScenario("Name Blowup",     0.0,  1.5, -0.30, 60),
    StressScenario("Bull Squeeze",    0.30, 1.5,  0.15, 45),
]

"""
    run_stress_scenarios(S₀, μ, σ, T; scenarios, n_paths, Δt) -> DataFrame

Run GBM simulation under each stress scenario and report distribution statistics.
Gap scenario bug fixed (TODO item 16): gaps now multiply all rows from gap_day onward
rather than incorrectly recomputing ratios from modified data.
"""
function run_stress_scenarios(S₀::Float64, μ::Float64, σ::Float64, T::Float64;
                               scenarios::Vector{StressScenario}=DEFAULT_STRESS_SCENARIOS,
                               n_paths::Int=5000, Δt::Float64=1.0/252.0)::DataFrame
    results = DataFrame(
        Scenario=String[], MeanReturn=Float64[], MedianReturn=Float64[],
        VaR95=Float64[], MaxDD=Float64[], PctBelow80=Float64[]
    )

    for sc in scenarios
        μ_eff = μ + sc.μ_override
        σ_eff = σ * sc.σ_multiplier

        paths = simulate_gbm(S₀, μ_eff, σ_eff, T; Δt=Δt, n_paths=n_paths)

        if sc.gap_pct != 0.0 && sc.gap_day > 0 && sc.gap_day < size(paths, 1)
            gap_factor = 1.0 + sc.gap_pct
            for t in sc.gap_day:size(paths, 1)
                for p in 1:n_paths
                    paths[t, p] *= gap_factor
                end
            end
        end

        terminal = paths[end, :]
        returns = (terminal .- S₀) ./ S₀

        max_dds = Float64[]
        for p in 1:n_paths
            pk = -Inf
            mdd = 0.0
            for t in 1:size(paths, 1)
                pk = max(pk, paths[t, p])
                mdd = max(mdd, (pk - paths[t, p]) / pk)
            end
            push!(max_dds, mdd)
        end

        sorted_ret = sort(returns)
        var_idx = max(1, floor(Int, 0.05 * length(sorted_ret)))

        push!(results, (
            sc.name,
            round(mean(returns) * 100, digits=2),
            round(median(returns) * 100, digits=2),
            round(-sorted_ret[var_idx] * 100, digits=2),
            round(mean(max_dds) * 100, digits=2),
            round(count(r -> r < -0.20, returns) / n_paths * 100, digits=2)
        ))
    end

    return results
end

"""
    apply_stress_to_prices(price_data; vol_mult=1.0, drift_adj=0.0, gap_pct=0.0, gap_day=0) -> Dict

Apply stress scenario to historical price data for portfolio-level stress testing (TODO item 15).
Transforms daily returns and returns modified price DataFrames.
"""
function apply_stress_to_prices(price_data::Dict{String, DataFrame};
                                  vol_mult::Float64=1.0, drift_adj::Float64=0.0,
                                  gap_pct::Float64=0.0, gap_day::Int=0)::Dict{String, DataFrame}
    stressed = Dict{String, DataFrame}()
    for (ticker, df) in price_data
        sdf = copy(df)
        prices = copy(sdf.adj_close)
        for i in 2:length(prices)
            orig_ret = log(prices[i] / prices[i-1])
            mean_ret = drift_adj / 252.0
            stressed_ret = mean_ret + (orig_ret - mean_ret) * vol_mult
            prices[i] = prices[i-1] * exp(stressed_ret)
        end
        if gap_pct != 0.0 && gap_day > 0 && gap_day < length(prices)
            factor = 1.0 + gap_pct
            for i in gap_day:length(prices)
                prices[i] *= factor
            end
        end
        sdf.adj_close = prices
        sdf.close = prices
        stressed[ticker] = sdf
    end
    return stressed
end

"""
    mc_summary(paths) -> NamedTuple

Quick summary statistics for a path matrix from simulate_gbm.
"""
function mc_summary(paths::Matrix{Float64})
    terminal = paths[end, :]
    S₀ = paths[1, 1]
    returns = (terminal .- S₀) ./ S₀
    sorted_ret = sort(returns)
    var_idx = max(1, floor(Int, 0.05 * length(sorted_ret)))

    return (
        mean_return = mean(returns),
        std_return = std(returns),
        median_return = median(returns),
        var_95 = -sorted_ret[var_idx],
        min_return = minimum(returns),
        max_return = maximum(returns),
    )
end
