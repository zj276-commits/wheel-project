"""
Simulation — HMM Regime Detection + GBM Path Simulation

Contains:
  1. HMM regime detection — Student-t emissions, Baum-Welch EM, Viterbi decoding
     (following Alswaidan & Varner, "A Hidden Markov Model for Modeling Equity")
  2. GBM path simulation (CHEME-5660 Week 5b)
  3. Regime-switching GBM, correlated multi-asset GBM (Cholesky, Week 6)
  4. Earnings jump diffusion with Poisson intensity (Merton 1976)
  5. Stress scenarios

HMM approach per Alswaidan-Varner paper:
  - K regimes (default K=3: bull/sideways/bear) with Student-t(ν=5) emissions
  - Baum-Welch EM for parameter estimation (π, A, emission params)
  - Viterbi algorithm for most-likely state sequence decoding
  - Grid search over K ∈ {2,3,4} with BIC model selection
  - Jump-diffusion overlay: Poisson(λ)-triggered jumps with LogNormal magnitude

GBM formula:  S_{t+Δt} = S_t · exp[(μ - σ²/2)·Δt + σ·√Δt · Z],  Z ~ N(0,1)
"""

# ── HMM Regime Detection (Alswaidan-Varner) ─────────────────────────────────

struct HMMRegimeModel
    K::Int                          # number of regimes
    π::Vector{Float64}              # initial state distribution
    A::Matrix{Float64}              # K×K transition matrix
    μ::Vector{Float64}              # emission means per regime
    σ::Vector{Float64}              # emission stdevs per regime
    ν::Float64                      # Student-t degrees of freedom
    log_likelihood::Float64
    bic::Float64
    viterbi_path::Vector{Int}
end

"""
Student-t log-pdf: log f(x | μ, σ, ν)
"""
function student_t_logpdf(x::Float64, μ::Float64, σ::Float64, ν::Float64)::Float64
    z = (x - μ) / σ
    return lgamma((ν + 1) / 2) - lgamma(ν / 2) - 0.5 * log(ν * π) - log(σ) -
           ((ν + 1) / 2) * log(1 + z^2 / ν)
end

"""
    baum_welch_em(returns, K; ν, max_iter, tol) -> HMMRegimeModel

Baum-Welch EM algorithm with Student-t(ν) emissions.
Per Alswaidan-Varner: uses scaled forward-backward to avoid underflow.
"""
function baum_welch_em(returns::Vector{Float64}, K::Int;
                        ν::Float64=5.0, max_iter::Int=200, tol::Float64=1e-6)::HMMRegimeModel
    T_len = length(returns)
    T_len < 2 * K && error("Need at least $(2K) observations for $K-state HMM")

    sorted_ret = sort(returns)
    μ = [sorted_ret[max(1, round(Int, (2k-1) * T_len / (2K)))] for k in 1:K]
    sort!(μ)
    σ = fill(std(returns), K)
    π_init = fill(1.0 / K, K)
    A = fill(1.0 / K, K, K)
    for k in 1:K
        A[k, k] = max(0.8, 1.0 - 0.1 * (K - 1))
        off_diag = (1.0 - A[k, k]) / max(K - 1, 1)
        for j in 1:K
            j != k && (A[k, j] = off_diag)
        end
    end

    prev_ll = -Inf

    for iter in 1:max_iter
        # E-step: scaled forward-backward
        log_B = Matrix{Float64}(undef, T_len, K)
        for t in 1:T_len, k in 1:K
            log_B[t, k] = student_t_logpdf(returns[t], μ[k], σ[k], ν)
        end
        B = exp.(log_B .- maximum(log_B, dims=2))

        α = Matrix{Float64}(undef, T_len, K)
        c = Vector{Float64}(undef, T_len)

        α[1, :] .= π_init .* B[1, :]
        c[1] = sum(α[1, :])
        c[1] < 1e-300 && (c[1] = 1e-300)
        α[1, :] ./= c[1]

        for t in 2:T_len
            for j in 1:K
                α[t, j] = sum(α[t-1, i] * A[i, j] for i in 1:K) * B[t, j]
            end
            c[t] = sum(α[t, :])
            c[t] < 1e-300 && (c[t] = 1e-300)
            α[t, :] ./= c[t]
        end

        β = Matrix{Float64}(undef, T_len, K)
        β[T_len, :] .= 1.0

        for t in (T_len-1):-1:1
            for i in 1:K
                β[t, i] = sum(A[i, j] * B[t+1, j] * β[t+1, j] for j in 1:K)
            end
            s = sum(β[t, :])
            s < 1e-300 && (s = 1e-300)
            β[t, :] ./= s
        end

        γ = α .* β
        for t in 1:T_len
            s = sum(γ[t, :])
            s < 1e-300 && (s = 1e-300)
            γ[t, :] ./= s
        end

        ξ = Array{Float64, 3}(undef, T_len - 1, K, K)
        for t in 1:(T_len-1)
            denom = 0.0
            for i in 1:K, j in 1:K
                denom += α[t, i] * A[i, j] * B[t+1, j] * β[t+1, j]
            end
            denom < 1e-300 && (denom = 1e-300)
            for i in 1:K, j in 1:K
                ξ[t, i, j] = α[t, i] * A[i, j] * B[t+1, j] * β[t+1, j] / denom
            end
        end

        ll = sum(log.(c))

        # M-step
        π_init .= γ[1, :]
        s_pi = sum(π_init)
        s_pi > 0 && (π_init ./= s_pi)

        for i in 1:K
            γ_sum_1 = sum(γ[t, i] for t in 1:(T_len-1))
            γ_sum_1 < 1e-300 && (γ_sum_1 = 1e-300)
            for j in 1:K
                A[i, j] = sum(ξ[t, i, j] for t in 1:(T_len-1)) / γ_sum_1
            end
            row_s = sum(A[i, :])
            row_s > 0 && (A[i, :] ./= row_s)
        end

        for k in 1:K
            γ_sum = sum(γ[t, k] for t in 1:T_len)
            γ_sum < 1e-300 && (γ_sum = 1e-300)

            w = (ν + 1) ./ (ν .+ ((returns .- μ[k]) ./ σ[k]) .^ 2)
            wγ = [w[t] * γ[t, k] for t in 1:T_len]
            denom = sum(wγ)
            denom < 1e-300 && (denom = 1e-300)

            μ[k] = sum(wγ[t] * returns[t] for t in 1:T_len) / denom
            σ[k] = sqrt(sum(wγ[t] * (returns[t] - μ[k])^2 for t in 1:T_len) / γ_sum)
            σ[k] = max(σ[k], 1e-8)
        end

        abs(ll - prev_ll) < tol && break
        prev_ll = ll
    end

    # Viterbi decoding
    log_δ = Matrix{Float64}(undef, T_len, K)
    ψ = Matrix{Int}(undef, T_len, K)

    for k in 1:K
        log_δ[1, k] = log(max(π_init[k], 1e-300)) + student_t_logpdf(returns[1], μ[k], σ[k], ν)
    end

    for t in 2:T_len
        for j in 1:K
            vals = [log_δ[t-1, i] + log(max(A[i, j], 1e-300)) for i in 1:K]
            log_δ[t, j] = maximum(vals) + student_t_logpdf(returns[t], μ[j], σ[j], ν)
            ψ[t, j] = argmax(vals)
        end
    end

    path = Vector{Int}(undef, T_len)
    path[T_len] = argmax(log_δ[T_len, :])
    for t in (T_len-1):-1:1
        path[t] = ψ[t+1, path[t+1]]
    end

    ll_final = sum(log.(c) for c in [sum(α[end, :])])
    ll_final = sum(log.(max.(c, 1e-300)) for c in [sum(α[t, :]) for t in 1:T_len])
    n_params = K - 1 + K * (K - 1) + 2 * K
    bic = -2 * prev_ll + n_params * log(T_len)

    return HMMRegimeModel(K, copy(π_init), copy(A), copy(μ), copy(σ), ν,
                          prev_ll, bic, path)
end

"""
    fit_hmm_grid_search(returns; K_range, ν_range) -> HMMRegimeModel

Grid search over number of states K and Student-t ν, selecting by BIC.
Per Alswaidan-Varner: systematic search for optimal regime specification.
"""
function fit_hmm_grid_search(returns::Vector{Float64};
                              K_range::Vector{Int}=[2, 3, 4],
                              ν_range::Vector{Float64}=[3.0, 5.0, 7.0, 10.0])::HMMRegimeModel
    best_model = nothing
    best_bic = Inf

    for K in K_range
        length(returns) < 2 * K && continue
        for ν in ν_range
            try
                model = baum_welch_em(returns, K; ν=ν)
                if model.bic < best_bic
                    best_bic = model.bic
                    best_model = model
                end
            catch
                continue
            end
        end
    end

    if best_model === nothing
        return baum_welch_em(returns, 2; ν=5.0)
    end
    return best_model
end

"""
    classify_regime(returns; window=20) -> Symbol

Simple regime classification from recent returns.
"""
function classify_regime(returns::Vector{Float64}; window::Int=20)::Symbol
    n = min(length(returns), window)
    n < 5 && return :normal
    r = returns[end-n+1:end]
    neg_frac = count(x -> x < 0, r) / n
    return neg_frac > 0.7 ? :stressed : :normal
end

"""
    fit_all_hmm(price_data, tickers; min_obs, K_range) -> Dict

Build HMM for each ticker via grid search (BIC selection).
Returns Dict{ticker => HMMRegimeModel}.
"""
function fit_all_hmm(price_data::Dict{String, DataFrame},
                      tickers::Vector{String};
                      min_obs::Int=60,
                      K_range::Vector{Int}=[2, 3])::Dict{String, Any}
    results = Dict{String, Any}()
    for tk in tickers
        !haskey(price_data, tk) && continue
        df = price_data[tk]
        nrow(df) < min_obs + 1 && continue
        try
            log_ret = diff(log.(df.adj_close))
            model = fit_hmm_grid_search(log_ret; K_range=K_range)
            results[tk] = model
            println("    $tk: K=$(model.K), ν=$(model.ν), BIC=$(round(model.bic, digits=1))")
        catch e
            @warn "HMM build failed for $tk: $e"
        end
    end
    return results
end

"""
    extract_regime_params(hmm::HMMRegimeModel) -> NamedTuple

Extract 2-regime (normal/stressed) parameters from an HMMRegimeModel.
Regime with highest μ → normal; lowest μ → stressed.
"""
function extract_regime_params(hmm::HMMRegimeModel)
    order = sortperm(hmm.μ)
    stressed_idx = order[1]
    normal_idx = order[end]

    path = hmm.viterbi_path
    n_s2n, n_s_total, n_n2s, n_n_total = 0.0, 0.0, 0.0, 0.0
    for i in 2:length(path)
        if path[i-1] == stressed_idx
            n_s_total += 1
            path[i] == normal_idx && (n_s2n += 1)
        elseif path[i-1] == normal_idx
            n_n_total += 1
            path[i] == stressed_idx && (n_n2s += 1)
        end
    end

    return (
        μ_normal      = hmm.μ[normal_idx],
        σ_normal      = max(hmm.σ[normal_idx], 1e-6),
        μ_stressed    = hmm.μ[stressed_idx],
        σ_stressed    = max(hmm.σ[stressed_idx], 1e-6),
        p_to_stressed = clamp(n_n_total > 0 ? n_n2s / n_n_total : 0.05, 0.001, 0.5),
        p_to_normal   = clamp(n_s_total > 0 ? n_s2n / n_s_total : 0.10, 0.001, 0.5),
        K             = hmm.K,
        bic           = hmm.bic,
    )
end

"""
Legacy compatibility wrapper — accepts old NamedTuple format from build_hmm_model.
"""
function extract_regime_params(hmm_result::NamedTuple)
    encoded = hmm_result.encoded
    decode = hmm_result.decode
    ns = length(hmm_result.states)
    mid = div(ns, 2)

    μ_stressed = mean(decode[s].μ for s in 1:mid)
    σ_stressed = sqrt(mean(decode[s].σ^2 for s in 1:mid))
    μ_normal = mean(decode[s].μ for s in (mid+1):ns)
    σ_normal = sqrt(mean(decode[s].σ^2 for s in (mid+1):ns))

    n_s2n, n_s_total, n_n2s, n_n_total = 0.0, 0.0, 0.0, 0.0
    for i in 2:length(encoded)
        prev, curr = encoded[i-1], encoded[i]
        if prev <= mid
            n_s_total += 1
            curr > mid && (n_s2n += 1)
        else
            n_n_total += 1
            curr <= mid && (n_n2s += 1)
        end
    end

    return (
        μ_normal      = μ_normal,
        σ_normal      = max(σ_normal, 1e-6),
        μ_stressed    = μ_stressed,
        σ_stressed    = max(σ_stressed, 1e-6),
        p_to_stressed = clamp(n_n_total > 0 ? n_n2s / n_n_total : 0.05, 0.001, 0.5),
        p_to_normal   = clamp(n_s_total > 0 ? n_s2n / n_s_total : 0.10, 0.001, 0.5),
    )
end

# ── Standard GBM ─────────────────────────────────────────────────────────────

"""
    simulate_gbm(S₀, μ, σ, T; Δt, n_paths) -> Matrix{Float64}

Generate price paths via Geometric Brownian Motion (CHEME-5660 Week 5b).
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

# ── Regime-Switching GBM ──────────────────────────────────────────────────────

"""
    simulate_regime_gbm(...) -> Matrix{Float64}

Two-state Markov regime-switching GBM.
Parameters (μ, σ, transition probs) come from extract_regime_params(build_hmm_model(...)).
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
            stressed = stressed ? (rand() > p_to_normal) : (rand() < p_to_stressed)
            μ = stressed ? μ_stressed : μ_normal
            σ = stressed ? σ_stressed : σ_normal
            drift = (μ - 0.5 * σ^2) * Δt
            diffusion = σ * sqrt(Δt)
            paths[t, p] = paths[t-1, p] * exp(drift + diffusion * randn())
        end
    end
    return paths
end

# ── Correlated Multi-Asset GBM ────────────────────────────────────────────────

"""
    simulate_correlated_gbm(S₀, μ, σ, ρ, T; Δt, n_paths) -> Array{Float64, 3}

Correlated price paths via Cholesky decomposition (CHEME-5660 Week 6).
Returns (n_steps+1, n_assets, n_paths).
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
    catch e
        @warn "Cholesky failed, using independent paths: $e"
        Matrix{Float64}(I, n_assets, n_assets)
    end

    paths = Array{Float64, 3}(undef, n_steps + 1, n_assets, n_paths)
    for j in 1:n_assets
        paths[1, j, :] .= S₀[j]
    end

    for p in 1:n_paths
        for t in 2:(n_steps + 1)
            Z_corr = L * randn(n_assets)
            for j in 1:n_assets
                drift = (μ[j] - 0.5 * σ[j]^2) * Δt
                diffusion = σ[j] * sqrt(Δt) * Z_corr[j]
                paths[t, j, p] = paths[t-1, j, p] * exp(drift + diffusion)
            end
        end
    end
    return paths
end

# ── Earnings Jump Diffusion ───────────────────────────────────────────────────

"""
    simulate_earnings_jump_gbm(...) -> Matrix{Float64}

GBM + Merton-style jump diffusion + post-earnings vol crush.
Per Alswaidan-Varner: Poisson(λ)-triggered jumps with LogNormal magnitude.
Earnings dates get deterministic jumps; between dates, random Poisson jumps
capture non-earnings discontinuities (e.g., macro shocks, analyst downgrades).
"""
function simulate_earnings_jump_gbm(S₀::Float64, μ::Float64, σ::Float64, T::Float64,
                                     earnings_days::Vector{Int};
                                     jump_mean::Float64=0.0, jump_std::Float64=0.07,
                                     vol_crush::Float64=0.60,
                                     λ_jump::Float64=0.05,
                                     Δt::Float64=1.0/252.0, n_paths::Int=1000)::Matrix{Float64}
    n_steps = ceil(Int, T / Δt)
    paths = Matrix{Float64}(undef, n_steps + 1, n_paths)
    paths[1, :] .= S₀

    earnings_set = Set(earnings_days)
    crush_steps = Set{Int}()
    for ed in earnings_days
        for d in (ed+1):min(ed+5, n_steps)
            push!(crush_steps, d)
        end
    end

    compensator = λ_jump * (exp(jump_mean + 0.5 * jump_std^2) - 1.0)

    for t in 2:(n_steps + 1)
        σ_t = (t - 1) in crush_steps ? σ * vol_crush : σ
        drift = (μ - 0.5 * σ_t^2 - compensator) * Δt
        diffusion = σ_t * sqrt(Δt)

        for p in 1:n_paths
            n_jumps = rand(Poisson(λ_jump * Δt))
            jump_component = 0.0
            if (t - 1) in earnings_set
                jump_component = jump_mean + jump_std * randn()
            elseif n_jumps > 0
                for _ in 1:n_jumps
                    jump_component += jump_mean + jump_std * randn()
                end
            end
            paths[t, p] = paths[t-1, p] * exp(drift + diffusion * randn() + jump_component)
        end
    end
    return paths
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

Run GBM under each stress scenario and report distribution statistics.
"""
function run_stress_scenarios(S₀::Float64, μ::Float64, σ::Float64, T::Float64;
                               scenarios::Vector{StressScenario}=DEFAULT_STRESS_SCENARIOS,
                               n_paths::Int=5000, Δt::Float64=1.0/252.0)::DataFrame
    results = DataFrame(
        Scenario=String[], MeanReturn=Float64[], MedianReturn=Float64[],
        VaR95=Float64[], MaxDD=Float64[], PctBelowMinus20=Float64[]
    )

    for sc in scenarios
        μ_eff = μ + sc.μ_override
        σ_eff = σ * sc.σ_multiplier
        paths = simulate_gbm(S₀, μ_eff, σ_eff, T; Δt=Δt, n_paths=n_paths)

        if sc.gap_pct != 0.0 && sc.gap_day > 0 && sc.gap_day < size(paths, 1)
            factor = 1.0 + sc.gap_pct
            for t in sc.gap_day:size(paths, 1), p in 1:n_paths
                paths[t, p] *= factor
            end
        end

        terminal = paths[end, :]
        returns = (terminal .- S₀) ./ S₀

        max_dds = Float64[]
        for p in 1:n_paths
            pk, mdd = -Inf, 0.0
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
            round(mean(returns)*100, digits=2),
            round(median(returns)*100, digits=2),
            round(-sorted_ret[var_idx]*100, digits=2),
            round(mean(max_dds)*100, digits=2),
            round(count(r -> r < -0.20, returns)/n_paths*100, digits=2)
        ))
    end
    return results
end

"""
    mc_summary(paths) -> NamedTuple

Quick summary statistics for a path matrix.
"""
function mc_summary(paths::Matrix{Float64})
    terminal = paths[end, :]
    S₀ = paths[1, 1]
    returns = (terminal .- S₀) ./ S₀
    sorted_ret = sort(returns)
    var_idx = max(1, floor(Int, 0.05 * length(sorted_ret)))
    return (
        mean_return   = mean(returns),
        std_return    = std(returns),
        median_return = median(returns),
        var_95        = -sorted_ret[var_idx],
        min_return    = minimum(returns),
        max_return    = maximum(returns),
    )
end

# ── Portfolio-Level Stress ────────────────────────────────────────────────────

"""
    apply_stress_to_prices(price_data; vol_mult, drift_adj, gap_pct, gap_day,
                           spread_widening, liquidity_thin_pct) -> Dict

Enhanced stress transformation with microstructure effects.
Per Varner PDF Section 7B.
"""
function apply_stress_to_prices(price_data::Dict{String, DataFrame};
                                  vol_mult::Float64=1.0, drift_adj::Float64=0.0,
                                  gap_pct::Float64=0.0, gap_day::Int=0,
                                  spread_widening::Float64=1.0,
                                  liquidity_thin_pct::Float64=0.0)::Dict{String, DataFrame}
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
        if hasproperty(sdf, :volume) && liquidity_thin_pct > 0.0
            sdf.volume = round.(Int, sdf.volume .* (1.0 - liquidity_thin_pct))
        end
        stressed[ticker] = sdf
    end
    return stressed
end

const EXTENDED_STRESS_SCENARIOS = [
    (label="Normal (baseline)",              vol_mult=1.0, drift_adj=0.0,  gap_pct=0.0,  gap_day=0,  spread_w=1.0, liq_thin=0.0),
    (label="Vol Spike (2x vol)",             vol_mult=2.0, drift_adj=0.0,  gap_pct=0.0,  gap_day=0,  spread_w=1.5, liq_thin=0.0),
    (label="Bear Market (-20% drift)",       vol_mult=1.5, drift_adj=-0.20,gap_pct=0.0,  gap_day=0,  spread_w=1.2, liq_thin=0.0),
    (label="Flash Crash (-10% gap d30)",     vol_mult=2.0, drift_adj=0.0,  gap_pct=-0.10,gap_day=30, spread_w=2.0, liq_thin=0.3),
    (label="Name Blowup (-30% gap d60)",     vol_mult=1.5, drift_adj=0.0,  gap_pct=-0.30,gap_day=60, spread_w=2.5, liq_thin=0.4),
    (label="Bull Squeeze (+15% gap d45)",    vol_mult=1.5, drift_adj=0.30, gap_pct=0.15, gap_day=45, spread_w=1.3, liq_thin=0.1),
    (label="Vol Crush (post-earnings)",      vol_mult=0.5, drift_adj=0.0,  gap_pct=0.0,  gap_day=0,  spread_w=0.8, liq_thin=0.0),
    (label="Liquidity Thinning (70% vol)",   vol_mult=1.2, drift_adj=-0.05,gap_pct=0.0,  gap_day=0,  spread_w=3.0, liq_thin=0.7),
    (label="Spread Widening (close micro.)", vol_mult=1.0, drift_adj=0.0,  gap_pct=0.0,  gap_day=0,  spread_w=4.0, liq_thin=0.5),
]

# ── Heston-Aware Regime GBM ──────────────────────────────────────────────────

"""
    simulate_heston_regime_gbm(S₀, hmm, heston_params, T; Δt, n_paths)

Regime-switching GBM where volatility comes from:
  1. HMM regime state (bull/bear/sideways) → selects σ regime
  2. Heston stochastic vol provides the per-step variance adjustment

This combines the HMM regime detection with the Heston vol dynamics.
"""
function simulate_heston_regime_gbm(S₀::Float64,
                                      regime_params::NamedTuple,
                                      heston_params::HestonParams,
                                      T::Float64;
                                      Δt::Float64=1.0/252.0,
                                      n_paths::Int=1000)::Matrix{Float64}
    n_steps = ceil(Int, T / Δt)
    paths = Matrix{Float64}(undef, n_steps + 1, n_paths)
    paths[1, :] .= S₀

    κ, θ, ξ, ρ = heston_params.kappa, heston_params.theta, heston_params.xi, heston_params.rho
    p_ts = regime_params.p_to_stressed
    p_tn = regime_params.p_to_normal

    for p in 1:n_paths
        stressed = false
        v = heston_params.v0

        for t in 2:(n_steps + 1)
            stressed = stressed ? (rand() > p_tn) : (rand() < p_ts)
            μ = stressed ? regime_params.μ_stressed : regime_params.μ_normal
            σ_regime = stressed ? regime_params.σ_stressed : regime_params.σ_normal

            Z1, Z2 = randn(), randn()
            Zv = ρ * Z1 + sqrt(1 - ρ^2) * Z2

            v_next = v + κ * (θ - v) * Δt + ξ * sqrt(max(v, 0.0) * Δt) * Zv
            v = max(v_next, 1e-8)

            σ_effective = 0.5 * σ_regime + 0.5 * sqrt(v)

            drift = (μ - 0.5 * σ_effective^2) * Δt
            diffusion = σ_effective * sqrt(Δt)
            paths[t, p] = paths[t-1, p] * exp(drift + diffusion * Z1)
        end
    end
    return paths
end
