"""
Simulation — Jump-HMM Regime Detection + GBM Path Simulation

Contains:
  1. Jump-HMM regime detection (from JumpHMM.jl by Varner):
     - Laplace partition: MLE fit → N equal-probability quantile bins
     - Frequency-based transition matrix estimation
     - Per-state Student-t(ν) emissions
     - Viterbi decoding (log-space)
     - Forward filter & log-likelihood (log-sum-exp)
     - Poisson jump mechanism (tune via grid search)
  2. GBM path simulation (CHEME-5660 Week 5b)
  3. Regime-switching GBM, correlated multi-asset GBM (Cholesky, Week 6)
  4. Earnings jump diffusion with Poisson intensity (Merton 1976)
  5. Stress scenarios

GBM formula:  S_{t+Δt} = S_t · exp[(μ - σ²/2)·Δt + σ·√Δt · Z],  Z ~ N(0,1)
"""

# ── Jump-HMM Types (from JumpHMM.jl) ────────────────────────────────────────

struct LaplacePartition
    mu::Float64
    b::Float64
    N::Int
    boundaries::Vector{Float64}

    function LaplacePartition(mu::Float64, b::Float64, N::Int)
        N >= 1 || error("N must be >= 1")
        b > 0.0 || error("b must be positive")
        cutoffs = range(0.0, 1.0, length=N + 1) |> collect
        d = Laplace(mu, b)
        boundaries = [quantile(d, p) for p in cutoffs]
        boundaries[1] = -Inf
        boundaries[end] = Inf
        return new(mu, b, N, boundaries)
    end
end

struct StudentTEmission
    mu::Float64
    sigma::Float64
    nu::Float64
    n_obs::Int
    is_fallback::Bool
end

struct JumpParameters
    epsilon::Float64
    lambda::Float64
    p_neg::Float64
    N_tail::Int

    function JumpParameters(epsilon::Float64, lambda::Float64;
                            p_neg::Float64=0.52, N_tail::Int=5)
        0.0 <= epsilon <= 1.0 || error("epsilon must be in [0, 1]")
        lambda > 0.0 || error("lambda must be positive")
        return new(epsilon, lambda, p_neg, N_tail)
    end
end

struct JumpHMM
    partition::LaplacePartition
    transition::Matrix{Float64}
    emissions::Vector{StudentTEmission}
    stationary::Vector{Float64}
    jump::JumpParameters
    nu::Float64
    rf::Float64
    dt::Float64
end

struct SimulationPath
    states::Vector{Int}
    observations::Vector{Float64}
    jumps::Vector{Bool}
end

struct SimulationResult
    paths::Vector{SimulationPath}
end

# ── Finance Helpers (from JumpHMM.jl) ────────────────────────────────────────

function excess_growth_rates(prices::AbstractVector{<:Real};
                             rf::Float64=0.0, dt::Float64=1/252)
    n = length(prices)
    G = Vector{Float64}(undef, n - 1)
    for i in 1:(n-1)
        G[i] = (1.0 / dt) * log(prices[i+1] / prices[i]) - rf
    end
    return G
end

function prices_from_growth_rates(G::AbstractVector{Float64}, P0::Float64;
                                  rf::Float64=0.0, dt::Float64=1/252)
    n = length(G)
    P = Vector{Float64}(undef, n + 1)
    P[1] = P0
    for i in 1:n
        P[i+1] = P[i] * exp((G[i] + rf) * dt)
    end
    return P
end

# ── Partition (from JumpHMM.jl) ──────────────────────────────────────────────

function fit_laplace_partition(observations::AbstractVector{Float64}; N::Int=100)
    length(observations) < 2 && error("Need at least 2 observations to fit partition")
    all(x -> x == observations[1], observations) && error("Cannot fit partition: constant series")
    d = fit_mle(Laplace, observations)
    return LaplacePartition(params(d)..., N)
end

function assign_states(partition::LaplacePartition,
                       observations::AbstractVector{Float64})
    boundaries = partition.boundaries
    N = partition.N
    states = Vector{Int}(undef, length(observations))
    for i in eachindex(observations)
        k = searchsortedlast(boundaries, observations[i])
        states[i] = clamp(k, 1, N)
    end
    return states
end

# ── Transition Matrix (from JumpHMM.jl) ─────────────────────────────────────

function estimate_transition(states::AbstractVector{Int}, N::Int)
    P = zeros(Float64, N, N)
    for i in 2:length(states)
        P[states[i-1], states[i]] += 1.0
    end
    for row in 1:N
        Z = sum(P[row, :])
        if Z > 0.0
            P[row, :] ./= Z
        else
            P[row, :] .= 1.0 / N
        end
    end
    return P
end

function stationary_distribution(T::Matrix{Float64})
    N = size(T, 1)
    A = transpose(T) - I
    A[N, :] .= 1.0
    b = zeros(N)
    b[N] = 1.0
    pi_bar = A \ b
    pi_bar .= max.(pi_bar, 0.0)
    pi_bar ./= sum(pi_bar)
    return pi_bar
end

# ── Emission (from JumpHMM.jl) ──────────────────────────────────────────────

function fit_emissions(states::AbstractVector{Int},
                       observations::AbstractVector{Float64}, N::Int;
                       nu::Float64=5.0, min_obs::Int=2)
    mu_global = mean(observations)
    sigma_global = std(observations)
    emissions = Vector{StudentTEmission}(undef, N)

    for k in 1:N
        idxs = findall(==(k), states)
        n_obs = length(idxs)
        if n_obs >= min_obs
            mu_k = mean(observations[idxs])
            sigma_k = std(observations[idxs])
            if sigma_k < 1e-12
                emissions[k] = StudentTEmission(mu_k, sigma_global, nu, n_obs, true)
            else
                emissions[k] = StudentTEmission(mu_k, sigma_k, nu, n_obs, false)
            end
        else
            emissions[k] = StudentTEmission(mu_global, sigma_global, nu, n_obs, true)
        end
    end
    return emissions
end

function sample_emission(e::StudentTEmission)
    return e.mu + e.sigma * rand(TDist(e.nu))
end

# ── Decode / Forward / Log-Likelihood (from JumpHMM.jl) ─────────────────────

function viterbi_decode(model::JumpHMM, observations::AbstractVector{Float64})
    isempty(observations) && error("observations must be non-empty")
    N = model.partition.N
    T_steps = length(observations)
    nu = model.nu

    log_T = log.(model.transition)
    dists = [model.emissions[k].mu + model.emissions[k].sigma * TDist(nu)
             for k in 1:N]

    delta = Matrix{Float64}(undef, T_steps, N)
    psi = Matrix{Int}(undef, T_steps, N)

    for k in 1:N
        lp = model.stationary[k] > 0.0 ? log(model.stationary[k]) : -Inf
        delta[1, k] = lp + logpdf(dists[k], observations[1])
        psi[1, k] = 0
    end

    for t in 2:T_steps
        for k in 1:N
            best_val = -Inf
            best_j = 1
            for j in 1:N
                val = delta[t-1, j] + log_T[j, k]
                if val > best_val
                    best_val = val
                    best_j = j
                end
            end
            delta[t, k] = best_val + logpdf(dists[k], observations[t])
            psi[t, k] = best_j
        end
    end

    states = Vector{Int}(undef, T_steps)
    states[T_steps] = argmax(delta[T_steps, :])
    for t in (T_steps-1):-1:1
        states[t] = psi[t+1, states[t+1]]
    end
    return states
end

function forward_filter(model::JumpHMM, observations::AbstractVector{Float64})
    isempty(observations) && error("observations must be non-empty")
    N = model.partition.N
    T_steps = length(observations)
    nu = model.nu

    dists = [model.emissions[k].mu + model.emissions[k].sigma * TDist(nu)
             for k in 1:N]
    log_T = log.(model.transition)

    alpha = Matrix{Float64}(undef, T_steps, N)
    log_alpha = Vector{Float64}(undef, N)

    for k in 1:N
        lp = model.stationary[k] > 0.0 ? log(model.stationary[k]) : -Inf
        log_alpha[k] = lp + logpdf(dists[k], observations[1])
    end
    m = maximum(log_alpha)
    if isfinite(m)
        alpha[1, :] .= exp.(log_alpha .- m) ./ sum(exp.(log_alpha .- m))
    else
        alpha[1, :] .= 1.0 / N
        log_alpha .= -log(N)
    end

    log_alpha_new = Vector{Float64}(undef, N)
    for t in 2:T_steps
        for k in 1:N
            max_val = -Inf
            for j in 1:N
                v = log_alpha[j] + log_T[j, k]
                v > max_val && (max_val = v)
            end
            s = 0.0
            for j in 1:N
                s += exp(log_alpha[j] + log_T[j, k] - max_val)
            end
            log_alpha_new[k] = max_val + log(s) + logpdf(dists[k], observations[t])
        end
        m = maximum(log_alpha_new)
        if isfinite(m)
            log_alpha .= log_alpha_new
            alpha[t, :] .= exp.(log_alpha_new .- m) ./ sum(exp.(log_alpha_new .- m))
        else
            alpha[t, :] .= 1.0 / N
            log_alpha .= -log(N)
        end
    end
    return alpha
end

function hmm_log_likelihood(model::JumpHMM, observations::AbstractVector{Float64})
    isempty(observations) && error("observations must be non-empty")
    N = model.partition.N
    T_steps = length(observations)
    nu = model.nu

    dists = [model.emissions[k].mu + model.emissions[k].sigma * TDist(nu)
             for k in 1:N]
    log_T = log.(model.transition)

    log_alpha = Vector{Float64}(undef, N)
    ll = 0.0

    for k in 1:N
        lp = model.stationary[k] > 0.0 ? log(model.stationary[k]) : -Inf
        log_alpha[k] = lp + logpdf(dists[k], observations[1])
    end
    m = maximum(log_alpha)
    if isfinite(m)
        log_Z = m + log(sum(exp.(log_alpha .- m)))
        ll += log_Z
        log_alpha .-= log_Z
    else
        return -Inf
    end

    log_alpha_new = Vector{Float64}(undef, N)
    for t in 2:T_steps
        for k in 1:N
            max_val = -Inf
            for j in 1:N
                v = log_alpha[j] + log_T[j, k]
                v > max_val && (max_val = v)
            end
            s = 0.0
            for j in 1:N
                s += exp(log_alpha[j] + log_T[j, k] - max_val)
            end
            log_alpha_new[k] = max_val + log(s) + logpdf(dists[k], observations[t])
        end
        m = maximum(log_alpha_new)
        if isfinite(m)
            log_Z = m + log(sum(exp.(log_alpha_new .- m)))
            ll += log_Z
            log_alpha .= log_alpha_new .- log_Z
        else
            return -Inf
        end
    end
    return ll
end

# ── Simulate (from JumpHMM.jl) ──────────────────────────────────────────────

function simulate_jumphmm(model::JumpHMM, n_steps::Int;
                           n_paths::Int=1000, start::Union{Int,Symbol}=:stationary)
    n_steps >= 1 || error("n_steps must be >= 1")
    N = model.partition.N
    eps = model.jump.epsilon
    lam = model.jump.lambda
    p_neg = model.jump.p_neg
    N_tail = model.jump.N_tail
    jump_dist = Poisson(lam)

    row_cats = [Categorical(model.transition[s, :]) for s in 1:N]
    bottom_states = collect(1:min(N_tail, N))
    top_states = collect(max(1, N - N_tail + 1):N)
    pi_cat = Categorical(model.stationary)

    paths = Vector{SimulationPath}(undef, n_paths)
    for p in 1:n_paths
        states_out = Vector{Int}(undef, n_steps)
        obs_out = Vector{Float64}(undef, n_steps)
        jumps_out = Vector{Bool}(undef, n_steps)

        s = (start === :stationary) ? rand(pi_cat) : start
        states_out[1] = s
        obs_out[1] = sample_emission(model.emissions[s])
        jumps_out[1] = false

        t = 2
        while t <= n_steps
            if eps > 0.0 && rand() < eps
                K = rand(jump_dist)
                if K == 0
                    s = rand(row_cats[s])
                    states_out[t] = s
                    obs_out[t] = sample_emission(model.emissions[s])
                    jumps_out[t] = false
                    t += 1
                else
                    for _ in 1:K
                        t > n_steps && break
                        s = rand() < p_neg ? rand(bottom_states) : rand(top_states)
                        states_out[t] = s
                        obs_out[t] = sample_emission(model.emissions[s])
                        jumps_out[t] = true
                        t += 1
                    end
                end
            else
                s = rand(row_cats[s])
                states_out[t] = s
                obs_out[t] = sample_emission(model.emissions[s])
                jumps_out[t] = false
                t += 1
            end
        end
        paths[p] = SimulationPath(states_out, obs_out, jumps_out)
    end
    return SimulationResult(paths)
end

# ── Fit Orchestrator (from JumpHMM.jl) ──────────────────────────────────────

function fit_jumphmm(prices::AbstractVector{<:Real};
                     rf::Float64=0.0, N::Int=100, nu::Float64=5.0,
                     dt::Float64=1/252, min_obs::Int=2)
    G = excess_growth_rates(prices; rf=rf, dt=dt)
    partition = fit_laplace_partition(G; N=N)
    states = assign_states(partition, G)
    T = estimate_transition(states, N)
    emissions = fit_emissions(states, G, N; nu=nu, min_obs=min_obs)
    pi_bar = stationary_distribution(T)
    jump = JumpParameters(0.0, 1.0)

    return JumpHMM(partition, T, emissions, pi_bar, jump, nu, rf, dt)
end

# ── Tune (from JumpHMM.jl) ──────────────────────────────────────────────────

function tune_jumphmm(model::JumpHMM, prices::AbstractVector{<:Real};
                      epsilon_range=range(1e-4, 2.5e-2, length=20),
                      lambda_range=range(10.0, 160.0, length=16),
                      n_paths::Int=200, n_steps::Int=0,
                      w_kappa::Float64=0.20, p_neg::Float64=0.52,
                      N_tail::Int=5, acf_lags::Int=25)
    G_emp = excess_growth_rates(prices; rf=model.rf, dt=model.dt)
    if n_steps == 0
        n_steps = length(G_emp)
    end
    acf_lags = min(acf_lags, min(length(G_emp), n_steps) - 1)
    lags = collect(1:acf_lags)
    acf_obs = autocor(abs.(G_emp), lags)
    kappa_obs = kurtosis(G_emp)

    best_J = Inf
    best_eps = first(epsilon_range)
    best_lam = first(lambda_range)

    for eps_cand in epsilon_range
        for lam_cand in lambda_range
            jump_cand = JumpParameters(Float64(eps_cand), Float64(lam_cand);
                                       p_neg=p_neg, N_tail=N_tail)
            candidate = JumpHMM(model.partition, model.transition, model.emissions,
                                model.stationary, jump_cand, model.nu, model.rf, model.dt)
            result = simulate_jumphmm(candidate, n_steps; n_paths=n_paths)

            J_accum = 0.0
            n_valid = 0
            for path in result.paths
                any(path.jumps) || continue
                acf_sim = autocor(abs.(path.observations), lags)
                kappa_sim = kurtosis(path.observations)
                acf_err = sum((acf_obs .- acf_sim).^2)
                kappa_err = w_kappa * (kappa_obs - kappa_sim)^2
                J_accum += acf_err + kappa_err
                n_valid += 1
            end

            if n_valid > 0
                J_mean = J_accum / n_valid
                if J_mean < best_J
                    best_J = J_mean
                    best_eps = Float64(eps_cand)
                    best_lam = Float64(lam_cand)
                end
            end
        end
    end

    if !isfinite(best_J)
        @warn "tune: no candidate produced paths with jumps. Consider widening ranges."
    end

    best_jump = JumpParameters(best_eps, best_lam; p_neg=p_neg, N_tail=N_tail)
    return JumpHMM(model.partition, model.transition, model.emissions,
                   model.stationary, best_jump, model.nu, model.rf, model.dt)
end

# ── Fit All Tickers (project integration) ───────────────────────────────────

function fit_all_hmm(price_data::Dict{String, DataFrame},
                      tickers::Vector{String};
                      min_obs::Int=60, N::Int=100,
                      nu::Float64=5.0,
                      tune_jumps::Bool=true)::Dict{String, JumpHMM}
    results = Dict{String, JumpHMM}()
    for tk in tickers
        !haskey(price_data, tk) && continue
        df = price_data[tk]
        nrow(df) < min_obs + 1 && continue
        try
            prices = Float64.(df.adj_close)
            model = fit_jumphmm(prices; N=N, nu=nu)
            if tune_jumps
                model = tune_jumphmm(model, prices)
            end
            results[tk] = model
            ll = hmm_log_likelihood(model, excess_growth_rates(prices))
            jp = model.jump
            println("    $tk: N=$(model.partition.N), ε=$(round(jp.epsilon, digits=4)), λ=$(round(jp.lambda, digits=1)), LL=$(round(ll, digits=1))")
        catch e
            @warn "JumpHMM fit failed for $tk: $e"
        end
    end
    return results
end

"""
    extract_regime_params(model::JumpHMM) -> NamedTuple

Extract 2-regime (normal/stressed) summary from a JumpHMM.
States are partitioned by median: lower half → stressed, upper half → normal.
Transition probabilities computed from Viterbi-decoded state path.
"""
function extract_regime_params(model::JumpHMM; prices::Union{AbstractVector{<:Real},Nothing}=nothing)
    N = model.partition.N
    mid = div(N, 2)

    mu_s = mean(model.emissions[k].mu for k in 1:mid)
    sig_s = sqrt(mean(model.emissions[k].sigma^2 for k in 1:mid))
    mu_n = mean(model.emissions[k].mu for k in (mid+1):N)
    sig_n = sqrt(mean(model.emissions[k].sigma^2 for k in (mid+1):N))

    n_s2n, n_s_total, n_n2s, n_n_total = 0.0, 0.0, 0.0, 0.0

    if prices !== nothing
        G = excess_growth_rates(prices; rf=model.rf, dt=model.dt)
        path = viterbi_decode(model, G)
        for i in 2:length(path)
            prev, curr = path[i-1], path[i]
            if prev <= mid
                n_s_total += 1
                curr > mid && (n_s2n += 1)
            else
                n_n_total += 1
                curr <= mid && (n_n2s += 1)
            end
        end
    end

    return (
        mu_normal      = mu_n,
        sig_normal     = max(sig_n, 1e-6),
        mu_stressed    = mu_s,
        sig_stressed   = max(sig_s, 1e-6),
        p_to_stressed  = clamp(n_n_total > 0 ? n_n2s / n_n_total : 0.05, 0.001, 0.5),
        p_to_normal    = clamp(n_s_total > 0 ? n_s2n / n_s_total : 0.10, 0.001, 0.5),
        N              = N,
        ll             = prices !== nothing ?
                           hmm_log_likelihood(model, excess_growth_rates(prices; rf=model.rf, dt=model.dt)) :
                           NaN,
    )
end

# ── Standard GBM ─────────────────────────────────────────────────────────────

"""
    simulate_gbm(S₀, μ, σ, T; Δt, n_paths) -> Matrix{Float64}

Generate price paths via Geometric Brownian Motion (CHEME-5660 Week 5b).
"""
function simulate_gbm(S0::Float64, mu::Float64, sig::Float64, T::Float64;
                       Δt::Float64=1.0/252.0, n_paths::Int=1000)::Matrix{Float64}
    n_steps = ceil(Int, T / Δt)
    paths = Matrix{Float64}(undef, n_steps + 1, n_paths)
    paths[1, :] .= S0

    drift = (mu - 0.5 * sig^2) * Δt
    diffusion = sig * sqrt(Δt)

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
function simulate_regime_gbm(S0::Float64,
                              mu_normal::Float64, sig_normal::Float64,
                              mu_stressed::Float64, sig_stressed::Float64,
                              p_to_stressed::Float64, p_to_normal::Float64,
                              T::Float64;
                              Δt::Float64=1.0/252.0, n_paths::Int=1000)::Matrix{Float64}
    n_steps = ceil(Int, T / Δt)
    paths = Matrix{Float64}(undef, n_steps + 1, n_paths)
    paths[1, :] .= S0

    for p in 1:n_paths
        stressed = false
        for t in 2:(n_steps + 1)
            stressed = stressed ? (rand() > p_to_normal) : (rand() < p_to_stressed)
            mu_t = stressed ? mu_stressed : mu_normal
            sig_t = stressed ? sig_stressed : sig_normal
            drift = (mu_t - 0.5 * sig_t^2) * Δt
            diffusion = sig_t * sqrt(Δt)
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
function simulate_correlated_gbm(S0_vec::Vector{Float64}, mu_vec::Vector{Float64},
                                   sig_vec::Vector{Float64}, rho_mat::Matrix{Float64},
                                   T::Float64;
                                   Δt::Float64=1.0/252.0, n_paths::Int=1000)::Array{Float64, 3}
    n_assets = length(S0_vec)
    n_steps = ceil(Int, T / Δt)

    rho_pd = rho_mat + 1e-8 * I
    L = try
        cholesky(Hermitian(rho_pd)).L
    catch e
        @warn "Cholesky failed, using independent paths: $e"
        Matrix{Float64}(I, n_assets, n_assets)
    end

    paths = Array{Float64, 3}(undef, n_steps + 1, n_assets, n_paths)
    for j in 1:n_assets
        paths[1, j, :] .= S0_vec[j]
    end

    for p in 1:n_paths
        for t in 2:(n_steps + 1)
            Z_corr = L * randn(n_assets)
            for j in 1:n_assets
                drift = (mu_vec[j] - 0.5 * sig_vec[j]^2) * Δt
                diffusion = sig_vec[j] * sqrt(Δt) * Z_corr[j]
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
function simulate_earnings_jump_gbm(S0::Float64, mu::Float64, sig::Float64, T::Float64,
                                     earnings_days::Vector{Int};
                                     jump_mean::Float64=0.0, jump_std::Float64=0.07,
                                     vol_crush::Float64=0.60,
                                     lam_jump::Float64=0.05,
                                     Δt::Float64=1.0/252.0, n_paths::Int=1000)::Matrix{Float64}
    n_steps = ceil(Int, T / Δt)
    paths = Matrix{Float64}(undef, n_steps + 1, n_paths)
    paths[1, :] .= S0

    earnings_set = Set(earnings_days)
    crush_steps = Set{Int}()
    for ed in earnings_days
        for d in (ed+1):min(ed+5, n_steps)
            push!(crush_steps, d)
        end
    end

    compensator = lam_jump * (exp(jump_mean + 0.5 * jump_std^2) - 1.0)

    for t in 2:(n_steps + 1)
        sig_t = (t - 1) in crush_steps ? sig * vol_crush : sig
        drift = (mu - 0.5 * sig_t^2 - compensator) * Δt
        diffusion = sig_t * sqrt(Δt)

        for p in 1:n_paths
            n_jumps = rand(Poisson(lam_jump * Δt))
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
    mu_override::Float64
    sig_multiplier::Float64
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
function run_stress_scenarios(S0::Float64, mu::Float64, sig::Float64, T::Float64;
                               scenarios::Vector{StressScenario}=DEFAULT_STRESS_SCENARIOS,
                               n_paths::Int=5000, Δt::Float64=1.0/252.0)::DataFrame
    results = DataFrame(
        Scenario=String[], MeanReturn=Float64[], MedianReturn=Float64[],
        VaR95=Float64[], MaxDD=Float64[], PctBelowMinus20=Float64[]
    )

    for sc in scenarios
        mu_eff = mu + sc.mu_override
        sig_eff = sig * sc.sig_multiplier
        paths = simulate_gbm(S0, mu_eff, sig_eff, T; Δt=Δt, n_paths=n_paths)

        if sc.gap_pct != 0.0 && sc.gap_day > 0 && sc.gap_day < size(paths, 1)
            factor = 1.0 + sc.gap_pct
            for t in sc.gap_day:size(paths, 1), p in 1:n_paths
                paths[t, p] *= factor
            end
        end

        terminal = paths[end, :]
        returns = (terminal .- S0) ./ S0

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
    s0 = paths[1, 1]
    returns = (terminal .- s0) ./ s0
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
        if spread_widening > 1.0
            for i in 1:length(prices)
                noise = (rand() - 0.5) * 2.0 * (spread_widening - 1.0) * 0.001
                prices[i] *= (1.0 + noise)
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
    simulate_heston_regime_gbm(S₀, regime_params, heston_params, T; Δt, n_paths)

Regime-switching GBM where volatility comes from:
  1. HMM regime state (bull/bear/sideways) → selects σ regime
  2. Modified Heston variance process provides per-step variance adjustment

Uses the modified Heston model:
    dv = κ(θ_base - v)dt + σ_v √v dW_v
with reflecting boundary, and v₀ = θ_base (start at equilibrium).
"""
function simulate_heston_regime_gbm(S0::Float64,
                                      regime_params::NamedTuple,
                                      heston_params::HestonParams,
                                      T::Float64;
                                      Δt::Float64=1.0/252.0,
                                      n_paths::Int=1000)::Matrix{Float64}
    n_steps = ceil(Int, T / Δt)
    paths = Matrix{Float64}(undef, n_steps + 1, n_paths)
    paths[1, :] .= S0

    κ = heston_params.κ
    θ_target = heston_params.θ_base
    σ_v = heston_params.σ_v
    p_ts = regime_params.p_to_stressed
    p_tn = regime_params.p_to_normal

    for p in 1:n_paths
        stressed = false
        v = θ_target

        for t in 2:(n_steps + 1)
            stressed = stressed ? (rand() > p_tn) : (rand() < p_ts)
            mu_t = stressed ? regime_params.mu_stressed : regime_params.mu_normal
            sig_regime = stressed ? regime_params.sig_stressed : regime_params.sig_normal

            Zv = randn()
            v_next = v + κ * (θ_target - v) * Δt + σ_v * sqrt(max(v, 0.0) * Δt) * Zv
            v = max(v_next, 1e-8)

            sig_eff = 0.5 * sig_regime + 0.5 * sqrt(v)

            drift = (mu_t - 0.5 * sig_eff^2) * Δt
            diffusion = sig_eff * sqrt(Δt)
            paths[t, p] = paths[t-1, p] * exp(drift + diffusion * randn())
        end
    end
    return paths
end
