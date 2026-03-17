"""
Simulation — Varner PDF Section 7B (Testing Plan B)

Contains:
  1. HMM regime detection (CHEME-5660 Week 13) — uses MyHiddenMarkovModel
  2. GBM path simulation (CHEME-5660 Week 5b)
  3. Regime-switching GBM, correlated multi-asset GBM (Cholesky, Week 6)
  4. Earnings jump diffusion (Merton 1976)
  5. Stress scenarios

HMM approach follows L13a-Example-HMM-SPY-IS-Fall-2025.ipynb:
  Discretize returns via CDF quantiles → count transitions → normalize → P̂.
  Uses MyHiddenMarkovModel from VLQuantitativeFinancePackage.

GBM formula:  S_{t+Δt} = S_t · exp[(μ - σ²/2)·Δt + σ·√Δt · Z],  Z ~ N(0,1)
  Full path simulation (not just endpoint) for drawdown/stress analysis.
  Course uses MyGeometricBrownianMotionEquityModel + sample_endpoint() for
  terminal-only sampling; we extend to full paths here.
"""

# ── HMM Regime Detection (CHEME-5660 Week 13) ────────────────────────────────
# Follows L13a-Example-HMM-SPY-IS-Fall-2025.ipynb exactly:
#   1. Fit Laplace distribution to returns
#   2. CDF quantile boundaries → encode returns as discrete states
#   3. Count transitions → normalize → probability matrix P̂
#   4. build(MyHiddenMarkovModel, ...) from VLQuantitativeFinancePackage
#   5. Per-state Normal decode distributions via fit_mle

"""
    build_hmm_model(returns; number_of_states=100) -> NamedTuple

Build an HMM from return data (CHEME-5660 Week 13 approach).
Returns: (model::MyHiddenMarkovModel, decode, encoded, bounds, P̂, states)
"""
function build_hmm_model(returns::Vector{Float64}; number_of_states::Int=100)
    length(returns) < 20 && error("Need at least 20 observations for HMM")

    states = collect(1:number_of_states)
    E = diagm(ones(number_of_states))

    d = fit_mle(Laplace, returns)
    pct = range(0.0, stop=1.0, length=number_of_states + 1) |> collect
    bounds = zeros(number_of_states, 2)
    for s in states
        bounds[s, 1] = quantile(d, max(pct[s], 1e-10))
        bounds[s, 2] = quantile(d, min(pct[s+1], 1.0 - 1e-10))
    end

    encoded = Int[]
    for val in returns
        ci = 1
        for s in states
            if bounds[s, 1] <= val < bounds[s, 2]
                ci = s; break
            end
        end
        push!(encoded, ci)
    end

    P = zeros(number_of_states, number_of_states)
    for i in 2:length(encoded)
        P[encoded[i-1], encoded[i]] += 1.0
    end

    P̂ = zeros(number_of_states, number_of_states)
    for row in states
        Z = sum(P[row, :])
        Z > 0 && (P̂[row, :] .= P[row, :] ./ Z)
    end

    decode = Dict{Int, Normal}()
    for s in states
        indices = findall(x -> x == s, encoded)
        if length(indices) >= 2
            decode[s] = fit_mle(Normal, returns[indices])
        else
            decode[s] = Normal(mean(returns), std(returns))
        end
    end

    model = build(MyHiddenMarkovModel, (
        states = states,
        T = P̂,
        E = E
    ))

    return (model=model, decode=decode, encoded=encoded, bounds=bounds, P̂=P̂, states=states)
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
    fit_all_hmm(price_data, tickers; min_obs=60, number_of_states=100) -> Dict

Build HMM for each ticker. Results can feed into extract_regime_params()
and then into simulate_regime_gbm().
"""
function fit_all_hmm(price_data::Dict{String, DataFrame},
                      tickers::Vector{String};
                      min_obs::Int=60, number_of_states::Int=100)::Dict{String, Any}
    results = Dict{String, Any}()
    for tk in tickers
        !haskey(price_data, tk) && continue
        df = price_data[tk]
        nrow(df) < min_obs + 1 && continue
        try
            results[tk] = build_hmm_model(diff(log.(df.adj_close));
                                           number_of_states=number_of_states)
        catch e
            @warn "HMM build failed for $tk: $e"
        end
    end
    return results
end

"""
    extract_regime_params(hmm_result) -> NamedTuple

Extract 2-regime (normal/stressed) parameters from an N-state HMM result.
Lower-half states → stressed; upper-half → normal.
Output can be passed directly to simulate_regime_gbm().
"""
function extract_regime_params(hmm_result)
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

GBM + Merton-style jump at earnings dates + post-earnings vol crush.
Per Varner PDF Section 7B: "earnings jumps, vol crush/expansion."
"""
function simulate_earnings_jump_gbm(S₀::Float64, μ::Float64, σ::Float64, T::Float64,
                                     earnings_days::Vector{Int};
                                     jump_mean::Float64=0.0, jump_std::Float64=0.07,
                                     vol_crush::Float64=0.60,
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

    for t in 2:(n_steps + 1)
        σ_t = (t - 1) in crush_steps ? σ * vol_crush : σ
        drift = (μ - 0.5 * σ_t^2) * Δt
        diffusion = σ_t * sqrt(Δt)

        for p in 1:n_paths
            jump = (t - 1) in earnings_set ? jump_mean + jump_std * randn() : 0.0
            paths[t, p] = paths[t-1, p] * exp(drift + diffusion * randn() + jump)
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

# ── ROBUST SIMULATION (PDF Section 7B) ───────────────────────────────────────
# TODO: Stochastic volatility model calibrated to IV surface
# TODO: Overnight gap modeling
# TODO: Full IV surface calibration per-name
# TODO: Cross-asset contagion scenarios
# Placeholder for future implementation — use WRDS IV data from IVData.jl
# to calibrate per-name stochastic vol models here.
