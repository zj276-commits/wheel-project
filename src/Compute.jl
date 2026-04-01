"""
Quantitative computations: log growth matrix, rolling volatility, dividend yield,
and return correlation.

log_growth_matrix: excess log return matrix for cross-sectional analysis.
  Reference: CHEME-5660 Week 5b — SAGBM parameter estimation.
  VLQuantitativeFinancePackage exports a similar function; ours adapts the
  column names to our Yahoo Finance data format (keycol=:adj_close).

compute_rolling_volatility: time-varying σ for each ticker.

trailing_dividend_yield: trailing 12-month annualized dividend yield (no look-ahead).

compute_return_correlation: pairwise correlation matrix for correlated MC.
  Reference: CHEME-5660 Week 6 — Multiple-Asset GBM with Cholesky decomposition.
"""

"""
    log_growth_matrix(dataset, firms; Δt, risk_free_rate, testfirm, keycol) -> Matrix

Compute the excess log growth matrix for a set of firms (CHEME-5660 Week 5b):
  μ_{t,t-1}(r_f) = (1/Δt) · ln(S_t / S_{t-1}) - r_f
"""
function log_growth_matrix(dataset::Dict{String, DataFrame},
    firms::Array{String,1}; Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0,
    testfirm="AAPL", keycol::Symbol = :adj_close)::Array{Float64,2}

    number_of_firms = length(firms)
    number_of_trading_days = nrow(dataset[testfirm])
    return_matrix = Array{Float64,2}(undef, number_of_trading_days-1, number_of_firms)

    for i ∈ eachindex(firms)
        firm_index = firms[i]
        firm_data = dataset[firm_index]
        for j ∈ 2:number_of_trading_days
            S₁ = firm_data[j-1, keycol]
            S₂ = firm_data[j, keycol]
            return_matrix[j-1, i] = (1/Δt)*(log(S₂/S₁)) - risk_free_rate
        end
    end

    return return_matrix
end

"""
    compute_rolling_volatility(price_data; window=30) -> Dict{String, Dict{Date, Float64}}

Annualized rolling volatility for each ticker over a trailing window.
  σ_t = std(r_{t-W+1}, ..., r_t) · √252
Self-designed (no direct course reference).
"""
function compute_rolling_volatility(price_data::Dict{String, DataFrame};
                                     window::Int=30)::Dict{String, Dict{Date, Float64}}
    result = Dict{String, Dict{Date, Float64}}()
    for (ticker, df) in price_data
        nrow(df) < window + 1 && continue
        vol_dict = Dict{Date, Float64}()
        prices = df.adj_close
        log_returns = log.(prices[2:end] ./ prices[1:end-1])

        for k in (window+1):nrow(df)
            window_returns = log_returns[(k-window):(k-1)]
            σ = std(window_returns) * sqrt(252)
            σ = max(σ, 0.01)
            vol_dict[df.date[k]] = σ
        end

        result[ticker] = vol_dict
    end
    return result
end

"""
    trailing_dividend_yield(div_df, current_price, date) -> Float64

Compute trailing 12-month annualized dividend yield as of `date`.
Only uses dividends with ex_date in [date-365, date] — NO forward look.
"""
function trailing_dividend_yield(div_df::DataFrame, current_price::Float64, date::Date)::Float64
    current_price <= 0.0 && return 0.0
    lookback = date - Day(365)
    trailing = sum(row.amount for row in eachrow(div_df) if lookback <= row.ex_date <= date; init=0.0)
    return trailing / current_price
end

"""
    compute_return_correlation(price_data, tickers; min_overlap=60) -> Matrix{Float64}

Compute pairwise return correlation matrix for correlated multi-asset GBM (TODO item 10).
Reference: CHEME-5660 Week 6 — Multiple-Asset GBM with Cholesky decomposition.
Uses daily log returns with pairwise-complete observations.
"""
function compute_return_correlation(price_data::Dict{String, DataFrame},
                                     tickers::Vector{String};
                                     min_overlap::Int=60)::Matrix{Float64}
    n = length(tickers)
    returns = Dict{String, Dict{Date, Float64}}()

    for tk in tickers
        !haskey(price_data, tk) && continue
        df = price_data[tk]
        nrow(df) < 2 && continue
        rd = Dict{Date, Float64}()
        for i in 2:nrow(df)
            rd[df.date[i]] = log(df.adj_close[i] / df.adj_close[i-1])
        end
        returns[tk] = rd
    end

    ρ = Matrix{Float64}(I, n, n)
    for i in 1:n, j in (i+1):n
        !haskey(returns, tickers[i]) && continue
        !haskey(returns, tickers[j]) && continue
        ri = returns[tickers[i]]
        rj = returns[tickers[j]]
        common_dates = intersect(keys(ri), keys(rj))
        if length(common_dates) >= min_overlap
            vi = [ri[d] for d in common_dates]
            vj = [rj[d] for d in common_dates]
            c = cor(vi, vj)
            ρ[i, j] = clamp(c, -0.99, 0.99)
            ρ[j, i] = ρ[i, j]
        else
            ρ[i, j] = 0.0
            ρ[j, i] = 0.0
        end
    end

    return ρ
end

# ── Corporate Action Detection ───────────────────────────────────────────────

"""
    detect_stock_splits(price_data) -> Dict{String, Vector{NamedTuple}}

Detect probable stock splits by comparing adj_close / close ratio changes.
When the ratio shifts by > 10% between consecutive days, flag as a split.
Returns a dictionary of tickers → list of (date, ratio) events.

⚠ Heuristic — adj_close already corrects prices, but detecting splits lets us
warn about periods where option pricing may be unreliable.
"""
function detect_stock_splits(price_data::Dict{String, DataFrame})
    splits = Dict{String, Vector{NamedTuple{(:date, :ratio), Tuple{Date, Float64}}}}()
    for (ticker, df) in price_data
        nrow(df) < 2 && continue
        events = NamedTuple{(:date, :ratio), Tuple{Date, Float64}}[]
        for i in 2:nrow(df)
            ratio_prev = df.adj_close[i-1] / max(df.close[i-1], 0.01)
            ratio_curr = df.adj_close[i] / max(df.close[i], 0.01)
            ratio_prev == 0.0 && continue
            change = abs(ratio_curr / ratio_prev - 1.0)
            if change > 0.10
                split_ratio = round(ratio_prev / ratio_curr, digits=2)
                push!(events, (date=df.date[i], ratio=split_ratio))
            end
        end
        if !isempty(events)
            splits[ticker] = events
        end
    end
    return splits
end

# crowding_multiplier moved to OperationsCosts.jl
