"""
Quantitative computations: log growth matrix, rolling volatility, dividend yield,
implied volatility calibration, and return correlation.

log_growth_matrix: excess log return matrix for cross-sectional analysis.
  Reference: CHEME-5660 Week 5b — SAGBM parameter estimation.
  VLQuantitativeFinancePackage exports a similar function; ours adapts the
  column names to our Yahoo Finance data format (keycol=:adj_close).

compute_rolling_volatility: time-varying σ for each ticker.

compute_dividend_yields: annualized dividend yield per ticker.
  Used to pass q to CRR pricing for accurate American option valuation.

calibrate_implied_vol: VRP model to convert realized vol → implied vol.
  Fallback when WRDS IV data is unavailable (see IVData.jl for primary source).
  Motivated by Varner PDF §7B: "calibrated to each name's IV surface."

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
    compute_dividend_yields(div_data, price_data; lookback_years=1.0) -> Dict{String, Float64}

⚠ DEPRECATED — this function uses the FULL dataset (forward-looking).
Kept for backward compatibility. Prefer trailing_dividend_yield() in the backtest loop.
"""
function compute_dividend_yields(div_data::Dict{String, DataFrame},
                                  price_data::Dict{String, DataFrame};
                                  lookback_years::Float64=1.0)::Dict{String, Float64}
    yields = Dict{String, Float64}()
    for (ticker, ddf) in div_data
        if nrow(ddf) == 0 || !haskey(price_data, ticker)
            yields[ticker] = 0.0
            continue
        end
        total_div = sum(ddf.amount)
        pdf = price_data[ticker]
        nrow(pdf) == 0 && (yields[ticker] = 0.0; continue)
        avg_price = mean(pdf.adj_close)
        avg_price <= 0.0 && (yields[ticker] = 0.0; continue)
        yields[ticker] = max(0.0, total_div / avg_price / lookback_years)
    end
    return yields
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

# ── Implied Volatility Calibration ────────────────────────────────────────────
# ⚠ SELF-DESIGNED — entire IV calibration module is original work.
# No course reference or textbook formula was used.
# Motivated by Varner PDF §7B: "calibrated to each name's IV surface."
#
# The VRP concept is well-established in academic finance:
#   Carr & Wu (2009) "Variance Risk Premiums"
#   Bollerslev, Tauchen & Zhou (2009) "Expected Stock Returns and Variance Risk Premia"
# But our specific parametric model (VRP × term × skew × volvol) is original.
#
# ⚠ DATA NEEDED FOR BETTER CALIBRATION:
#   Currently uses fixed multipliers (Safe=1.15×, Aggressive=1.25×).
#   For production accuracy, you would need:
#   1. Real options chain data (bid/ask for multiple strikes) → use estimate_implied_vol()
#      to extract actual IV → regress IV/RV ratio per ticker
#   2. VIX historical data → make VRP time-varying (VRP spikes during market panics)
#   3. CBOE DataShop or OptionMetrics subscription for historical IV surfaces

"""
    IVCalibration

⚠ SELF-DESIGNED — Parametric model for converting realized vol to implied vol.

The variance risk premium (VRP) is one of the most robust findings in
empirical finance: implied volatility systematically exceeds realized
volatility. Option sellers earn this premium for bearing volatility risk.

Model:  σ_IV(S, K, T) = σ_RV × vrp_multiplier × term_adjustment(T) × skew_adjustment(moneyness)

Components:
  vrp_multiplier — base IV/RV ratio (empirically 1.1–1.3 for equity options)
  term_adjustment — shorter-dated options have higher IV/RV ratio
  skew_adjustment — OTM puts have higher IV (negative skew in equities)
"""
struct IVCalibration
    vrp_multiplier::Float64     # base IV/RV ratio
    term_slope::Float64         # term structure: IV/RV increases as T → 0
    skew_slope::Float64         # put skew: IV increases for OTM puts
    vol_of_vol::Float64         # IV increases more when RV is already high
end

"""
    default_iv_calibration(sleeve) -> IVCalibration

Default IV calibration parameters by sleeve type.
Aggressive (high-vol) names have higher VRP and steeper skew.
"""
function default_iv_calibration(sleeve::String)::IVCalibration
    if sleeve == "Aggressive"
        return IVCalibration(1.25, 0.15, 0.08, 0.10)
    else
        return IVCalibration(1.15, 0.10, 0.05, 0.05)
    end
end

"""
    calibrate_iv(σ_rv, T, moneyness, cal) -> Float64

Convert realized volatility to implied volatility using the VRP model.

Arguments:
  σ_rv      — annualized realized vol from rolling window
  T         — time to expiry in years
  moneyness — (S - K) / S, negative for OTM puts, positive for OTM calls
  cal       — IVCalibration parameters

Model:
  σ_IV = σ_RV × VRP × (1 + term_adj) × (1 + skew_adj) × (1 + volvol_adj)

  term_adj  = term_slope × max(0, 30/365 - T) / (30/365)
    → increases IV for options shorter than 30 DTE

  skew_adj  = skew_slope × max(0, -moneyness)
    → increases IV for OTM puts (negative moneyness)

  volvol_adj = vol_of_vol × max(0, σ_RV - 0.30)
    → IV rises faster than RV in high-vol environments
"""
function calibrate_iv(σ_rv::Float64, T::Float64, moneyness::Float64,
                       cal::IVCalibration)::Float64
    σ_rv <= 0.0 && return 0.01

    term_adj = cal.term_slope * max(0.0, (30.0/365.0 - T) / (30.0/365.0))

    skew_adj = cal.skew_slope * max(0.0, -moneyness)

    volvol_adj = cal.vol_of_vol * max(0.0, σ_rv - 0.30)

    σ_iv = σ_rv * cal.vrp_multiplier * (1.0 + term_adj) * (1.0 + skew_adj) * (1.0 + volvol_adj)

    return clamp(σ_iv, 0.01, 5.0)
end

"""
    compute_rolling_iv(rolling_vol, sleeves_map; iv_cals=nothing) -> Dict{String, Dict{Date, Float64}}

Convert rolling realized vol to rolling implied vol for all tickers.
Applies the VRP model with default ATM moneyness and 30-day tenor.

This is the primary integration point: the backtest can use this instead of
rolling_vol to price options with implied volatility.

Arguments:
  rolling_vol  — Dict{String, Dict{Date, Float64}} from compute_rolling_volatility
  sleeves_map  — Dict{String, String} mapping ticker → sleeve ("Safe" or "Aggressive")
  iv_cals      — optional Dict{String, IVCalibration} for per-ticker overrides
"""
function compute_rolling_iv(rolling_vol::Dict{String, Dict{Date, Float64}},
                              sleeves_map::Dict{String, String};
                              iv_cals::Union{Nothing, Dict{String, IVCalibration}}=nothing
                              )::Dict{String, Dict{Date, Float64}}
    result = Dict{String, Dict{Date, Float64}}()

    for (ticker, vol_dict) in rolling_vol
        sleeve = get(sleeves_map, ticker, "Safe")
        cal = if iv_cals !== nothing && haskey(iv_cals, ticker)
            iv_cals[ticker]
        else
            default_iv_calibration(sleeve)
        end

        iv_dict = Dict{Date, Float64}()
        for (date, σ_rv) in vol_dict
            iv_dict[date] = calibrate_iv(σ_rv, 30.0/365.0, 0.0, cal)
        end
        result[ticker] = iv_dict
    end

    return result
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
