"""
Risk & Compliance — Varner PDF Section 4

Key risks, core KPIs, risk overlays, position limits, and regulatory checks.

Reference:
  "Core KPIs. Distribution yield; premium capture rate; assignment rate;
   % called-away; annualized volatility; max drawdown; Sharpe/Sortino;
   Value at Risk (VaR); Expected Shortfall (ES)."
  "Regulatory. Comply with U.S. SEC Rule 18f-4. Daily relative VaR vs benchmark,
   and stress testing."
"""

# ── VaR / ES ──────────────────────────────────────────────────────────────────

"""
    compute_trailing_risk(daily_records, lookback) -> (var95, es95)

Trailing VaR and ES from daily NAV log-returns over a lookback window.
Used as risk overlay: throttle new option sales when limits exceeded.
"""
function compute_trailing_risk(daily_records, lookback::Int)
    n = length(daily_records)
    n < lookback + 1 && return (0.0, 0.0)

    navs = [daily_records[i].nav for i in (n - lookback):n]
    dr = diff(log.(navs))
    length(dr) < 5 && return (0.0, 0.0)

    srt = sort(dr)
    vi = max(1, floor(Int, 0.05 * length(srt)))
    var95 = -srt[vi]
    es95 = vi > 0 ? -mean(srt[1:vi]) : 0.0
    return (var95, es95)
end

# ── Position Limits ───────────────────────────────────────────────────────────

"""
    check_name_weight(state, price, nav, config) -> Bool

Per Varner PDF Section 3: "<= 5% NAV per name."
Returns true if ticker exposure is within cap.
"""
function check_name_weight(state, price::Float64,
                            nav::Float64, config::WheelConfig)::Bool
    nav <= 0.0 && return true
    total_shares = state.block_a_shares + sum(s.shares_held for s in state.slots)
    exposure = total_shares * price
    for slot in state.slots
        exposure += slot.capital
    end
    return exposure / nav <= config.max_name_weight
end

"""
    check_sector_cap(ticker, sector_map, portfolio, prices, nav; max_sector_weight=0.25) -> Bool

Per Varner PDF Section 3: "sector caps."
Returns true if sector exposure is within cap.
"""
function check_sector_cap(ticker::String, sector_map::Dict{String, String},
                           portfolio, prices::Dict{String, Float64},
                           nav::Float64; max_sector_weight::Float64=0.25)::Bool
    nav <= 0.0 && return true
    !haskey(sector_map, ticker) && return true
    my_sector = sector_map[ticker]

    sector_exposure = 0.0
    for (tk, state) in portfolio.states
        get(sector_map, tk, "") != my_sector && continue
        p = get(prices, tk, NaN)
        isnan(p) && continue
        ts = state.block_a_shares + sum(s.shares_held for s in state.slots)
        sector_exposure += ts * p
    end

    return sector_exposure / nav <= max_sector_weight
end

# ── KPI Computation ──────────────────────────────────────────────────────────

"""
    compute_kpis(navs, initial_nav) -> NamedTuple

Core KPIs per Varner PDF Section 4.
"""
function compute_kpis(navs::Vector{Float64}, initial_nav::Float64)
    n = length(navs)
    n == 0 && return (sharpe=0.0, sortino=0.0, max_dd=0.0, ann_vol=0.0,
                      daily_var=0.0, daily_es=0.0)
    dr = diff(log.(navs))
    nd = length(dr)
    ann_vol = nd > 0 ? std(dr) * sqrt(252) * 100.0 : 0.0
    sharpe  = nd > 0 && std(dr) > 0 ? (mean(dr)*252) / (std(dr)*sqrt(252)) : 0.0
    ds = filter(x -> x < 0, dr)
    sortino = length(ds) > 0 && std(ds) > 0 ? (mean(dr)*252) / (std(ds)*sqrt(252)) : 0.0

    pk, mdd = -Inf, 0.0
    for v in navs; pk = max(pk, v); mdd = max(mdd, (pk - v) / pk); end

    srt = sort(dr)
    vi = max(1, floor(Int, 0.05 * length(srt)))
    daily_var = length(srt) > 0 ? -srt[vi] : 0.0
    daily_es  = vi > 0 && length(srt) >= vi ? -mean(srt[1:vi]) : 0.0

    return (sharpe=sharpe, sortino=sortino, max_dd=mdd, ann_vol=ann_vol,
            daily_var=daily_var, daily_es=daily_es)
end
