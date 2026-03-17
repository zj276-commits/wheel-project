"""
Portfolio Construction — Varner PDF Section 3

Universe, weights, option settings, roll rules, liquidity screens,
adaptive controls, and market holiday calendar.

Reference:
  "Universe. Safe sleeve: bottom-quintile realized volatility, above-median dividend
   yield, sufficient option open interest (OI) -- aggressive sleeve: top-quintile
   volatility and deep options markets."
  "Weights. Initial 60/40 (Safe/Aggressive); <= 5% NAV per name; sector caps;
   >= 25 names Safe, >= 10 names Aggressive."
  "Options settings. Tenor ladder tau in {7,14,30} calendar days."
  "Roll rules. Time-based and trigger-based."
  "Liquidity screens. Minimum OI/ADV per strike and max bid/ask spread width."
"""

# ── Configuration ─────────────────────────────────────────────────────────────

Base.@kwdef struct WheelConfig
    risk_free_rate::Float64 = 0.045
    # Options settings (PDF Section 3)
    tenor_days::Vector{Int} = [7, 14, 30]
    delta_put_safe::Tuple{Float64, Float64} = (0.20, 0.30)
    delta_call_safe::Tuple{Float64, Float64} = (0.25, 0.35)
    delta_put_aggr::Tuple{Float64, Float64} = (0.25, 0.35)
    delta_call_aggr::Tuple{Float64, Float64} = (0.30, 0.40)
    # Roll rules (PDF Section 3)
    roll_dte_min::Int = 3
    roll_dte_max::Int = 5
    premium_decay_threshold::Float64 = 0.80
    itm_otm_band_pct::Float64 = 0.05
    breakeven_breach_pct::Float64 = 0.02
    # Basis-repair policy (PDF Section 6)
    repair_loss_trigger::Float64 = 0.10
    max_avg_down_pct::Float64 = 0.25
    max_repairs_per_qtr::Int = 2
    # Fees (PDF Section 5)
    commission_per_contract::Float64 = 0.65
    slippage_pct::Float64 = 0.02
    mgmt_fee_annual::Float64 = 0.0068
    # Laddering (PDF Section 6)
    max_ladders::Int = 1
    # Earnings policy (PDF Section 6)
    earnings_buffer_days::Int = 5
    earnings_policy::Symbol = :avoid
    earnings_wider_delta::Float64 = 0.05
    earnings_size_reduction::Float64 = 0.50
    # Spread model
    bid_ask_spread_model::Bool = true
    crr_steps::Int = 50
    # Extended cost model (PDF Section 5)
    exchange_fee_per_contract::Float64 = 0.03
    clearing_fee_per_contract::Float64 = 0.02
    borrow_fee_annual::Float64 = 0.005
    # Liquidity screens (PDF Section 3)
    min_open_interest::Int = 100
    max_spread_width_pct::Float64 = 0.10
    # Partial fill model (PDF Section 7A)
    fill_rate_model::Bool = true
    min_fill_rate::Float64 = 0.5
    # Risk overlays (PDF Section 4)
    var_limit_daily::Float64 = 0.025
    es_limit_daily::Float64 = 0.035
    risk_lookback::Int = 20
    # Position limits (PDF Section 3)
    max_name_weight::Float64 = 0.05
    # Adaptive controls (PDF Section 3)
    adaptive_tenor::Bool = true
    adaptive_delta::Bool = true
    vol_high_threshold::Float64 = 0.40
    vol_low_threshold::Float64 = 0.15
end

function default_config()::WheelConfig
    return WheelConfig()
end

# ── US Market Holiday Calendar ────────────────────────────────────────────────

const US_MARKET_HOLIDAYS_2024 = Set{Date}([
    Date(2024, 1, 1),   Date(2024, 1, 15),  Date(2024, 2, 19),
    Date(2024, 3, 29),  Date(2024, 5, 27),  Date(2024, 6, 19),
    Date(2024, 7, 4),   Date(2024, 9, 2),   Date(2024, 11, 28),
    Date(2024, 12, 25),
])

const US_MARKET_HOLIDAYS_2025 = Set{Date}([
    Date(2025, 1, 1),   Date(2025, 1, 20),  Date(2025, 2, 17),
    Date(2025, 4, 18),  Date(2025, 5, 26),  Date(2025, 6, 19),
    Date(2025, 7, 4),   Date(2025, 9, 1),   Date(2025, 11, 27),
    Date(2025, 12, 25),
])

const US_MARKET_HOLIDAYS = union(US_MARKET_HOLIDAYS_2024, US_MARKET_HOLIDAYS_2025)

# ── Tenor & Expiry ────────────────────────────────────────────────────────────

"""
    select_tenor(config, date; σ) -> Int

Adaptive tenor: shorter in high-vol (more annualized premium),
longer in low-vol (fewer transaction costs).
"""
function select_tenor(config::WheelConfig, date::Date;
                       σ::Union{Nothing, Float64}=nothing)::Int
    if config.adaptive_tenor && σ !== nothing
        σ > config.vol_high_threshold && return minimum(config.tenor_days)
        σ < config.vol_low_threshold  && return maximum(config.tenor_days)
    end
    idx = mod(Dates.week(date), length(config.tenor_days)) + 1
    return config.tenor_days[idx]
end

"""
    find_expiry(date, tenor_days) -> Date

Round to next Friday, skip US market holidays.
"""
function find_expiry(date::Date, tenor_days::Int)::Date
    target = date + Day(tenor_days)
    dow = dayofweek(target)
    expiry = dow <= 5 ? target + Day(5 - dow) : target + Day(12 - dow)
    while expiry in US_MARKET_HOLIDAYS || dayofweek(expiry) > 5
        expiry -= Day(1)
    end
    return expiry
end

# ── Delta Targeting ───────────────────────────────────────────────────────────

"""
    get_target_delta(state, config, otype; σ, near_earn, drawdown_pct) -> Float64

Adaptive delta within configured range per sleeve/option type.
High vol or large drawdown → lower delta (more OTM, safer).
Low vol → higher delta (more ITM, higher premium).
"""
function get_target_delta(state, config::WheelConfig, otype::Symbol;
                           σ::Union{Nothing, Float64}=nothing,
                           near_earn::Bool=false,
                           drawdown_pct::Float64=0.0)::Float64
    r = if state.sleeve == "Safe"
        otype == :put ? config.delta_put_safe : config.delta_call_safe
    else
        otype == :put ? config.delta_put_aggr : config.delta_call_aggr
    end

    if !config.adaptive_delta || σ === nothing
        return (r[1] + r[2]) / 2.0
    end

    t = 0.5
    if σ > config.vol_high_threshold
        t = max(0.0, 0.5 - (σ - config.vol_high_threshold) * 2.0)
    elseif σ < config.vol_low_threshold
        t = min(1.0, 0.5 + (config.vol_low_threshold - σ) * 2.0)
    end
    near_earn    && (t = max(0.0, t - 0.2))
    drawdown_pct > 0.05 && (t = max(0.0, t - drawdown_pct))

    return r[1] + t * (r[2] - r[1])
end
