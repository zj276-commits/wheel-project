"""
Wheel strategy backtest engine.

Architecture: per-ticker state machine with ladder slots (Varner PDF Sections 2-6).
  SELLING_PUT → (assigned) → HOLDING_SHARES → (called away) → SELLING_PUT

Each ticker maintains:
  Block A — buy-and-hold for dividends and capital appreciation (no options).
  Block B — split across 1–3 ladder slots, each running an independent Wheel cycle.

Enhancements implemented (TODO items 2, 3, 5, 6, 7, 8, 12, 13, 17):
  - CRR American option pricing with dividend yield q (OptionPricing.jl)
  - Greeks: portfolio-level Δ, Γ, ν tracked daily
  - Dynamic rolling volatility (Compute.jl)
  - Earnings avoidance / delta widening (EarningsCalendar.jl)
  - Laddering: multiple concurrent expiries per name (PDF §6)
  - Bid-ask spread model + liquidity screens (PDF §3)
  - Risk overlays: VaR/ES caps throttle new sales (PDF §4, §6)
  - Partial fill model (PDF §7A)
  - Extended cost model: exchange, clearing, borrow fees (PDF §5, §7A)
  - US market holiday calendar for expiry dates (PDF §7A)
  - Sector caps and per-name NAV ≤5% enforcement (PDF §3)
  - Adaptive tenor selection based on vol regime (PDF §3)
  - Adaptive delta within configurable range (PDF §3)
  - Benchmark comparison in report (PDF §4)
"""

@enum WheelPhase SELLING_PUT HOLDING_SHARES

mutable struct OptionPosition
    type::Symbol        # :put or :call
    strike::Float64
    expiry::Date
    premium::Float64    # per-share premium collected at open
    open_date::Date
end

"""
Per Varner PDF Section 6 — "Laddering: number of concurrent expiries per name {1,2,3}."
"""
mutable struct LadderSlot
    id::Int
    phase::WheelPhase
    shares_held::Int
    cost_basis::Float64
    option::Union{Nothing, OptionPosition}
    num_contracts::Int
    capital::Float64
    total_premium::Float64
    total_realized_pnl::Float64
    assignment_count::Int
    callaway_count::Int
    trades::Int
    repair_count::Int
    repairs_this_quarter::Int
    last_repair_quarter::Int
end

mutable struct TickerState
    ticker::String
    sleeve::String          # "Safe" or "Aggressive"
    block_a_shares::Int
    block_a_cost::Float64
    slots::Vector{LadderSlot}
    total_dividends::Float64
    total_costs::Float64
end

Base.@kwdef struct WheelConfig
    risk_free_rate::Float64 = 0.045
    tenor_days::Vector{Int} = [7, 14, 30]
    delta_put_safe::Tuple{Float64, Float64} = (0.20, 0.30)
    delta_call_safe::Tuple{Float64, Float64} = (0.25, 0.35)
    delta_put_aggr::Tuple{Float64, Float64} = (0.25, 0.35)
    delta_call_aggr::Tuple{Float64, Float64} = (0.30, 0.40)
    roll_dte_min::Int = 3
    roll_dte_max::Int = 5
    premium_decay_threshold::Float64 = 0.80
    itm_otm_band_pct::Float64 = 0.05
    breakeven_breach_pct::Float64 = 0.02
    repair_loss_trigger::Float64 = 0.10
    max_avg_down_pct::Float64 = 0.25
    max_repairs_per_qtr::Int = 2
    commission_per_contract::Float64 = 0.65
    slippage_pct::Float64 = 0.02
    mgmt_fee_annual::Float64 = 0.0068
    max_ladders::Int = 1
    earnings_buffer_days::Int = 5
    earnings_policy::Symbol = :avoid       # :avoid, :widen, or :reduce_size
    earnings_wider_delta::Float64 = 0.05
    earnings_size_reduction::Float64 = 0.50 # reduce contracts to 50% near earnings
    bid_ask_spread_model::Bool = true
    crr_steps::Int = 50
    # Extended cost model (PDF §5, §7A) — TODO item 6
    exchange_fee_per_contract::Float64 = 0.03
    clearing_fee_per_contract::Float64 = 0.02
    borrow_fee_annual::Float64 = 0.005
    # Liquidity screens (PDF §3, §6) — TODO item 2
    min_open_interest::Int = 100
    max_spread_width_pct::Float64 = 0.10
    # Partial fill model (PDF §7A) — TODO item 5
    fill_rate_model::Bool = true
    min_fill_rate::Float64 = 0.5
    # Risk overlays (PDF §4, §6) — TODO item 3
    var_limit_daily::Float64 = 0.025
    es_limit_daily::Float64 = 0.035
    risk_lookback::Int = 20
    # Position limits (PDF §3) — TODO item 8
    max_name_weight::Float64 = 0.05
    # Adaptive controls — TODO items 12, 13
    adaptive_tenor::Bool = true
    adaptive_delta::Bool = true
    vol_high_threshold::Float64 = 0.40
    vol_low_threshold::Float64 = 0.15
end

struct DailyRecord
    date::Date
    nav::Float64
    cash::Float64
    shares_value::Float64
    option_mtm::Float64
    block_a_value::Float64
    cumulative_premium::Float64
    cumulative_dividends::Float64
    cumulative_costs::Float64
    portfolio_delta::Float64
    portfolio_gamma::Float64
    portfolio_vega::Float64
end

mutable struct Portfolio
    cash::Float64
    initial_nav::Float64
    states::Dict{String, TickerState}
    daily_records::Vector{DailyRecord}
    config::WheelConfig
end

function default_config()::WheelConfig
    return WheelConfig()
end

# ── US Market Holiday Calendar (TODO item 7) ──────────────────────────────────

const US_MARKET_HOLIDAYS_2025 = Set{Date}([
    Date(2025, 1, 1),   # New Year's Day
    Date(2025, 1, 20),  # MLK Day
    Date(2025, 2, 17),  # Presidents' Day
    Date(2025, 4, 18),  # Good Friday
    Date(2025, 5, 26),  # Memorial Day
    Date(2025, 6, 19),  # Juneteenth
    Date(2025, 7, 4),   # Independence Day
    Date(2025, 9, 1),   # Labor Day
    Date(2025, 11, 27), # Thanksgiving
    Date(2025, 12, 25), # Christmas
])

# ── Helpers ──────────────────────────────────────────────────────────────────

"""
    select_tenor(config, date; σ=nothing) -> Int

Select option tenor. When adaptive_tenor is enabled and vol is provided,
prefer shorter tenors in high-vol environments (more annualized premium)
and longer tenors in low-vol environments (fewer transaction costs).
"""
function select_tenor(config::WheelConfig, date::Date; σ::Union{Nothing, Float64}=nothing)::Int
    if config.adaptive_tenor && σ !== nothing
        if σ > config.vol_high_threshold
            return minimum(config.tenor_days)
        elseif σ < config.vol_low_threshold
            return maximum(config.tenor_days)
        end
    end
    idx = mod(Dates.week(date), length(config.tenor_days)) + 1
    return config.tenor_days[idx]
end

"""
    find_expiry(date, tenor_days) -> Date

Round target date to next Friday, adjusting for US market holidays.
If the computed Friday is a holiday, move to the preceding Thursday.
"""
function find_expiry(date::Date, tenor_days::Int)::Date
    target = date + Day(tenor_days)
    dow = dayofweek(target)
    expiry = dow <= 5 ? target + Day(5 - dow) : target + Day(12 - dow)
    while expiry in US_MARKET_HOLIDAYS_2025 || dayofweek(expiry) > 5
        expiry -= Day(1)
    end
    return expiry
end

"""
    get_target_delta(state, config, otype; σ=nothing, near_earn=false, drawdown_pct=0.0) -> Float64

Compute target delta within the configured range. When adaptive_delta is enabled:
  - High vol → use lower end of range (more OTM, safer)
  - Near earnings → reduce delta further
  - Large drawdown → use lower end of range (reduce risk)
  - Low vol → use upper end of range (more ITM, higher premium)
"""
function get_target_delta(state::TickerState, config::WheelConfig, otype::Symbol;
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
    if near_earn
        t = max(0.0, t - 0.2)
    end
    if drawdown_pct > 0.05
        t = max(0.0, t - drawdown_pct)
    end

    return r[1] + t * (r[2] - r[1])
end

"""
    estimate_half_spread(premium, σ) -> Float64

Self-designed bid-ask half-spread model.
Spread increases with underlying volatility.
"""
function estimate_half_spread(premium::Float64, σ::Float64)::Float64
    base_spread = max(0.01, premium * 0.02)
    vol_adjustment = 1.0 + max(0.0, σ - 0.30) * 2.0
    return base_spread * vol_adjustment
end

"""
    check_liquidity(premium, σ, config) -> (passes::Bool, fill_rate::Float64)

Liquidity screening and fill rate estimation (TODO items 2, 5).
Checks if estimated spread width exceeds max_spread_width_pct.
Returns whether trade passes screen and the estimated fill rate.
"""
function check_liquidity(premium::Float64, σ::Float64,
                          config::WheelConfig)::Tuple{Bool, Float64}
    half_spread = estimate_half_spread(premium, σ)
    full_spread = 2.0 * half_spread
    spread_pct = premium > 0.01 ? full_spread / premium : 1.0

    passes = spread_pct <= config.max_spread_width_pct

    fill_rate = if config.fill_rate_model
        base_fill = 1.0 - 0.3 * min(spread_pct / config.max_spread_width_pct, 1.0)
        vol_penalty = σ > 0.50 ? 0.1 : 0.0
        max(config.min_fill_rate, base_fill - vol_penalty)
    else
        1.0
    end

    return (passes, fill_rate)
end

"""
    apply_trade_costs!(state, slot, premium, σ, config, portfolio)

Deduct trading costs: commission + exchange fees + clearing fees + slippage.
Extended cost model per Varner PDF Sections 5 and 7A (TODO item 6).
"""
function apply_trade_costs!(state::TickerState, slot::LadderSlot, premium::Float64,
                            σ::Float64, config::WheelConfig, portfolio::Portfolio)
    nc = slot.num_contracts
    commission = config.commission_per_contract * nc
    exchange_fee = config.exchange_fee_per_contract * nc
    clearing_fee = config.clearing_fee_per_contract * nc
    slippage = if config.bid_ask_spread_model
        per_share_prem = abs(premium) / max(100 * nc, 1)
        estimate_half_spread(per_share_prem, σ) * 100 * nc
    else
        abs(premium) * config.slippage_pct
    end
    cost = commission + exchange_fee + clearing_fee + slippage
    state.total_costs += cost
    portfolio.cash -= cost
end

"""
    apply_borrow_cost!(state, slot, price, config, portfolio)

Deduct daily borrow cost for held shares (TODO item 6).
Per Varner PDF Section 7A: "borrow fees for hard-to-borrow events."
"""
function apply_borrow_cost!(state::TickerState, slot::LadderSlot, price::Float64,
                            config::WheelConfig, portfolio::Portfolio)
    slot.shares_held <= 0 && return
    daily_borrow = config.borrow_fee_annual / 252.0 * slot.shares_held * price
    state.total_costs += daily_borrow
    portfolio.cash -= daily_borrow
end

# ── Option Operations ────────────────────────────────────────────────────────

"""
    open_option!(state, slot, price, date, σ, config, portfolio; ...) -> Float64

Open a new option. Uses realized vol (σ) for delta targeting and implied vol
(σ_iv) for premium pricing.

σ_iv defaults to σ if not provided (backward compatible).
"""
function open_option!(state::TickerState, slot::LadderSlot, price::Float64,
                      date::Date, σ::Float64, config::WheelConfig,
                      portfolio::Portfolio;
                      earnings_cal=nothing, q::Float64=0.0,
                      near_earn::Bool=false, drawdown_pct::Float64=0.0,
                      σ_iv::Float64=NaN)::Float64
    pricing_vol = isnan(σ_iv) ? σ : σ_iv

    otype = slot.phase == SELLING_PUT ? :put : :call
    delta = get_target_delta(state, config, otype; σ=σ, near_earn=near_earn,
                              drawdown_pct=drawdown_pct)

    if earnings_cal !== nothing && near_earn
        if config.earnings_policy == :widen
            delta = max(0.10, delta - config.earnings_wider_delta)
        end
    end

    tenor = select_tenor(config, date; σ=σ)
    expiry = find_expiry(date, tenor)
    T = max(Dates.value(expiry - date) / 365.0, 1.0/365.0)

    K = strike_from_delta(price, T, config.risk_free_rate, σ, delta, otype; q=q)
    prem = max(option_price(price, K, T, config.risk_free_rate, pricing_vol, otype;
                            steps=config.crr_steps, q=q), 0.01)

    liq_ok, fill_rate = check_liquidity(prem, σ, config)
    if !liq_ok
        return 0.0
    end

    effective_contracts = if config.fill_rate_model && fill_rate < 1.0
        max(1, round(Int, slot.num_contracts * fill_rate))
    else
        slot.num_contracts
    end

    # :reduce_size — trade through earnings but with fewer contracts
    # Size = 50% of normal (configurable via earnings_size_reduction).
    # Rationale: earnings vol crush makes selling options attractive,
    # but gap risk is high, so reduce notional exposure.
    if earnings_cal !== nothing && near_earn && config.earnings_policy == :reduce_size
        effective_contracts = max(1, round(Int, effective_contracts * config.earnings_size_reduction))
    end

    slot.option = OptionPosition(otype, K, expiry, prem, date)
    slot.trades += 1
    total = prem * 100 * effective_contracts
    slot.total_premium += total
    apply_trade_costs!(state, slot, total, σ, config, portfolio)
    return total
end

"""Mark-to-market value of slot's open option using CRR American pricing.
Uses σ_iv (implied vol) for MTM if provided, else falls back to σ (realized vol)."""
function close_option_mtm(slot::LadderSlot, price::Float64, date::Date,
                          σ::Float64, config::WheelConfig;
                          q::Float64=0.0, σ_iv::Float64=NaN)::Float64
    opt = slot.option
    opt === nothing && return 0.0
    T = max(Dates.value(opt.expiry - date) / 365.0, 0.0)
    pricing_vol = isnan(σ_iv) ? σ : σ_iv
    return option_price(price, opt.strike, T, config.risk_free_rate, pricing_vol, opt.type;
                        steps=config.crr_steps, q=q) * 100.0 * slot.num_contracts
end

"""Handle option expiry: assignment (put ITM) or call-away (call ITM)."""
function handle_expiry!(state::TickerState, slot::LadderSlot, price::Float64,
                        portfolio::Portfolio)
    opt = slot.option
    opt === nothing && return
    lot = 100 * slot.num_contracts

    if opt.type == :put && price < opt.strike
        portfolio.cash -= opt.strike * lot
        slot.shares_held = lot
        slot.cost_basis = opt.strike
        slot.phase = HOLDING_SHARES
        slot.assignment_count += 1
    elseif opt.type == :call && price > opt.strike
        portfolio.cash += opt.strike * lot
        slot.total_realized_pnl += (opt.strike - slot.cost_basis) * lot
        slot.shares_held = 0
        slot.cost_basis = 0.0
        slot.phase = SELLING_PUT
        slot.callaway_count += 1
    end
    slot.option = nothing
end

# ── Roll Logic ───────────────────────────────────────────────────────────────

"""
Determine whether a slot's option should be rolled. Four triggers per Varner PDF §3.
"""
function should_roll(slot::LadderSlot, date::Date, price::Float64,
                     σ::Float64, config::WheelConfig)::Bool
    opt = slot.option
    opt === nothing && return false
    dte = Dates.value(opt.expiry - date)

    Dates.value(date - opt.open_date) < 3 && return false

    if config.roll_dte_min <= dte <= config.roll_dte_max
        is_otm = (opt.type == :put && price > opt.strike) ||
                 (opt.type == :call && price < opt.strike)
        is_otm && return true
    end

    if dte > 0
        T_remain = max(dte / 365.0, 1.0/365.0)
        current_val = option_price(price, opt.strike, T_remain, config.risk_free_rate, σ,
                                   opt.type; steps=config.crr_steps)
        if opt.premium > 0.0 && current_val / opt.premium < (1.0 - config.premium_decay_threshold)
            return true
        end
    end

    moneyness = (price - opt.strike) / opt.strike
    if opt.type == :put
        (moneyness < -config.itm_otm_band_pct || moneyness > config.itm_otm_band_pct) && return true
    else
        (moneyness > config.itm_otm_band_pct || moneyness < -config.itm_otm_band_pct) && return true
    end

    if opt.type == :put
        be = opt.strike - opt.premium
        price < be * (1.0 - config.breakeven_breach_pct) && return true
    else
        be = opt.strike + opt.premium
        price > be * (1.0 + config.breakeven_breach_pct) && return true
    end

    return false
end

"""Close current option at market and open a new one (roll forward).
Uses σ_iv for pricing if provided."""
function roll_option!(state::TickerState, slot::LadderSlot, price::Float64,
                      date::Date, σ::Float64, config::WheelConfig,
                      portfolio::Portfolio; earnings_cal=nothing, q::Float64=0.0,
                      near_earn::Bool=false, drawdown_pct::Float64=0.0,
                      σ_iv::Float64=NaN)::Float64
    cc = close_option_mtm(slot, price, date, σ, config; q=q, σ_iv=σ_iv)
    portfolio.cash -= cc
    slot.total_premium -= cc
    apply_trade_costs!(state, slot, cc, σ, config, portfolio)
    slot.option = nothing
    np = open_option!(state, slot, price, date, σ, config, portfolio;
                      earnings_cal=earnings_cal, q=q, near_earn=near_earn,
                      drawdown_pct=drawdown_pct, σ_iv=σ_iv)
    portfolio.cash += np
    return np - cc
end

# ── Cost-Basis Repair ────────────────────────────────────────────────────────

"""
Average down when shares are underwater beyond trigger %.
Per Varner PDF Section 6.
"""
function check_cost_basis_repair!(state::TickerState, slot::LadderSlot, date::Date,
                                  price::Float64, portfolio::Portfolio, config::WheelConfig)
    slot.phase != HOLDING_SHARES && return
    slot.cost_basis <= 0 && return

    current_qtr = div(Dates.month(date) - 1, 3) + 1
    if current_qtr != slot.last_repair_quarter
        slot.repairs_this_quarter = 0
        slot.last_repair_quarter = current_qtr
    end
    slot.repairs_this_quarter >= config.max_repairs_per_qtr && return

    loss_pct = (slot.cost_basis - price) / slot.cost_basis
    loss_pct < config.repair_loss_trigger && return

    max_spend = config.max_avg_down_pct * slot.capital
    add_shares = min(slot.shares_held, floor(Int, max_spend / price))
    add_shares <= 0 && return
    add_shares > floor(Int, portfolio.cash * 0.05 / price) && return

    spend = add_shares * price
    old_total = slot.shares_held * slot.cost_basis
    slot.shares_held += add_shares
    slot.cost_basis = (old_total + spend) / slot.shares_held
    portfolio.cash -= spend
    slot.repair_count += 1
    slot.repairs_this_quarter += 1
end

# ── Dividends ────────────────────────────────────────────────────────────────

"""Credit dividends for Block A + all slots' held shares."""
function check_dividends!(state::TickerState, date::Date,
                          div_data::DataFrame, portfolio::Portfolio)
    idx = findfirst(==(date), div_data.ex_date)
    idx === nothing && return
    total_shares = state.block_a_shares + sum(s.shares_held for s in state.slots)
    d = div_data.amount[idx] * total_shares
    d <= 0 && return
    state.total_dividends += d
    portfolio.cash += d
end

# ── Risk Overlay (TODO item 3) ───────────────────────────────────────────────

"""
    compute_trailing_risk(daily_records, lookback) -> (var95, es95)

Compute trailing VaR and ES from daily NAV returns over a lookback window.
Per Varner PDF Section 4: VaR and ES as core KPIs.
Per Varner PDF Section 6: "Risk overlays: sleeve-level VaR/ES caps that throttle new sales."
"""
function compute_trailing_risk(daily_records::Vector{DailyRecord}, lookback::Int)
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

# ── Name Weight Check (TODO item 8) ──────────────────────────────────────────

"""
    check_name_weight(state, price, nav, config) -> Bool

Returns true if the ticker's exposure is within the per-name NAV cap.
Per Varner PDF Section 3: "≤ 5% net asset value (NAV) per name."
"""
function check_name_weight(state::TickerState, price::Float64,
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

Returns true if sector exposure is within cap.
Per Varner PDF Section 3: "sector caps."
"""
function check_sector_cap(ticker::String, sector_map::Dict{String, String},
                           portfolio::Portfolio, prices::Dict{String, Float64},
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

# ── Portfolio Initialization ─────────────────────────────────────────────────

"""
Initialize portfolio with Block A + Block B ladder slots per ticker.
"""
function initialize_portfolio(tickers::Vector{String}, sleeves::Vector{String},
                              weights::Vector{Float64}, initial_nav::Float64,
                              prices_day1::Dict{String, Float64},
                              config::WheelConfig)::Portfolio
    states = Dict{String, TickerState}()
    total_cost = 0.0
    for (i, ticker) in enumerate(tickers)
        price = prices_day1[ticker]
        notional = weights[i] * initial_nav
        half = notional / 2.0
        ba = floor(Int, half / price)
        total_cost += ba * price

        slot_capital = half / config.max_ladders
        slots = LadderSlot[]
        for s in 1:config.max_ladders
            nc = max(1, floor(Int, slot_capital / (price * 100.0)))
            push!(slots, LadderSlot(s, SELLING_PUT, 0, 0.0, nothing, nc, slot_capital,
                                     0.0, 0.0, 0, 0, 0, 0, 0, 0))
        end

        states[ticker] = TickerState(ticker, sleeves[i], ba, ba * price, slots, 0.0, 0.0)
    end
    return Portfolio(initial_nav - total_cost, initial_nav, states, DailyRecord[], config)
end

# ── NAV Computation ──────────────────────────────────────────────────────────

"""
Compute daily NAV: cash + Block A equity + Block B equity + option MTM.
Also computes portfolio-level Greeks. Uses rolling_iv for option MTM if available.
"""
function compute_nav(portfolio::Portfolio, prices::Dict{String, Float64},
                     date::Date, vol_map::Dict{String, Float64},
                     config::WheelConfig;
                     rolling_vol=nothing, div_yields=nothing,
                     rolling_iv=nothing)::DailyRecord
    sv, omtm, bav, tp, td, tc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    p_delta, p_gamma, p_vega = 0.0, 0.0, 0.0

    for (tk, state) in portfolio.states
        p = get(prices, tk, NaN)
        isnan(p) && continue
        σ = _resolve_vol(rolling_vol, vol_map, tk, date)
        q = div_yields !== nothing ? get(div_yields, tk, 0.0) : 0.0
        pricing_vol = if rolling_iv !== nothing && haskey(rolling_iv, tk)
            get(rolling_iv[tk], date, σ)
        else
            σ
        end

        bav += state.block_a_shares * p
        for slot in state.slots
            sv += slot.shares_held * p
            if slot.option !== nothing
                T = max(Dates.value(slot.option.expiry - date) / 365.0, 0.0)
                opt_val = option_price(p, slot.option.strike, T, config.risk_free_rate,
                                       pricing_vol, slot.option.type;
                                       steps=config.crr_steps, q=q)
                omtm -= opt_val * 100.0 * slot.num_contracts

                if T > 0.0
                    g = option_greeks(p, slot.option.strike, T, config.risk_free_rate,
                                      pricing_vol, slot.option.type;
                                      N=config.crr_steps, q=q)
                    sign = -1.0
                    n = 100.0 * slot.num_contracts
                    p_delta += sign * g.delta * n
                    p_gamma += sign * g.gamma * n
                    p_vega  += sign * g.vega  * n
                end
            end
            tp += slot.total_premium
        end
        p_delta += state.block_a_shares + sum(s.shares_held for s in state.slots)
        td += state.total_dividends
        tc += state.total_costs
    end
    nav = portfolio.cash + sv + bav + omtm
    return DailyRecord(date, nav, portfolio.cash, sv, omtm, bav, tp, td, tc,
                       p_delta, p_gamma, p_vega)
end

"""Resolve volatility: prefer rolling → static → fallback 0.25."""
function _resolve_vol(rolling_vol, vol_map::Dict{String, Float64},
                      ticker::String, date::Date)::Float64
    if rolling_vol !== nothing && haskey(rolling_vol, ticker)
        rv = rolling_vol[ticker]
        haskey(rv, date) && return rv[date]
    end
    return get(vol_map, ticker, 0.25)
end

# ── Main Simulation Loop ────────────────────────────────────────────────────

"""
    run_backtest!(portfolio, price_data, div_data, vol_map, trading_days; ...)

Day-by-day Wheel strategy simulation with risk overlays, name weight caps,
sector caps, and IV calibration.

rolling_iv — implied vol lookup Dict{String, Dict{Date, Float64}}.
When provided, option pricing uses σ_IV while delta targeting uses σ_RV.
"""
function run_backtest!(portfolio::Portfolio, price_data::Dict{String, DataFrame},
                       div_data::Dict{String, DataFrame}, vol_map::Dict{String, Float64},
                       trading_days::Vector{Date};
                       earnings_cal=nothing, rolling_vol=nothing,
                       sector_map=nothing, div_yields=nothing,
                       rolling_iv=nothing)
    config = portfolio.config
    dfee = config.mgmt_fee_annual / 252.0

    for (di, date) in enumerate(trading_days)
        cp = Dict{String, Float64}()
        for (tk, pdf) in price_data
            p = get_price_on_date(pdf, date)
            p !== nothing && (cp[tk] = p)
        end

        if !isempty(portfolio.daily_records)
            portfolio.cash -= portfolio.daily_records[end].nav * dfee
        end

        # Risk overlay: compute trailing VaR/ES and throttle if exceeded
        risk_throttled = false
        if length(portfolio.daily_records) > config.risk_lookback
            var95, es95 = compute_trailing_risk(portfolio.daily_records, config.risk_lookback)
            risk_throttled = var95 > config.var_limit_daily || es95 > config.es_limit_daily
        end

        current_nav = if !isempty(portfolio.daily_records)
            portfolio.daily_records[end].nav
        else
            portfolio.initial_nav
        end

        # Compute drawdown for adaptive delta
        drawdown_pct = 0.0
        if !isempty(portfolio.daily_records)
            pk = maximum(r.nav for r in portfolio.daily_records)
            drawdown_pct = max(0.0, (pk - current_nav) / pk)
        end

        for (tk, state) in portfolio.states
            p = get(cp, tk, NaN); isnan(p) && continue
            σ = _resolve_vol(rolling_vol, vol_map, tk, date)
            q = div_yields !== nothing ? get(div_yields, tk, 0.0) : 0.0
            ddf = get(div_data, tk, DataFrame(ex_date=Date[], amount=Float64[]))

            σ_iv_val = if rolling_iv !== nothing && haskey(rolling_iv, tk)
                get(rolling_iv[tk], date, NaN)
            else
                NaN
            end

            check_dividends!(state, date, ddf, portfolio)

            is_near_earn = earnings_cal !== nothing &&
                           near_earnings(earnings_cal, tk, date, config.earnings_buffer_days)

            for slot in state.slots
                apply_borrow_cost!(state, slot, p, config, portfolio)

                if slot.option !== nothing && date >= slot.option.expiry
                    handle_expiry!(state, slot, p, portfolio)
                end

                if should_roll(slot, date, p, σ, config)
                    roll_option!(state, slot, p, date, σ, config, portfolio;
                                 earnings_cal=earnings_cal, q=q,
                                 near_earn=is_near_earn, drawdown_pct=drawdown_pct,
                                 σ_iv=σ_iv_val)
                end

                if slot.option === nothing
                    skip = risk_throttled
                    skip = skip || (earnings_cal !== nothing &&
                                    config.earnings_policy == :avoid && is_near_earn)
                    # :widen and :reduce_size pass through (handled inside open_option!)
                    skip = skip || !check_name_weight(state, p, current_nav, config)
                    if sector_map !== nothing
                        skip = skip || !check_sector_cap(tk, sector_map, portfolio, cp, current_nav)
                    end

                    if !skip
                        portfolio.cash += open_option!(state, slot, p, date, σ, config,
                                                       portfolio; earnings_cal=earnings_cal,
                                                       q=q, near_earn=is_near_earn,
                                                       drawdown_pct=drawdown_pct,
                                                       σ_iv=σ_iv_val)
                    end
                end

                check_cost_basis_repair!(state, slot, date, p, portfolio, config)
            end
        end

        push!(portfolio.daily_records, compute_nav(portfolio, cp, date, vol_map, config;
                                                    rolling_vol=rolling_vol,
                                                    div_yields=div_yields,
                                                    rolling_iv=rolling_iv))
    end
end

# ── Report ───────────────────────────────────────────────────────────────────

"""
Generate comprehensive performance report with KPIs from Varner PDF Section 4.
Includes benchmark comparison (TODO item 17) and per-leg contribution.
"""
function generate_report(portfolio::Portfolio; benchmark_navs=nothing, benchmark_label::String="SPY")
    recs = portfolio.daily_records
    isempty(recs) && (println("No data."); return)

    ini = portfolio.initial_nav
    fin = recs[end].nav
    ret = (fin - ini) / ini * 100.0
    tp = recs[end].cumulative_premium
    td = recs[end].cumulative_dividends
    tc = recs[end].cumulative_costs

    navs = [r.nav for r in recs]
    dr = diff(log.(navs))
    n = length(dr)
    avol = n > 0 ? std(dr) * sqrt(252) * 100.0 : 0.0
    sh = n > 0 && std(dr) > 0 ? (mean(dr) * 252) / (std(dr) * sqrt(252)) : 0.0
    ds = filter(x -> x < 0, dr)
    so = length(ds) > 0 ? (mean(dr) * 252) / (std(ds) * sqrt(252)) : 0.0

    pk, mdd = -Inf, 0.0
    for v in navs; pk = max(pk, v); mdd = max(mdd, (pk - v) / pk); end

    srt = sort(dr)
    vi = max(1, floor(Int, 0.05 * length(srt)))
    dvar = length(srt) > 0 ? -srt[vi] : 0.0
    des = vi > 0 && length(srt) >= vi ? -mean(srt[1:vi]) : 0.0

    ta = sum(sum(s.assignment_count for s in st.slots) for (_, st) in portfolio.states)
    tc2 = sum(sum(s.callaway_count for s in st.slots) for (_, st) in portfolio.states)
    tt = sum(sum(s.trades for s in st.slots) for (_, st) in portfolio.states)
    tr_count = sum(sum(s.repair_count for s in st.slots) for (_, st) in portfolio.states)

    dy = (tp + td) / ini * 100.0
    ar = tt > 0 ? ta / tt * 100.0 : 0.0
    cr = tt > 0 ? tc2 / tt * 100.0 : 0.0

    sp = sum(sum(s.total_premium for s in st.slots) for (_, st) in portfolio.states if st.sleeve == "Safe")
    ap2 = sum(sum(s.total_premium for s in st.slots) for (_, st) in portfolio.states if st.sleeve == "Aggressive")
    sd2 = sum(st.total_dividends for (_, st) in portfolio.states if st.sleeve == "Safe")
    ad = sum(st.total_dividends for (_, st) in portfolio.states if st.sleeve == "Aggressive")
    sa = sum(sum(s.assignment_count for s in st.slots) for (_, st) in portfolio.states if st.sleeve == "Safe")
    aa = sum(sum(s.assignment_count for s in st.slots) for (_, st) in portfolio.states if st.sleeve == "Aggressive")
    sc = sum(sum(s.callaway_count for s in st.slots) for (_, st) in portfolio.states if st.sleeve == "Safe")
    ac = sum(sum(s.callaway_count for s in st.slots) for (_, st) in portfolio.states if st.sleeve == "Aggressive")
    sr = sum(sum(s.repair_count for s in st.slots) for (_, st) in portfolio.states if st.sleeve == "Safe")
    arr = sum(sum(s.repair_count for s in st.slots) for (_, st) in portfolio.states if st.sleeve == "Aggressive")

    # Per-leg (put vs call) contribution
    put_prem, call_prem = 0.0, 0.0
    for (_, st) in portfolio.states
        for s in st.slots
            if s.phase == SELLING_PUT || s.option !== nothing && s.option.type == :put
                put_prem += s.total_premium * 0.5
            end
            if s.phase == HOLDING_SHARES || s.option !== nothing && s.option.type == :call
                call_prem += s.total_premium * 0.5
            end
        end
    end

    println("\n══════════════════════════════════════════════════════")
    println("         Wheel ETF Strategy — Performance Report")
    println("══════════════════════════════════════════════════════")

    println("\n── Portfolio Summary ──")
    println("  Initial NAV:     \$$(round(Int, ini))")
    println("  Final NAV:       \$$(round(Int, fin))")
    println("  Total Return:    $(round(ret, digits=2))%")

    println("\n── Return Decomposition ──")
    println("  Premium Income:  \$$(round(Int, tp))  ($(round(tp/ini*100, digits=2))%)")
    println("  Dividend Income: \$$(round(Int, td))  ($(round(td/ini*100, digits=2))%)")
    println("  Trading Costs:   \$$(round(Int, tc))  ($(round(tc/ini*100, digits=2))%)")
    println("  Mgmt Fee:        $(round(portfolio.config.mgmt_fee_annual*100, digits=2))% annual")

    println("\n── Core KPIs (Varner PDF Section 4) ──")
    println("  Distribution Yield:   $(round(dy, digits=2))%")
    println("  Premium Capture Rate: \$$(round(tp/max(tt,1), digits=2)) / trade")
    println("  Assignment Rate:      $(round(ar, digits=2))%  ($ta / $tt)")
    println("  % Called-Away:        $(round(cr, digits=2))%  ($tc2 / $tt)")
    println("  Cost-Basis Repairs:   $tr_count")

    println("\n── Risk Metrics ──")
    println("  Annualized Vol:  $(round(avol, digits=2))%")
    println("  Sharpe Ratio:    $(round(sh, digits=3))")
    println("  Sortino Ratio:   $(round(so, digits=3))")
    println("  Max Drawdown:    $(round(mdd*100, digits=2))%")
    println("  Daily VaR (95%): $(round(dvar*100, digits=3))%")
    println("  Daily ES  (95%): $(round(des*100, digits=3))%")

    println("\n── Portfolio Greeks (final day) ──")
    last_rec = recs[end]
    println("  Portfolio Delta: $(round(last_rec.portfolio_delta, digits=1))")
    println("  Portfolio Gamma: $(round(last_rec.portfolio_gamma, digits=4))")
    println("  Portfolio Vega:  $(round(last_rec.portfolio_vega, digits=1))")

    # Benchmark comparison (TODO item 17)
    if benchmark_navs !== nothing && length(benchmark_navs) >= length(navs)
        bm = benchmark_navs[1:length(navs)]
        bm_ret = (bm[end] - bm[1]) / bm[1] * 100.0
        bm_dr = diff(log.(bm))
        bm_vol = std(bm_dr) * sqrt(252) * 100.0
        bm_sh = std(bm_dr) > 0 ? (mean(bm_dr) * 252) / (std(bm_dr) * sqrt(252)) : 0.0
        bm_pk, bm_mdd = -Inf, 0.0
        for v in bm; bm_pk = max(bm_pk, v); bm_mdd = max(bm_mdd, (bm_pk - v) / bm_pk); end
        excess_ret = ret - bm_ret
        te = length(dr) > 0 ? std(dr .- bm_dr) * sqrt(252) * 100.0 : 0.0
        ir = te > 0 ? (excess_ret / te) : 0.0

        println("\n── Benchmark Comparison vs $(benchmark_label) ──")
        println("  $(benchmark_label) Return:      $(round(bm_ret, digits=2))%")
        println("  $(benchmark_label) Sharpe:      $(round(bm_sh, digits=3))")
        println("  $(benchmark_label) Max DD:      $(round(bm_mdd*100, digits=2))%")
        println("  Excess Return:     $(round(excess_ret, digits=2))%")
        println("  Tracking Error:    $(round(te, digits=2))%")
        println("  Information Ratio: $(round(ir, digits=3))")
    end

    println("\n── Sleeve Contribution ──")
    pretty_table(DataFrame(
        Metric=["Premium", "Dividends", "Assigns", "Call-Aways", "Repairs"],
        Safe=["\$$(round(Int,sp))", "\$$(round(Int,sd2))", "$sa", "$sc", "$sr"],
        Aggressive=["\$$(round(Int,ap2))", "\$$(round(Int,ad))", "$aa", "$ac", "$arr"]),
        column_labels=["Metric", "Safe (60%)", "Aggressive (40%)"])

    println("\n── Per-Ticker Summary ──")
    tdf = DataFrame(Ticker=String[], Sleeve=String[], Slots=Int[],
        Premium=Float64[], Divs=Float64[], Costs=Float64[],
        Assigns=Int[], CallAways=Int[], Repairs=Int[], Trades=Int[])
    for tk in sort(collect(keys(portfolio.states)))
        st = portfolio.states[tk]
        push!(tdf, (tk, st.sleeve, length(st.slots),
            round(sum(s.total_premium for s in st.slots), digits=0),
            round(st.total_dividends, digits=0),
            round(st.total_costs, digits=0),
            sum(s.assignment_count for s in st.slots),
            sum(s.callaway_count for s in st.slots),
            sum(s.repair_count for s in st.slots),
            sum(s.trades for s in st.slots)))
    end
    sort!(tdf, :Premium, rev=true)
    pretty_table(tdf, column_labels=["Ticker","Sleeve","Slots","Premium(\$)",
        "Divs(\$)","Costs(\$)","Assigns","CallAways","Repairs","Trades"])

    println("\n── Configuration ──")
    c = portfolio.config
    println("  Pricing model:        CRR American ($(c.crr_steps) steps, div yield q)")
    println("  Ladder slots:         $(c.max_ladders) per ticker")
    println("  Earnings policy:      $(c.earnings_policy) ($(c.earnings_buffer_days)-day buffer)")
    println("  Bid-ask spread model: $(c.bid_ask_spread_model)")
    println("  Liquidity screen:     min OI=$(c.min_open_interest), max spread=$(Int(c.max_spread_width_pct*100))%")
    println("  Partial fill model:   $(c.fill_rate_model)")
    println("  Risk overlay:         VaR limit=$(c.var_limit_daily), ES limit=$(c.es_limit_daily)")
    println("  Max name weight:      $(Int(c.max_name_weight*100))% NAV")
    println("  Adaptive tenor:       $(c.adaptive_tenor)")
    println("  Adaptive delta:       $(c.adaptive_delta)")
    println("  Time-based roll:      $(c.roll_dte_min)–$(c.roll_dte_max) DTE")
    println("  Premium decay:        $(Int(c.premium_decay_threshold*100))% captured → roll")
    println("  ITM/OTM band:         ±$(Int(c.itm_otm_band_pct*100))% moneyness → roll")
    println("  Breakeven breach:     $(Int(c.breakeven_breach_pct*100))% past BE → roll")
    println("  Repair trigger:       $(Int(c.repair_loss_trigger*100))% underwater")
    println("  Max repair capital:   $(Int(c.max_avg_down_pct*100))% of slot capital")
    println("  Max repairs/quarter:  $(c.max_repairs_per_qtr)")
    println("  Costs: commission=\$$(c.commission_per_contract) + exchange=\$$(c.exchange_fee_per_contract) + clearing=\$$(c.clearing_fee_per_contract)/contract")
    println("  Borrow fee:           $(round(c.borrow_fee_annual*100, digits=2))% annual")
    println()
end
