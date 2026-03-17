"""
Operations & Costs — Varner PDF Section 5

Cost model: commissions, exchange/clearing fees, borrow fees, slippage.
Liquidity screening and partial fill model.

Reference:
  "Fees. Management fee benchmarked to option-income active ETFs (0.60%-1.00%):
   model commissions, exchange/clearing fees, and half-spread slippage."
  "Data & systems. Historical options data with Greeks; OMS/EMS; OCC event handling;
   corporate actions calendar."
"""

# ── Bid-Ask Spread Model ──────────────────────────────────────────────────────

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

# ── Liquidity Screen ──────────────────────────────────────────────────────────

"""
    check_liquidity(premium, σ, config) -> (passes::Bool, fill_rate::Float64)

Liquidity screening and fill rate estimation.
Checks if estimated spread width exceeds max_spread_width_pct.
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

# ── Crowding-Adjusted Slippage ────────────────────────────────────────────────

"""
    crowding_multiplier(adv, notional; crowd_threshold=0.02) -> Float64

Slippage multiplier based on trade size relative to ADV.
Per Varner PDF Section 7A: "slippage calibrated to spreads and crowding."
"""
function crowding_multiplier(adv::Float64, notional::Float64;
                              crowd_threshold::Float64=0.02)::Float64
    adv <= 0.0 && return 1.0
    participation = notional / adv
    if participation > crowd_threshold
        return 1.0 + (participation / crowd_threshold - 1.0) * 0.5
    end
    return 1.0
end

# ── Trade Cost Application ────────────────────────────────────────────────────

"""
    apply_trade_costs!(state, slot, premium, σ, config, portfolio; adv)

Deduct trading costs: commission + exchange + clearing + slippage.
Slippage includes crowding adjustment when ADV is provided.
"""
function apply_trade_costs!(state, slot, premium::Float64,
                            σ::Float64, config::WheelConfig, portfolio;
                            adv::Float64=0.0)
    nc = slot.num_contracts
    commission = config.commission_per_contract * nc
    exchange_fee = config.exchange_fee_per_contract * nc
    clearing_fee = config.clearing_fee_per_contract * nc
    slippage = if config.bid_ask_spread_model
        per_share_prem = abs(premium) / max(100 * nc, 1)
        base_slip = estimate_half_spread(per_share_prem, σ) * 100 * nc
        crowd_mult = adv > 0.0 ? crowding_multiplier(adv, abs(premium)) : 1.0
        base_slip * crowd_mult
    else
        abs(premium) * config.slippage_pct
    end
    cost = commission + exchange_fee + clearing_fee + slippage
    state.total_costs += cost
    portfolio.cash -= cost
end

"""
    apply_borrow_cost!(state, slot, price, config, portfolio)

Daily borrow cost for held shares.
Per Varner PDF Section 7A: "borrow fees for hard-to-borrow events."
"""
function apply_borrow_cost!(state, slot, price::Float64,
                            config::WheelConfig, portfolio)
    slot.shares_held <= 0 && return
    daily_borrow = config.borrow_fee_annual / 252.0 * slot.shares_held * price
    state.total_costs += daily_borrow
    portfolio.cash -= daily_borrow
end
