"""
Wheel Strategy Backtest Engine — Varner PDF Section 7A (Testing Plan A)

Core state machine + daily simulation loop.

Architecture: per-ticker state machine with ladder slots.
  SELLING_PUT → (assigned) → HOLDING_SHARES → (called away) → SELLING_PUT

Each ticker maintains:
  Block A — buy-and-hold for dividends and capital appreciation (no options).
  Block B — split across 1–3 ladder slots, each running an independent Wheel cycle.

Dependencies (loaded before this file):
  PortfolioConstruction.jl — WheelConfig, select_tenor, find_expiry, get_target_delta
  OperationsCosts.jl       — apply_trade_costs!, apply_borrow_cost!, check_liquidity
  RiskCompliance.jl        — compute_trailing_risk, check_name_weight, check_sector_cap
  OptionPricing.jl         — option_price, option_greeks, strike_from_delta
  Compute.jl               — trailing_dividend_yield
"""

# ── Core Structs ──────────────────────────────────────────────────────────────

@enum WheelPhase SELLING_PUT HOLDING_SHARES

mutable struct OptionPosition
    type::Symbol        # :put or :call
    strike::Float64
    expiry::Date
    premium::Float64    # per-share premium collected at open
    open_date::Date
end

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

# ── Option Operations ────────────────────────────────────────────────────────

"""Open a new option. Uses σ for delta targeting and σ_iv for pricing."""
function open_option!(state::TickerState, slot::LadderSlot, price::Float64,
                      date::Date, σ::Float64, config::WheelConfig,
                      portfolio::Portfolio;
                      earnings_cal=nothing, q::Float64=0.0,
                      near_earn::Bool=false, drawdown_pct::Float64=0.0,
                      σ_iv::Float64=NaN, adv::Float64=0.0)::Float64
    pricing_vol = isnan(σ_iv) ? σ : σ_iv

    otype = slot.phase == SELLING_PUT ? :put : :call
    delta = get_target_delta(state, config, otype; σ=σ, near_earn=near_earn,
                              drawdown_pct=drawdown_pct)

    if earnings_cal !== nothing && near_earn && config.earnings_policy == :widen
        delta = max(0.10, delta - config.earnings_wider_delta)
    end

    tenor = select_tenor(config, date; σ=σ)
    expiry = find_expiry(date, tenor)
    T = max(Dates.value(expiry - date) / 365.0, 1.0/365.0)

    K = strike_from_delta(price, T, config.risk_free_rate, σ, delta, otype; q=q)
    prem = max(option_price(price, K, T, config.risk_free_rate, pricing_vol, otype;
                            steps=config.crr_steps, q=q), 0.01)

    liq_ok, fill_rate = check_liquidity(prem, σ, config)
    !liq_ok && return 0.0

    effective_contracts = if config.fill_rate_model && fill_rate < 1.0
        max(1, round(Int, slot.num_contracts * fill_rate))
    else
        slot.num_contracts
    end

    if earnings_cal !== nothing && near_earn && config.earnings_policy == :reduce_size
        effective_contracts = max(1, round(Int, effective_contracts * config.earnings_size_reduction))
    end

    slot.option = OptionPosition(otype, K, expiry, prem, date)
    slot.trades += 1
    total = prem * 100 * effective_contracts
    slot.total_premium += total
    apply_trade_costs!(state, slot, total, σ, config, portfolio; adv=adv)
    return total
end

"""Mark-to-market value of slot's open option using CRR American pricing."""
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

# ── Roll Logic ────────────────────────────────────────────────────────────────

"""Four roll triggers per Varner PDF Section 3."""
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

"""Close current option and open a new one (roll forward)."""
function roll_option!(state::TickerState, slot::LadderSlot, price::Float64,
                      date::Date, σ::Float64, config::WheelConfig,
                      portfolio::Portfolio; earnings_cal=nothing, q::Float64=0.0,
                      near_earn::Bool=false, drawdown_pct::Float64=0.0,
                      σ_iv::Float64=NaN, adv::Float64=0.0)::Float64
    cc = close_option_mtm(slot, price, date, σ, config; q=q, σ_iv=σ_iv)
    portfolio.cash -= cc
    slot.total_premium -= cc
    apply_trade_costs!(state, slot, cc, σ, config, portfolio; adv=adv)
    slot.option = nothing
    np = open_option!(state, slot, price, date, σ, config, portfolio;
                      earnings_cal=earnings_cal, q=q, near_earn=near_earn,
                      drawdown_pct=drawdown_pct, σ_iv=σ_iv, adv=adv)
    portfolio.cash += np
    return np - cc
end

# ── Cost-Basis Repair ─────────────────────────────────────────────────────────

"""Average down when shares are underwater beyond trigger %."""
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

# ── Dividends ─────────────────────────────────────────────────────────────────

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

# ── Portfolio Initialization ──────────────────────────────────────────────────

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

# ── NAV Computation ───────────────────────────────────────────────────────────

"""Compute daily NAV and portfolio Greeks."""
function compute_nav(portfolio::Portfolio, prices::Dict{String, Float64},
                     date::Date, vol_map::Dict{String, Float64},
                     config::WheelConfig;
                     rolling_vol=nothing, div_yields=nothing,
                     div_data=nothing, rolling_iv=nothing)::DailyRecord
    sv, omtm, bav, tp, td, tc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    p_delta, p_gamma, p_vega = 0.0, 0.0, 0.0

    for (tk, state) in portfolio.states
        p = get(prices, tk, NaN)
        isnan(p) && continue
        σ = _resolve_vol(rolling_vol, vol_map, tk, date)
        q = if div_data !== nothing && haskey(div_data, tk)
            trailing_dividend_yield(div_data[tk], p, date)
        elseif div_yields !== nothing
            get(div_yields, tk, 0.0)
        else
            0.0
        end
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
                    n = 100.0 * slot.num_contracts
                    p_delta += -1.0 * g.delta * n
                    p_gamma += -1.0 * g.gamma * n
                    p_vega  += -1.0 * g.vega  * n
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

function _resolve_vol(rolling_vol, vol_map::Dict{String, Float64},
                      ticker::String, date::Date)::Float64
    if rolling_vol !== nothing && haskey(rolling_vol, ticker)
        rv = rolling_vol[ticker]
        haskey(rv, date) && return rv[date]
    end
    return get(vol_map, ticker, 0.25)
end

# ── Main Simulation Loop ─────────────────────────────────────────────────────

"""
    run_backtest!(portfolio, price_data, div_data, vol_map, trading_days; ...)

Day-by-day Wheel strategy simulation with risk overlays, position limits,
and IV calibration.
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

        drawdown_pct = 0.0
        if !isempty(portfolio.daily_records)
            pk = maximum(r.nav for r in portfolio.daily_records)
            drawdown_pct = max(0.0, (pk - current_nav) / pk)
        end

        for (tk, state) in portfolio.states
            p = get(cp, tk, NaN); isnan(p) && continue
            σ = _resolve_vol(rolling_vol, vol_map, tk, date)
            ddf = get(div_data, tk, DataFrame(ex_date=Date[], amount=Float64[]))
            q = trailing_dividend_yield(ddf, p, date)

            σ_iv_val = if rolling_iv !== nothing && haskey(rolling_iv, tk)
                get(rolling_iv[tk], date, NaN)
            else
                NaN
            end

            pdf_tk = get(price_data, tk, nothing)
            adv_val = 0.0
            if pdf_tk !== nothing && hasproperty(pdf_tk, :volume)
                idx = findfirst(==(date), pdf_tk.date)
                if idx !== nothing && idx > 20
                    adv_val = mean(Float64.(pdf_tk.volume[max(1,idx-20):idx])) * p
                end
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
                                 σ_iv=σ_iv_val, adv=adv_val)
                end

                if slot.option === nothing
                    skip = risk_throttled
                    skip = skip || (earnings_cal !== nothing &&
                                    config.earnings_policy == :avoid && is_near_earn)
                    skip = skip || !check_name_weight(state, p, current_nav, config)
                    if sector_map !== nothing
                        skip = skip || !check_sector_cap(tk, sector_map, portfolio, cp, current_nav)
                    end

                    if !skip
                        portfolio.cash += open_option!(state, slot, p, date, σ, config,
                                                       portfolio; earnings_cal=earnings_cal,
                                                       q=q, near_earn=is_near_earn,
                                                       drawdown_pct=drawdown_pct,
                                                       σ_iv=σ_iv_val, adv=adv_val)
                    end
                end

                check_cost_basis_repair!(state, slot, date, p, portfolio, config)
            end
        end

        push!(portfolio.daily_records, compute_nav(portfolio, cp, date, vol_map, config;
                                                    rolling_vol=rolling_vol,
                                                    div_data=div_data,
                                                    rolling_iv=rolling_iv))
    end
end

# ── Report ────────────────────────────────────────────────────────────────────

"""Generate comprehensive performance report with KPIs (Varner PDF Section 4)."""
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

    println("\n--- Wheel ETF Strategy -- Performance Report ---\n")

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
    println("  Time-based roll:      $(c.roll_dte_min)-$(c.roll_dte_max) DTE")
    println("  Premium decay:        $(Int(c.premium_decay_threshold*100))% captured -> roll")
    println("  ITM/OTM band:         +/-$(Int(c.itm_otm_band_pct*100))% moneyness -> roll")
    println("  Breakeven breach:     $(Int(c.breakeven_breach_pct*100))% past BE -> roll")
    println("  Repair trigger:       $(Int(c.repair_loss_trigger*100))% underwater")
    println("  Max repair capital:   $(Int(c.max_avg_down_pct*100))% of slot capital")
    println("  Max repairs/quarter:  $(c.max_repairs_per_qtr)")
    println("  Costs: commission=\$$(c.commission_per_contract) + exchange=\$$(c.exchange_fee_per_contract) + clearing=\$$(c.clearing_fee_per_contract)/contract")
    println("  Borrow fee:           $(round(c.borrow_fee_annual*100, digits=2))% annual")
    println()
end
