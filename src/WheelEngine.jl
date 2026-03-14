"""
Wheel strategy backtest engine (per Varner PDF requirements).

Per-ticker state machine: SELLING_PUT → (assigned) → HOLDING_SHARES → (called away) → SELLING_PUT
Two blocks per name: Block A (buy-and-hold) + Block B (wheel cycle).

Roll rules (Section 3):
  - Time-based: roll OTM options at 3–5 DTE
  - Trigger-based:
    (a) Premium decay — roll when ≥ threshold of original premium has been captured
    (b) %ITM/OTM bands — roll when option is deep ITM or deep OTM beyond band %
    (c) Breakeven breach — roll when price breaches short option breakeven

Cost-basis repair (Sections 1, 2, 6):
  - Average down when shares are underwater beyond trigger %
  - Max additional capital and quarterly limits
"""

@enum WheelPhase SELLING_PUT HOLDING_SHARES

mutable struct OptionPosition
    type::Symbol        # :put or :call
    strike::Float64
    expiry::Date
    premium::Float64    # per-share premium collected at open
    open_date::Date
end

mutable struct TickerState
    ticker::String
    sleeve::String
    phase::WheelPhase
    shares_held::Int
    cost_basis::Float64
    option::Union{Nothing, OptionPosition}
    num_contracts::Int
    block_a_shares::Int
    block_a_cost::Float64
    block_b_capital::Float64
    total_premium::Float64
    total_dividends::Float64
    total_realized_pnl::Float64
    total_costs::Float64
    assignment_count::Int
    callaway_count::Int
    trades::Int
    repair_count::Int
    repairs_this_quarter::Int
    last_repair_quarter::Int
end

struct WheelConfig
    risk_free_rate::Float64
    tenor_days::Vector{Int}
    delta_put_safe::Tuple{Float64, Float64}
    delta_call_safe::Tuple{Float64, Float64}
    delta_put_aggr::Tuple{Float64, Float64}
    delta_call_aggr::Tuple{Float64, Float64}
    # time-based roll
    roll_dte_min::Int
    roll_dte_max::Int
    # trigger-based roll
    premium_decay_threshold::Float64   # e.g. 0.80 → roll when 80% of premium captured
    itm_otm_band_pct::Float64         # e.g. 0.05 → roll when 5% deep ITM/OTM
    breakeven_breach_pct::Float64      # e.g. 0.02 → roll when price breaches BE by 2%
    # cost-basis repair
    repair_loss_trigger::Float64       # e.g. 0.10 → repair when 10% underwater
    max_avg_down_pct::Float64          # max additional capital as % of block_b_capital
    max_repairs_per_qtr::Int
    # costs
    commission_per_contract::Float64
    slippage_pct::Float64
    mgmt_fee_annual::Float64
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
end

mutable struct Portfolio
    cash::Float64
    initial_nav::Float64
    states::Dict{String, TickerState}
    daily_records::Vector{DailyRecord}
    config::WheelConfig
end

function default_config()::WheelConfig
    return WheelConfig(
        0.045,                         # risk-free rate
        [7, 14, 30],                   # tenor ladder
        (0.20, 0.30), (0.25, 0.35),   # safe put/call delta ranges
        (0.25, 0.35), (0.30, 0.40),   # aggressive put/call delta ranges
        3, 5,                          # roll DTE window
        0.80,                          # premium decay: roll when 80% captured
        0.05,                          # ITM/OTM band: 5% deep triggers roll
        0.02,                          # breakeven breach: 2% past BE triggers roll
        0.10,                          # repair trigger: 10% underwater
        0.25,                          # max avg-down: 25% of block_b_capital
        2,                             # max 2 repairs per quarter
        0.65,                          # commission per contract
        0.02,                          # slippage as % of premium
        0.0068,                        # management fee 0.68% annual
    )
end

# ── Helpers ──

function select_tenor(config::WheelConfig, date::Date)::Int
    idx = mod(Dates.week(date), length(config.tenor_days)) + 1
    return config.tenor_days[idx]
end

function find_expiry(date::Date, tenor_days::Int)::Date
    target = date + Day(tenor_days)
    dow = dayofweek(target)
    return dow <= 5 ? target + Day(5 - dow) : target + Day(12 - dow)
end

function get_target_delta(state::TickerState, config::WheelConfig, otype::Symbol)::Float64
    r = if state.sleeve == "Safe"
        otype == :put ? config.delta_put_safe : config.delta_call_safe
    else
        otype == :put ? config.delta_put_aggr : config.delta_call_aggr
    end
    return (r[1] + r[2]) / 2.0
end

function apply_trade_costs!(state::TickerState, premium::Float64,
                            config::WheelConfig, portfolio::Portfolio)
    cost = config.commission_per_contract * state.num_contracts +
           abs(premium) * config.slippage_pct
    state.total_costs += cost
    portfolio.cash -= cost
end

# ── Option Operations ──

function open_option!(state::TickerState, price::Float64, date::Date,
                      σ::Float64, config::WheelConfig, portfolio::Portfolio)::Float64
    otype = state.phase == SELLING_PUT ? :put : :call
    delta = get_target_delta(state, config, otype)
    expiry = find_expiry(date, select_tenor(config, date))
    T = max(Dates.value(expiry - date) / 365.0, 1.0/365.0)

    K = strike_from_delta(price, T, config.risk_free_rate, σ, delta, otype)
    prem = max(option_price(price, K, T, config.risk_free_rate, σ, otype), 0.01)

    state.option = OptionPosition(otype, K, expiry, prem, date)
    state.trades += 1
    total = prem * 100 * state.num_contracts
    state.total_premium += total
    apply_trade_costs!(state, total, config, portfolio)
    return total
end

function close_option_mtm(state::TickerState, price::Float64, date::Date,
                          σ::Float64, config::WheelConfig)::Float64
    opt = state.option
    opt === nothing && return 0.0
    T = max(Dates.value(opt.expiry - date) / 365.0, 0.0)
    return option_price(price, opt.strike, T, config.risk_free_rate, σ, opt.type) *
           100.0 * state.num_contracts
end

function handle_expiry!(state::TickerState, price::Float64, portfolio::Portfolio)
    opt = state.option
    opt === nothing && return
    lot = 100 * state.num_contracts
    if opt.type == :put && price < opt.strike
        portfolio.cash -= opt.strike * lot
        state.shares_held = lot
        state.cost_basis = opt.strike
        state.phase = HOLDING_SHARES
        state.assignment_count += 1
    elseif opt.type == :call && price > opt.strike
        portfolio.cash += opt.strike * lot
        state.total_realized_pnl += (opt.strike - state.cost_basis) * lot
        state.shares_held = 0
        state.cost_basis = 0.0
        state.phase = SELLING_PUT
        state.callaway_count += 1
    end
    state.option = nothing
end

# ── Roll Logic (time-based + trigger-based) ──

function should_roll(state::TickerState, date::Date, price::Float64,
                     σ::Float64, config::WheelConfig)::Bool
    opt = state.option
    opt === nothing && return false
    dte = Dates.value(opt.expiry - date)

    # Cooldown: don't roll within 3 days of opening
    Dates.value(date - opt.open_date) < 3 && return false

    # (1) Time-based: roll OTM options at 3–5 DTE
    if config.roll_dte_min <= dte <= config.roll_dte_max
        is_otm = (opt.type == :put && price > opt.strike) ||
                 (opt.type == :call && price < opt.strike)
        is_otm && return true
    end

    # (2) Premium decay: roll when ≥ threshold of original premium captured
    if dte > 0
        T_remain = max(dte / 365.0, 1.0/365.0)
        current_val = option_price(price, opt.strike, T_remain, config.risk_free_rate, σ, opt.type)
        if opt.premium > 0.0 && current_val / opt.premium < (1.0 - config.premium_decay_threshold)
            return true
        end
    end

    # (3) %ITM/OTM bands: roll when option is deep ITM or deep OTM
    moneyness = (price - opt.strike) / opt.strike
    if opt.type == :put
        # deep ITM put: price far below strike (moneyness << 0)
        moneyness < -config.itm_otm_band_pct && return true
        # deep OTM put: price far above strike
        moneyness > config.itm_otm_band_pct && return true
    else
        # deep ITM call: price far above strike
        moneyness > config.itm_otm_band_pct && return true
        # deep OTM call: price far below strike
        moneyness < -config.itm_otm_band_pct && return true
    end

    # (4) Breakeven breach: seller's breakeven
    if opt.type == :put
        be = opt.strike - opt.premium
        price < be * (1.0 - config.breakeven_breach_pct) && return true
    else
        be = opt.strike + opt.premium
        price > be * (1.0 + config.breakeven_breach_pct) && return true
    end

    return false
end

function roll_option!(state::TickerState, price::Float64, date::Date,
                      σ::Float64, config::WheelConfig, portfolio::Portfolio)::Float64
    cc = close_option_mtm(state, price, date, σ, config)
    portfolio.cash -= cc
    state.total_premium -= cc
    apply_trade_costs!(state, cc, config, portfolio)
    state.option = nothing
    np = open_option!(state, price, date, σ, config, portfolio)
    portfolio.cash += np
    return np - cc
end

# ── Cost-Basis Repair ──

function check_cost_basis_repair!(state::TickerState, date::Date, price::Float64,
                                  portfolio::Portfolio, config::WheelConfig)
    state.phase != HOLDING_SHARES && return
    state.cost_basis <= 0 && return

    current_qtr = div(Dates.month(date) - 1, 3) + 1
    if current_qtr != state.last_repair_quarter
        state.repairs_this_quarter = 0
        state.last_repair_quarter = current_qtr
    end
    state.repairs_this_quarter >= config.max_repairs_per_qtr && return

    loss_pct = (state.cost_basis - price) / state.cost_basis
    loss_pct < config.repair_loss_trigger && return

    max_spend = config.max_avg_down_pct * state.block_b_capital
    add_shares = min(state.shares_held, floor(Int, max_spend / price))
    add_shares <= 0 && return
    add_shares > floor(Int, portfolio.cash * 0.05 / price) && return

    spend = add_shares * price
    old_total = state.shares_held * state.cost_basis
    state.shares_held += add_shares
    state.cost_basis = (old_total + spend) / state.shares_held
    portfolio.cash -= spend
    state.repair_count += 1
    state.repairs_this_quarter += 1
end

# ── Dividends ──

function check_dividends!(state::TickerState, date::Date,
                          div_data::DataFrame, portfolio::Portfolio)
    idx = findfirst(==(date), div_data.ex_date)
    idx === nothing && return
    d = div_data.amount[idx] * (state.block_a_shares + state.shares_held)
    d <= 0 && return
    state.total_dividends += d
    portfolio.cash += d
end

# ── Portfolio Initialization ──

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
        nc = max(1, floor(Int, half / (price * 100.0)))
        states[ticker] = TickerState(
            ticker, sleeves[i], SELLING_PUT, 0, 0.0, nothing,
            nc, ba, price, half,
            0.0, 0.0, 0.0, 0.0, 0, 0, 0,
            0, 0, 0)  # repair_count, repairs_this_quarter, last_repair_quarter
    end
    return Portfolio(initial_nav - total_cost, initial_nav, states, DailyRecord[], config)
end

# ── NAV Computation ──

function compute_nav(portfolio::Portfolio, prices::Dict{String, Float64},
                     date::Date, vol_map::Dict{String, Float64},
                     config::WheelConfig)::DailyRecord
    sv, omtm, bav, tp, td, tc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for (tk, s) in portfolio.states
        p = get(prices, tk, NaN)
        isnan(p) && continue
        σ = get(vol_map, tk, 0.25)
        bav += s.block_a_shares * p
        sv += s.shares_held * p
        if s.option !== nothing
            T = max(Dates.value(s.option.expiry - date) / 365.0, 0.0)
            omtm -= option_price(p, s.option.strike, T, config.risk_free_rate, σ,
                                 s.option.type) * 100.0 * s.num_contracts
        end
        tp += s.total_premium; td += s.total_dividends; tc += s.total_costs
    end
    nav = portfolio.cash + sv + bav + omtm
    return DailyRecord(date, nav, portfolio.cash, sv, omtm, bav, tp, td, tc)
end

# ── Main Simulation Loop ──

function run_backtest!(portfolio::Portfolio, price_data::Dict{String, DataFrame},
                       div_data::Dict{String, DataFrame}, vol_map::Dict{String, Float64},
                       trading_days::Vector{Date})
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

        for (tk, s) in portfolio.states
            p = get(cp, tk, NaN); isnan(p) && continue
            σ = get(vol_map, tk, 0.25)
            ddf = get(div_data, tk, DataFrame(ex_date=Date[], amount=Float64[]))

            check_dividends!(s, date, ddf, portfolio)

            if s.option !== nothing && date >= s.option.expiry
                handle_expiry!(s, p, portfolio)
            end

            if should_roll(s, date, p, σ, config)
                roll_option!(s, p, date, σ, config, portfolio)
            end

            if s.option === nothing
                portfolio.cash += open_option!(s, p, date, σ, config, portfolio)
            end

            check_cost_basis_repair!(s, date, p, portfolio, config)
        end

        push!(portfolio.daily_records, compute_nav(portfolio, cp, date, vol_map, config))
        if di % 50 == 0 || di == length(trading_days)
            println("  Day $di / $(length(trading_days)) | $date | NAV = \$$(round(Int, portfolio.daily_records[end].nav))")
        end
    end
end

# ── Report ──

function generate_report(portfolio::Portfolio)
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
    avol = std(dr) * sqrt(252) * 100.0
    sh = n > 0 ? (mean(dr) * 252) / (std(dr) * sqrt(252)) : 0.0
    ds = filter(x -> x < 0, dr)
    so = length(ds) > 0 ? (mean(dr) * 252) / (std(ds) * sqrt(252)) : 0.0

    pk, mdd = -Inf, 0.0
    for v in navs; pk = max(pk, v); mdd = max(mdd, (pk - v) / pk); end

    srt = sort(dr)
    vi = max(1, floor(Int, 0.05 * length(srt)))
    dvar = -srt[vi]
    des = -mean(srt[1:vi])

    ta = sum(s.assignment_count for (_, s) in portfolio.states)
    tc2 = sum(s.callaway_count for (_, s) in portfolio.states)
    tt = sum(s.trades for (_, s) in portfolio.states)
    tr_count = sum(s.repair_count for (_, s) in portfolio.states)

    dy = (tp + td) / ini * 100.0
    ar = tt > 0 ? ta / tt * 100.0 : 0.0
    cr = tt > 0 ? tc2 / tt * 100.0 : 0.0

    # sleeve-level
    sp = sum(s.total_premium for (_, s) in portfolio.states if s.sleeve == "Safe")
    ap = sum(s.total_premium for (_, s) in portfolio.states if s.sleeve == "Aggressive")
    sd2 = sum(s.total_dividends for (_, s) in portfolio.states if s.sleeve == "Safe")
    ad = sum(s.total_dividends for (_, s) in portfolio.states if s.sleeve == "Aggressive")
    sa = sum(s.assignment_count for (_, s) in portfolio.states if s.sleeve == "Safe")
    aa = sum(s.assignment_count for (_, s) in portfolio.states if s.sleeve == "Aggressive")
    sc = sum(s.callaway_count for (_, s) in portfolio.states if s.sleeve == "Safe")
    ac = sum(s.callaway_count for (_, s) in portfolio.states if s.sleeve == "Aggressive")
    sr = sum(s.repair_count for (_, s) in portfolio.states if s.sleeve == "Safe")
    arr = sum(s.repair_count for (_, s) in portfolio.states if s.sleeve == "Aggressive")

    println("\n── Portfolio Summary ──")
    println("  Initial NAV:     \$$(round(Int, ini))")
    println("  Final NAV:       \$$(round(Int, fin))")
    println("  Total Return:    $(round(ret, digits=2))%")

    println("\n── Return Decomposition ──")
    println("  Premium Income:  \$$(round(Int, tp))  ($(round(tp/ini*100, digits=2))%)")
    println("  Dividend Income: \$$(round(Int, td))  ($(round(td/ini*100, digits=2))%)")
    println("  Trading Costs:   \$$(round(Int, tc))  ($(round(tc/ini*100, digits=2))%)")
    println("  Mgmt Fee:        $(round(portfolio.config.mgmt_fee_annual*100, digits=2))% annual")

    println("\n── Core KPIs ──")
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

    println("\n── Sleeve Contribution ──")
    pretty_table(DataFrame(
        Metric=["Premium", "Dividends", "Assigns", "Call-Aways", "Repairs"],
        Safe=["\$$(round(Int,sp))", "\$$(round(Int,sd2))", "$sa", "$sc", "$sr"],
        Aggressive=["\$$(round(Int,ap))", "\$$(round(Int,ad))", "$aa", "$ac", "$arr"]),
        column_labels=["Metric", "Safe (60%)", "Aggressive (40%)"])

    println("\n── Per-Ticker Summary ──")
    tdf = DataFrame(Ticker=String[], Sleeve=String[], Phase=String[],
        Premium=Float64[], Divs=Float64[], Costs=Float64[],
        Assigns=Int[], CallAways=Int[], Repairs=Int[], Trades=Int[])
    for tk in sort(collect(keys(portfolio.states)))
        s = portfolio.states[tk]
        push!(tdf, (tk, s.sleeve, string(s.phase), round(s.total_premium, digits=0),
            round(s.total_dividends, digits=0), round(s.total_costs, digits=0),
            s.assignment_count, s.callaway_count, s.repair_count, s.trades))
    end
    sort!(tdf, :Premium, rev=true)
    pretty_table(tdf, column_labels=["Ticker","Sleeve","Phase","Premium(\$)",
        "Divs(\$)","Costs(\$)","Assigns","CallAways","Repairs","Trades"])

    println("\n── Roll / Trigger Config ──")
    c = portfolio.config
    println("  Time-based roll:      $(c.roll_dte_min)–$(c.roll_dte_max) DTE")
    println("  Premium decay:        $(Int(c.premium_decay_threshold*100))% captured → roll")
    println("  ITM/OTM band:         ±$(Int(c.itm_otm_band_pct*100))% moneyness → roll")
    println("  Breakeven breach:     $(Int(c.breakeven_breach_pct*100))% past BE → roll")
    println("  Repair trigger:       $(Int(c.repair_loss_trigger*100))% underwater")
    println("  Max repair capital:   $(Int(c.max_avg_down_pct*100))% of block B")
    println("  Max repairs/quarter:  $(c.max_repairs_per_qtr)")
    println()
end
