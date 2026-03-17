# Wheel Strategy ETF Backtest Engine

A Julia-based backtesting system for the **Wheel option strategy** applied to an S&P 500 universe, following the design specification in the Varner PDF and CHEME-5660 course materials.

## Strategy Overview

Per ticker, maintain two 100-share blocks:
1. **Block A (Dividend/Growth)** — Buy-and-hold for dividends and capital appreciation.
2. **Block B (Option Income)** — Run the Wheel cycle:
   - Sell cash-secured puts → assigned → hold shares → sell covered calls → called away → repeat.

Portfolio split: 60% Safe sleeve (low-vol, high-dividend) / 40% Aggressive sleeve (high-vol tech).

## Key Features

| Feature | Implementation | Reference |
|---------|---------------|-----------|
| **American option pricing** | CRR binomial lattice (N=50) with dividend yield q | CHEME-5660 Week 10 |
| **Greeks** | Δ, Γ, Θ, ν via finite differences on CRR lattice | CHEME-5660 Week 11 |
| **IV surface** | WRDS OptionMetrics historical IV (VRP fallback) | WRDS + Self-designed |
| **HMM regime detection** | N-state discrete HMM via CDF quantile discretization | CHEME-5660 Week 13 |
| **GBM simulation** | Standard, regime-switching, correlated (Cholesky), earnings jump | CHEME-5660 Week 5–6 |
| **Stress testing** | Name-specific gaps, vol spikes, liquidity thinning | Varner PDF §7B |
| **Risk overlays** | VaR/ES throttle, per-name 5% cap, sector caps | Varner PDF §4 |
| **Cost model** | Commissions, exchange/clearing fees, borrow, crowding slippage | Varner PDF §5 |
| **Adaptive controls** | Vol-regime tenor selection, adaptive delta targeting | Varner PDF §3 |
| **Roll rules** | Time-based + 3 trigger-based (premium decay, ITM/OTM band, breakeven breach) | Varner PDF §3 |
| **Cost-basis repair** | Average-down with quarterly caps | Varner PDF §6 |
| **Earnings avoidance** | Skip, widen delta, or reduce size near earnings | Varner PDF §6 |
| **Laddering** | 1–3 concurrent expiries per name | Varner PDF §6 |
| **Parameter sweep** | Grid search over delta, tenor, ladder, Safe/Aggressive ratio | Varner PDF §6 |

## Project Structure

Code is organized to mirror the Varner PDF sections:

```
wheel-project-1/
├── Backtest.jl                        # Main entry point (configurable year via BACKTEST_YEAR)
├── Include.jl                         # Environment setup, package loading, source includes
├── Project.toml                       # Julia package dependencies
├── src/
│   ├── Files.jl                       # JLD2/CSV data loading utilities
│   ├── DataDownload.jl                # Yahoo Finance price/dividend download
│   ├── IVData.jl                      # WRDS OptionMetrics IV surface loading & lookup
│   ├── Compute.jl                     # Rolling vol, dividend yield, IV calibration (VRP)
│   ├── OptionPricing.jl               # CRR American pricing, Greeks, strike_from_delta
│   ├── EarningsCalendar.jl            # Earnings date loading and proximity check
│   ├── PortfolioConstruction.jl       # [PDF §3] WheelConfig, holidays, tenor, delta targeting
│   ├── OperationsCosts.jl             # [PDF §5] Fees, slippage, borrow, liquidity screens
│   ├── RiskCompliance.jl              # [PDF §4] VaR/ES, position limits, sector caps, KPIs
│   ├── Simulation.jl                  # [PDF §7B] HMM, GBM variants, stress scenarios
│   └── WheelEngine.jl                 # [PDF §7A] Core state machine + backtest loop
└── data/
    ├── wrds_iv_surface.csv            # WRDS OptionMetrics IV (local only, not in repo)
    ├── SAGBM-Parameters-Fall-2025.csv # Static drift/vol per ticker (fallback)
    ├── finviz.csv                     # Finviz screener (div yield > 3%)
    ├── prices_2025/                   # Cached daily OHLC per ticker
    └── dividends_2025/               # Cached dividend data per ticker
```

## Quick Start

```bash
julia Backtest.jl
```

Change the backtest year at the top of `Backtest.jl`:
```julia
const BACKTEST_YEAR = 2025   # change to 2024 for prior-year backtest
```

Enable parameter sweep or stress testing:
```julia
const RUN_PARAMETER_SWEEP = true
const RUN_STRESS_TEST     = true
```

## Configuration

All strategy parameters live in `WheelConfig` (see `src/PortfolioConstruction.jl`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `risk_free_rate` | 4.5% | Risk-free rate for option pricing |
| `tenor_days` | [7, 14, 30] | Expiry tenor ladder (calendar days) |
| `delta_put_safe` | (0.20, 0.30) | Put delta range — Safe sleeve |
| `delta_call_safe` | (0.25, 0.35) | Call delta range — Safe sleeve |
| `delta_put_aggr` | (0.25, 0.35) | Put delta range — Aggressive sleeve |
| `delta_call_aggr` | (0.30, 0.40) | Call delta range — Aggressive sleeve |
| `max_ladders` | 1 | Concurrent expiries per name (1–3) |
| `earnings_policy` | `:avoid` | `:avoid`, `:widen`, or `:reduce_size` |
| `crr_steps` | 50 | CRR lattice depth |
| `var_limit_daily` | 2.5% | Daily VaR cap for throttling |
| `max_name_weight` | 5% | Per-name NAV cap |

## Outputs

Each backtest generates 10 charts and 2 CSV files in `data/`:

| Output | Description |
|--------|-------------|
| `nav_curve_YYYY.png` | NAV vs SPY benchmark |
| `drawdown_YYYY.png` | Drawdown from peak |
| `monthly_returns_YYYY.png` | Monthly return heatmap |
| `income_decomposition_YYYY.png` | Premium vs dividend vs capital gain |
| `nav_composition_YYYY.png` | Cash / shares / option MTM breakdown |
| `option_mtm_YYYY.png` | Daily option mark-to-market |
| `premium_by_ticker_YYYY.png` | Premium collected per ticker |
| `rolling_sharpe_YYYY.png` | 60-day rolling Sharpe ratio |
| `return_distribution_YYYY.png` | Daily return histogram |
| `greeks_delta/gamma/vega_YYYY.png` | Portfolio-level Greeks over time |
| `daily_nav_YYYY.csv` | Full daily NAV series |
| `ticker_performance_YYYY.csv` | Per-ticker P&L breakdown |

## Annotations: Self-Designed vs Course Reference

**Course-aligned components (CHEME-5660):**
- CRR binomial lattice with dividend yield — Week 10
- Greeks (Δ, Γ, Θ, ν) via finite differences — Week 11
- Delta-based strike selection — Week 12b
- GBM path simulation — Week 5b
- Correlated multi-asset GBM (Cholesky) — Week 6a
- HMM regime detection (CDF quantile + `MyHiddenMarkovModel`) — Week 13

**Self-designed components (per Varner PDF):**
- Rolling volatility estimator and IV calibration (VRP model)
- WRDS OptionMetrics IV surface integration
- Bid-ask spread model with crowding-adjusted slippage
- Earnings calendar and avoidance/widening logic
- Ladder slot data structure and multi-slot state machine
- Regime-switching GBM and earnings jump diffusion
- Adaptive tenor/delta controls
- Cost-basis repair policy
- Parameter sweep framework

## Dependencies

See `Project.toml`. Key packages:
- `VLQuantitativeFinancePackage` — Varner Lab quantitative finance toolkit (HMM, lattice models)
- `YFinance` — Yahoo Finance data download
- `Distributions` — Laplace/Normal fitting for HMM, option pricing
- `DataFrames`, `CSV` — Data handling
- `Plots`, `StatsPlots` — Visualization
- `JLD2` — Binary data caching

## License

MIT — Zheyu Jin, 2026.
