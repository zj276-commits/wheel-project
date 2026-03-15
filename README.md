# Wheel Strategy ETF Backtest Engine

A Julia-based backtesting system for the **Wheel option strategy** applied to an S&P 500 universe, following the design specification in [Varner PDF: Wheel-Fund-draft](docs/Wheel-Fund-draft-Varner.pdf).

## Strategy Overview

Per ticker, maintain two 100-share blocks:
1. **Block A (Dividend/Growth)** — Buy-and-hold for dividends and capital appreciation.
2. **Block B (Option Income)** — Run the Wheel cycle:
   - Sell cash-secured puts → assigned → hold shares → sell covered calls → called away → repeat.

Portfolio split: 60% Safe sleeve (low-vol, high-dividend) / 40% Aggressive sleeve (high-vol).

## Key Features

| Feature | Implementation | Reference |
|---------|---------------|-----------|
| **American option pricing** | CRR binomial lattice (N=50 steps) | CHEME-5660 Week 10 |
| **Dynamic volatility** | 30-day rolling realized vol | Self-designed |
| **Earnings avoidance** | Skip or widen delta near earnings | Self-designed (Varner PDF §6) |
| **Laddering** | 1–3 concurrent expiries per name | Self-designed (Varner PDF §6) |
| **Bid-ask spread model** | Vol-adjusted half-spread cost | Self-designed |
| **Roll rules** | Time-based + 3 trigger-based rules | Varner PDF §3 |
| **Cost-basis repair** | Avg-down with quarterly caps | Varner PDF §6 |
| **Monte Carlo stress testing** | GBM + regime-switching | CHEME-5660 Week 5–6 |
| **Parameter sweep** | Grid search over design variables | Varner PDF §6 |

## Project Structure

```
wheel-project-1/
├── Backtest.jl                     # Main entry point — run backtest, sweep, stress test
├── Include.jl                      # Environment setup, package loading, source includes
├── Project.toml                    # Julia package dependencies
├── README.md
├── src/
│   ├── OptionPricing.jl            # BS European + CRR American option pricing
│   ├── Compute.jl                  # Log growth matrix + rolling volatility
│   ├── DataDownload.jl             # Yahoo Finance price/dividend download
│   ├── EarningsCalendar.jl         # Earnings date loading and proximity check
│   ├── Files.jl                    # JLD2/CSV data loading utilities
│   ├── MonteCarloSim.jl            # GBM simulation + stress scenarios
│   └── WheelEngine.jl              # Core backtest engine (state machine, rolls, repair)
└── data/
    ├── SAGBM-Parameters-Fall-2025.csv    # Static drift/vol per ticker (fallback)
    ├── finviz.csv                        # Finviz screener (div yield > 3%)
    ├── earnings_calendar.csv             # Optional: actual earnings dates
    ├── prices_2025/                      # Cached daily OHLC per ticker
    └── dividends_2025/                   # Cached dividend data per ticker
```

## Quick Start

```julia
# Run the baseline backtest
julia Backtest.jl
```

To enable parameter sweep or stress testing, edit the switches at the top of `Backtest.jl`:
```julia
const RUN_PARAMETER_SWEEP = true    # grid search over config variants
const RUN_STRESS_TEST     = true    # Monte Carlo stress scenarios
```

## Configuration

All strategy parameters are in `WheelConfig` (see `src/WheelEngine.jl`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `risk_free_rate` | 4.5% | Used for option pricing |
| `tenor_days` | [7, 14, 30] | Expiry ladder |
| `delta_put_safe` | (0.20, 0.30) | Put delta range for Safe sleeve |
| `delta_call_safe` | (0.25, 0.35) | Call delta range for Safe sleeve |
| `max_ladders` | 1 | Concurrent expiries per name (1–3) |
| `earnings_policy` | `:avoid` | `:avoid` or `:widen` near earnings |
| `earnings_buffer_days` | 5 | Days to avoid around earnings |
| `bid_ask_spread_model` | true | Use vol-adjusted spread model |
| `crr_steps` | 50 | CRR lattice depth |

## Annotations: Self-Designed vs Course Reference

Components marked **"Self-designed"** have no direct equivalent in the CHEME-5660 course:
- Rolling volatility estimator (`Compute.jl`)
- Earnings calendar and avoidance logic (`EarningsCalendar.jl`)
- Bid-ask spread model (`WheelEngine.jl: estimate_half_spread`)
- Ladder slot data structure (`WheelEngine.jl: LadderSlot`)
- Regime-switching GBM stress scenarios (`MonteCarloSim.jl`)
- Parameter sweep framework (`Backtest.jl`)

Components adapted from course materials:
- CRR binomial lattice: CHEME-5660 Week 10, L10a/L10b
- GBM path simulation: CHEME-5660 Week 5b–6a
- Delta from lattice: CHEME-5660 Week 12b
- Black-Scholes analytical: Standard reference (retained for strike selection)

## Dependencies

See `Project.toml`. Key packages:
- `VLQuantitativeFinancePackage` — Varner Lab quantitative finance toolkit
- `YFinance` — Yahoo Finance data download
- `Distributions` — Normal CDF/quantile for BS formulas
- `DataFrames`, `CSV` — Data handling
- `Plots`, `StatsPlots` — Visualization

## License

MIT — Zheyu Jin, 2026.
