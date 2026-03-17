# Include.jl — Environment setup for the Wheel ETF Strategy project
#
# Load order mirrors the PDF architecture:
#   Foundation → PDF §3 (Portfolio) → PDF §5 (Ops/Costs) → PDF §4 (Risk)
#   → PDF §7B (Simulation, incl. HMM) → PDF §7A (Backtest Engine)

const _ROOT = @__DIR__;
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_DATA = joinpath(_ROOT, "data");

using Pkg
Pkg.activate(_ROOT);
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

using VLQuantitativeFinancePackage
using DataFrames
using CSV
using Dates
using LinearAlgebra
using Statistics
using Plots
using Colors
using StatsPlots
using JLD2
using FileIO
using Distributions
using PrettyTables
using StatsBase
using HTTP
using JSON3
using YFinance

# ── Source modules (dependency order) ─────────────────────────────────────────

# 1. Foundation layer
include(joinpath(_PATH_TO_SRC, "Files.jl"));
include(joinpath(_PATH_TO_SRC, "DataDownload.jl"));
include(joinpath(_PATH_TO_SRC, "IVData.jl"));                # WRDS OptionMetrics IV surface
include(joinpath(_PATH_TO_SRC, "Compute.jl"));                # rolling vol, trailing div yield, correlation
include(joinpath(_PATH_TO_SRC, "OptionPricing.jl"));          # CRR, Greeks, strike_from_delta
include(joinpath(_PATH_TO_SRC, "EarningsCalendar.jl"));       # earnings dates

# 2. PDF Section 3 — Portfolio Construction
include(joinpath(_PATH_TO_SRC, "PortfolioConstruction.jl"));

# 3. PDF Section 5 — Operations & Costs
include(joinpath(_PATH_TO_SRC, "OperationsCosts.jl"));

# 4. PDF Section 4 — Risk & Compliance
include(joinpath(_PATH_TO_SRC, "RiskCompliance.jl"));

# 5. PDF Section 7B — Simulation (HMM via MyHiddenMarkovModel + GBM + stress)
include(joinpath(_PATH_TO_SRC, "Simulation.jl"));

# 6. PDF Section 7A — Backtest Engine (state machine + loop)
include(joinpath(_PATH_TO_SRC, "WheelEngine.jl"));
