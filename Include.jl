# Include.jl — Environment setup for the Wheel ETF Strategy project
# Sets up directory paths, installs/loads Julia packages, and includes
# local source modules in dependency order.

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

# Source modules — order matters (dependencies must come first)
include(joinpath(_PATH_TO_SRC, "Files.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
include(joinpath(_PATH_TO_SRC, "DataDownload.jl"));
include(joinpath(_PATH_TO_SRC, "OptionPricing.jl"));       # replaces BlackScholes.jl
include(joinpath(_PATH_TO_SRC, "EarningsCalendar.jl"));     # earnings avoidance
include(joinpath(_PATH_TO_SRC, "MonteCarloSim.jl"));        # GBM stress testing
include(joinpath(_PATH_TO_SRC, "WheelEngine.jl"));
