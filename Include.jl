# =============================================================================
# Include.jl — Environment setup for the Wheel ETF Strategy project
# =============================================================================
# Sets up directory paths, installs/loads Julia packages, and includes
# local source modules (Files.jl, Compute.jl).
# =============================================================================

# Directory paths (relative to this file's location)
const _ROOT = @__DIR__;
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_DATA = joinpath(_ROOT, "data");

# Package environment setup
using Pkg
Pkg.activate(_ROOT);
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# External packages
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

# Local source modules
include(joinpath(_PATH_TO_SRC, "Files.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
include(joinpath(_PATH_TO_SRC, "DataDownload.jl"));
include(joinpath(_PATH_TO_SRC, "BlackScholes.jl"));
include(joinpath(_PATH_TO_SRC, "WheelEngine.jl"));
