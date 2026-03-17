"""
WRDS OptionMetrics Implied Volatility Surface — Data Loading & Lookup

Loads IV data from OptionMetrics CSV (downloaded via WRDS) and provides
fast lookup functions for the backtest engine.

Data columns used: ticker, date, days (DTE), delta, impl_volatility, cp_flag

Output format matches rolling_iv: Dict{String, Dict{Date, Float64}}
so it integrates seamlessly with WheelEngine.jl's run_backtest!().
"""

const _WRDS_IV_FILE = joinpath(_PATH_TO_DATA, "wrds_iv_surface.csv")

"""
    load_iv_surface(csv_path; tickers=nothing) -> DataFrame

Load WRDS OptionMetrics IV CSV. Filters to requested tickers and needed columns.
The file can be very large (>800MB), so we only keep what we need.
"""
function load_iv_surface(csv_path::String=_WRDS_IV_FILE;
                          tickers::Union{Nothing, Vector{String}}=nothing)::DataFrame
    if !isfile(csv_path)
        @warn "WRDS IV file not found at $csv_path — will use VRP fallback"
        return DataFrame()
    end

    println("  Loading WRDS IV surface from $(basename(csv_path))...")
    df = CSV.read(csv_path, DataFrame;
                  select=[:ticker, :date, :days, :delta, :impl_volatility, :cp_flag],
                  types=Dict(:date => Date, :delta => Float64, :impl_volatility => Float64,
                             :days => Int, :cp_flag => String))

    dropmissing!(df, :impl_volatility)
    filter!(row -> row.impl_volatility > 0.0, df)

    if tickers !== nothing
        filter!(row -> row.ticker in tickers, df)
    end

    sort!(df, [:ticker, :date, :days, :delta])
    println("  Loaded $(nrow(df)) IV records for $(length(unique(df.ticker))) tickers")
    return df
end

"""
    lookup_iv(iv_df, ticker, date, target_delta, dte; cp_flag="P") -> Union{Float64, Nothing}

Find the IV closest to target_delta for a given ticker/date/DTE.
Returns nothing if no data found.

delta convention in OptionMetrics: negative for puts (e.g., -30), positive for calls.
"""
function lookup_iv(iv_df::DataFrame, ticker::String, date::Date,
                    target_delta::Float64, dte::Int;
                    cp_flag::String="P")::Union{Float64, Nothing}
    nrow(iv_df) == 0 && return nothing

    mask = (iv_df.ticker .== ticker) .& (iv_df.date .== date) .&
           (iv_df.cp_flag .== cp_flag)
    subset = iv_df[mask, :]
    nrow(subset) == 0 && return nothing

    dte_diffs = abs.(subset.days .- dte)
    min_dte_diff = minimum(dte_diffs)
    subset = subset[dte_diffs .<= min_dte_diff + 5, :]

    delta_diffs = abs.(subset.delta .- target_delta)
    best_idx = argmin(delta_diffs)
    return subset.impl_volatility[best_idx]
end

"""
    lookup_atm_iv(iv_df, ticker, date; dte=30) -> Union{Float64, Nothing}

ATM implied volatility: delta ≈ -50 for puts (standard ATM convention).
"""
function lookup_atm_iv(iv_df::DataFrame, ticker::String, date::Date;
                        dte::Int=30)::Union{Float64, Nothing}
    return lookup_iv(iv_df, ticker, date, -50.0, dte; cp_flag="P")
end

"""
    build_daily_iv_map(iv_df, tickers, trading_days; dte=30) -> Dict{String, Dict{Date, Float64}}

Pre-build an IV lookup map in the same format as compute_rolling_iv() output.
Uses ATM put IV (delta=-50) as the representative daily IV per ticker.

Falls back to nearest available date within 3 calendar days if exact date missing.
"""
function build_daily_iv_map(iv_df::DataFrame, tickers::Vector{String},
                             trading_days::Vector{Date};
                             dte::Int=30)::Dict{String, Dict{Date, Float64}}
    result = Dict{String, Dict{Date, Float64}}()
    nrow(iv_df) == 0 && return result

    for tk in tickers
        tk_mask = iv_df.ticker .== tk
        tk_df = iv_df[tk_mask, :]
        nrow(tk_df) == 0 && continue

        iv_dict = Dict{Date, Float64}()
        available_dates = unique(tk_df.date)
        date_set = Set(available_dates)

        for d in trading_days
            iv = lookup_atm_iv(tk_df, tk, d; dte=dte)
            if iv !== nothing
                iv_dict[d] = iv
            else
                for offset in 1:3
                    for dd in [d - Day(offset), d + Day(offset)]
                        dd in date_set || continue
                        iv2 = lookup_atm_iv(tk_df, tk, dd; dte=dte)
                        if iv2 !== nothing
                            iv_dict[d] = iv2
                            @goto found
                        end
                    end
                end
                @label found
            end
        end

        if !isempty(iv_dict)
            result[tk] = iv_dict
        end
    end

    println("  Built daily IV map: $(length(result)) tickers with WRDS IV data")
    return result
end
