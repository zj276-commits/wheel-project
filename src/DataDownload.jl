"""Convert YFinance timestamps to Date vector, handling DateTime or numeric types."""
function _to_dates(timestamps)::Vector{Date}
    return [t isa DateTime ? Date(t) : Date(unix2datetime(Float64(t))) for t in timestamps]
end

"""Convert YFinance numeric vectors to Float64, replacing nothing/missing with NaN."""
function _to_float_vec(v)::Vector{Float64}
    return [x === nothing || ismissing(x) ? NaN : Float64(x) for x in v]
end

"""
    download_price_data(ticker, start_date, end_date) -> DataFrame

Download daily OHLC price data from Yahoo Finance via YFinance.jl.
Caches results to data/prices_2025/<TICKER>.csv to avoid repeated API calls.
Returns a DataFrame with columns: date, open, high, low, close, adj_close, volume.
"""
function download_price_data(ticker::String, start_date::Date, end_date::Date)::DataFrame
    cache_dir = joinpath(_PATH_TO_DATA, "prices_2025")
    mkpath(cache_dir)
    cache_file = joinpath(cache_dir, "$(ticker).csv")

    if isfile(cache_file)
        return CSV.read(cache_file, DataFrame)
    end

    sd = Dates.format(start_date, "yyyy-mm-dd")
    ed = Dates.format(end_date, "yyyy-mm-dd")

    raw = get_prices(ticker, startdt=sd, enddt=ed, interval="1d")

    if !haskey(raw, "timestamp") || isempty(raw["timestamp"])
        @warn "No price data returned for $ticker"
        return DataFrame(date=Date[], open=Float64[], high=Float64[], low=Float64[],
                         close=Float64[], adj_close=Float64[], volume=Float64[])
    end

    timestamps = raw["timestamp"]
    dates = _to_dates(timestamps)

    df = DataFrame(
        date      = dates,
        open      = _to_float_vec(get(raw, "open", fill(NaN, length(dates)))),
        high      = _to_float_vec(get(raw, "high", fill(NaN, length(dates)))),
        low       = _to_float_vec(get(raw, "low", fill(NaN, length(dates)))),
        close     = _to_float_vec(get(raw, "close", fill(NaN, length(dates)))),
        adj_close = _to_float_vec(get(raw, "adjclose", get(raw, "close", fill(NaN, length(dates))))),
        volume    = _to_float_vec(get(raw, "vol", fill(0.0, length(dates)))),
    )

    filter!(row -> !isnan(row.close), df)
    sort!(df, :date)

    CSV.write(cache_file, df)
    return df
end

"""
    download_all_prices(tickers, start_date, end_date) -> Dict{String, DataFrame}

Download daily prices for all tickers. Shows progress and handles errors gracefully.
"""
function download_all_prices(tickers::Vector{String}, start_date::Date, end_date::Date)::Dict{String, DataFrame}
    result = Dict{String, DataFrame}()
    for (i, ticker) in enumerate(tickers)
        print("  [$i/$(length(tickers))] $ticker ... ")
        try
            result[ticker] = download_price_data(ticker, start_date, end_date)
            println("$(nrow(result[ticker])) days")
        catch e
            @warn "Failed to download $ticker: $e"
        end
    end
    return result
end

"""
    download_dividends(ticker, start_date, end_date) -> DataFrame

Download dividend payment data from Yahoo Finance.
Returns DataFrame with columns: ex_date, amount.
"""
function download_dividends(ticker::String, start_date::Date, end_date::Date)::DataFrame
    cache_dir = joinpath(_PATH_TO_DATA, "dividends_2025")
    mkpath(cache_dir)
    cache_file = joinpath(cache_dir, "$(ticker).csv")

    if isfile(cache_file)
        return CSV.read(cache_file, DataFrame)
    end

    sd = Dates.format(start_date, "yyyy-mm-dd")
    ed = Dates.format(end_date, "yyyy-mm-dd")

    raw = get_prices(ticker, startdt=sd, enddt=ed, interval="1d", divsplits=true)

    if !haskey(raw, "timestamp") || isempty(raw["timestamp"]) || !haskey(raw, "div")
        CSV.write(cache_file, DataFrame(ex_date=Date[], amount=Float64[]))
        return DataFrame(ex_date=Date[], amount=Float64[])
    end

    timestamps = raw["timestamp"]
    dates = _to_dates(timestamps)
    divs = _to_float_vec(raw["div"])

    div_df = DataFrame(ex_date = dates, amount = divs)
    filter!(row -> row.amount > 0.0, div_df)

    CSV.write(cache_file, div_df)
    return div_df
end

"""
    download_all_dividends(tickers, start_date, end_date) -> Dict{String, DataFrame}

Download dividend data for all tickers.
"""
function download_all_dividends(tickers::Vector{String}, start_date::Date, end_date::Date)::Dict{String, DataFrame}
    result = Dict{String, DataFrame}()
    for ticker in tickers
        try
            result[ticker] = download_dividends(ticker, start_date, end_date)
        catch e
            result[ticker] = DataFrame(ex_date = Date[], amount = Float64[])
        end
    end
    return result
end

"""
    get_trading_days(price_data) -> Vector{Date}

Extract the union of all trading dates across all tickers, sorted chronologically.
"""
function get_trading_days(price_data::Dict{String, DataFrame})::Vector{Date}
    all_dates = Set{Date}()
    for (_, df) in price_data
        union!(all_dates, df.date)
    end
    return sort(collect(all_dates))
end

"""
    get_price_on_date(price_df, date) -> Union{Float64, Nothing}

Look up the adjusted close price on a given date. Returns nothing if not found.
"""
function get_price_on_date(price_df::DataFrame, date::Date)::Union{Float64, Nothing}
    idx = findfirst(==(date), price_df.date)
    return idx === nothing ? nothing : price_df.adj_close[idx]
end
