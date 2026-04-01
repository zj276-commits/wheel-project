"""
Data loading from VLQuantitativeFinancePackage (Polygon.io).

Training set: MyTrainingMarketDataSet()  → SP500 OHLC, 2014-01-03 to 2024-12-31
Testing set:  MyTestingMarketDataSet()   → SP500 OHLC, 2025-01-02 to present

Polygon columns → our columns:
  timestamp → date (Date)
  open, high, low, close, volume → same names
  close → adj_close (Polygon prices are already split-adjusted)
  volume_weighted_average_price, number_of_transactions → dropped

VXX (iPath VIX ETN) is used as the VIX proxy.
Dividend data still uses cached CSVs from prior YFinance downloads.
"""

const TICKER_ALIASES = Dict{String,String}(
    "META" => "FB",
)

# Lazy-loaded global caches — loaded once on first access
const _VLQF_TRAINING = Ref{Union{Nothing, Dict{String, DataFrame}}}(nothing)
const _VLQF_TESTING  = Ref{Union{Nothing, Dict{String, DataFrame}}}(nothing)

function _ensure_training_loaded()
    if _VLQF_TRAINING[] === nothing
        println("  Loading VLQuantitativeFinancePackage training data (2014-2024)...")
        _VLQF_TRAINING[] = MyTrainingMarketDataSet()["dataset"]
        println("  Training data: $(length(_VLQF_TRAINING[])) tickers")
    end
    return _VLQF_TRAINING[]
end

function _ensure_testing_loaded()
    if _VLQF_TESTING[] === nothing
        println("  Loading VLQuantitativeFinancePackage testing data (2025)...")
        _VLQF_TESTING[] = MyTestingMarketDataSet()["dataset"]
        println("  Testing data: $(length(_VLQF_TESTING[])) tickers")
    end
    return _VLQF_TESTING[]
end

"""
Convert a Polygon DataFrame to our standard format.
Polygon columns: volume, volume_weighted_average_price, open, close, high, low, timestamp, number_of_transactions
Our columns: date, open, high, low, close, adj_close, volume
"""
function _polygon_to_standard(df::DataFrame)::DataFrame
    result = DataFrame(
        date      = Date.(df.timestamp),
        open      = Float64.(df.open),
        high      = Float64.(df.high),
        low       = Float64.(df.low),
        close     = Float64.(df.close),
        adj_close = Float64.(df.close),
        volume    = Float64.(df.volume),
    )
    filter!(row -> !isnan(row.close) && row.close > 0.0, result)
    sort!(result, :date)
    return result
end

"""
    download_price_data(ticker, start_date, end_date; cache_year=2025) -> DataFrame

Load daily OHLC price data from VLQuantitativeFinancePackage.
Falls back to cached CSV if ticker not found in package data.
Returns a DataFrame with columns: date, open, high, low, close, adj_close, volume.
"""
function download_price_data(ticker::String, start_date::Date, end_date::Date;
                              cache_year::Int=2025)::DataFrame
    vlqf_data = cache_year >= 2025 ? _ensure_testing_loaded() : _ensure_training_loaded()

    actual_ticker = ticker == "^VIX" ? "VXX" : ticker

    if haskey(vlqf_data, actual_ticker)
        raw_df = vlqf_data[actual_ticker]
        std_df = _polygon_to_standard(raw_df)
        return filter(row -> start_date <= row.date <= end_date, std_df)
    end

    if haskey(TICKER_ALIASES, actual_ticker)
        alias = TICKER_ALIASES[actual_ticker]
        if haskey(vlqf_data, alias)
            println("    $actual_ticker not found → using alias $alias")
            raw_df = vlqf_data[alias]
            std_df = _polygon_to_standard(raw_df)
            return filter(row -> start_date <= row.date <= end_date, std_df)
        end
    end

    if cache_year < 2025
        training = _ensure_training_loaded()
        if haskey(training, actual_ticker)
            raw_df = training[actual_ticker]
            std_df = _polygon_to_standard(raw_df)
            return filter(row -> start_date <= row.date <= end_date, std_df)
        end
        if haskey(TICKER_ALIASES, actual_ticker)
            alias = TICKER_ALIASES[actual_ticker]
            if haskey(training, alias)
                println("    $actual_ticker not found → using alias $alias (training)")
                raw_df = training[alias]
                std_df = _polygon_to_standard(raw_df)
                return filter(row -> start_date <= row.date <= end_date, std_df)
            end
        end
    end

    cache_dir = joinpath(_PATH_TO_DATA, "prices_$(cache_year)")
    cache_file = joinpath(cache_dir, "$(ticker).csv")
    if isfile(cache_file)
        return CSV.read(cache_file, DataFrame)
    end

    @warn "Ticker $ticker not found in VLQuantitativeFinancePackage or cache"
    return DataFrame(date=Date[], open=Float64[], high=Float64[], low=Float64[],
                     close=Float64[], adj_close=Float64[], volume=Float64[])
end

"""
    download_all_prices(tickers, start_date, end_date; cache_year=2025) -> Dict{String, DataFrame}

Load daily prices for all tickers from VLQuantitativeFinancePackage.
"""
function download_all_prices(tickers::Vector{String}, start_date::Date, end_date::Date;
                              cache_year::Int=2025)::Dict{String, DataFrame}
    cache_year >= 2025 ? _ensure_testing_loaded() : _ensure_training_loaded()

    result = Dict{String, DataFrame}()
    for (i, ticker) in enumerate(tickers)
        print("  [$i/$(length(tickers))] $ticker ... ")
        try
            result[ticker] = download_price_data(ticker, start_date, end_date; cache_year=cache_year)
            println("$(nrow(result[ticker])) days")
        catch e
            @warn "Failed to load $ticker: $e"
        end
    end
    return result
end

"""
    download_dividends(ticker, start_date, end_date; cache_year=2025) -> DataFrame

Load dividend data from cached CSVs (dividend data not available in VLQuantitativeFinancePackage).
Returns DataFrame with columns: ex_date, amount.
"""
function download_dividends(ticker::String, start_date::Date, end_date::Date;
                             cache_year::Int=2025)::DataFrame
    cache_dir = joinpath(_PATH_TO_DATA, "dividends_$(cache_year)")
    cache_file = joinpath(cache_dir, "$(ticker).csv")

    if isfile(cache_file)
        return CSV.read(cache_file, DataFrame)
    end

    return DataFrame(ex_date=Date[], amount=Float64[])
end

"""
    download_all_dividends(tickers, start_date, end_date; cache_year=2025) -> Dict{String, DataFrame}

Load dividend data for all tickers from cached CSVs.
"""
function download_all_dividends(tickers::Vector{String}, start_date::Date, end_date::Date;
                                 cache_year::Int=2025)::Dict{String, DataFrame}
    result = Dict{String, DataFrame}()
    for ticker in tickers
        try
            result[ticker] = download_dividends(ticker, start_date, end_date; cache_year=cache_year)
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
