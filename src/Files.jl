"""
Data loading utilities.

Stock price data (OHLC) is loaded via VLQuantitativeFinancePackage:
  MyTrainingMarketDataSet() → 2014-2024
  MyTestingMarketDataSet()  → 2025

These functions are exported by the package and should NOT be redefined here.
"""

"""
    load_sagbm_parameters() -> DataFrame

Load the S&P 500 GBM parameters CSV containing precomputed drift and volatility
for each ticker. This can serve as a fallback when JLD2 market data is unavailable.
"""
function load_sagbm_parameters()::DataFrame
    path = joinpath(_PATH_TO_DATA, "SAGBM-Parameters-Fall-2025.csv");
    if !isfile(path)
        @warn "SAGBM parameters file not found at $path — returning empty DataFrame"
        return DataFrame(ticker=String[], drift=Float64[], volatility=Float64[])
    end
    return CSV.read(path, DataFrame);
end

"""
    load_finviz_screener() -> DataFrame

Load the finviz screener CSV (pre-filtered for dividend yield > 3%).
Parses the Dividend Yield column from percentage strings to Float64.
"""
function load_finviz_screener()::DataFrame
    path = joinpath(_PATH_TO_DATA, "finviz.csv");
    if !isfile(path)
        @warn "Finviz screener file not found at $path — returning empty DataFrame"
        return DataFrame(Ticker=String[], div_yield=Float64[])
    end
    df = CSV.read(path, DataFrame);

    df[!, :div_yield] = map(df[!, "Dividend Yield"]) do x
        s = replace(string(x), "%" => "")
        val = tryparse(Float64, s)
        return val === nothing ? 0.0 : val
    end

    return df;
end
