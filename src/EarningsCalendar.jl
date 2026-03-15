"""
Earnings calendar for Wheel strategy earnings-avoidance logic.

The Varner PDF Section 6 lists "Earnings policy: avoid vs trade-through with
reduced size/wider strikes" as a key design variable.

Enhancements (TODO item 18):
  - Per-company estimated earnings dates based on typical SEC filing patterns
  - HTTP-based fetch from Yahoo Finance calendarEvents API
  - CSV loader as primary source when available
"""

struct EarningsCalendar
    dates::Dict{String, Vector{Date}}
end

"""
    load_earnings_calendar(tickers; year=2025) -> EarningsCalendar

Load earnings dates from `data/earnings_calendar.csv` if available.
Expected CSV format: ticker,date (one row per earnings event).
Falls back to per-company estimated dates if CSV is missing.
"""
function load_earnings_calendar(tickers::Vector{String}; year::Int=2025,
                                 try_fetch::Bool=false)::EarningsCalendar
    path = joinpath(_PATH_TO_DATA, "earnings_calendar.csv")
    if isfile(path)
        df = CSV.read(path, DataFrame)
        dates = Dict{String, Vector{Date}}()
        for row in eachrow(df)
            t = string(row.ticker)
            d = Date(row.date)
            haskey(dates, t) || (dates[t] = Date[])
            push!(dates[t], d)
        end
        for (_, v) in dates
            sort!(v)
        end
        return EarningsCalendar(dates)
    end

    cal = estimate_earnings_calendar(tickers; year=year)

    if try_fetch
        for ticker in tickers
            fetched = fetch_earnings_dates(ticker; year=year)
            if !isempty(fetched)
                cal.dates[ticker] = fetched
            end
        end
    end

    return cal
end

const KNOWN_EARNINGS_PATTERNS = Dict{String, Vector{Tuple{Int,Int}}}(
    "AAPL"  => [(1,30), (5,1),  (7,31), (10,30)],
    "MSFT"  => [(1,28), (4,29), (7,22), (10,28)],
    "AMZN"  => [(2,6),  (4,29), (8,1),  (10,30)],
    "GOOG"  => [(2,4),  (4,24), (7,22), (10,29)],
    "META"  => [(1,29), (4,30), (7,30), (10,29)],
    "NVDA"  => [(2,26), (5,28), (8,27), (11,19)],
    "TSLA"  => [(1,29), (4,22), (7,22), (10,21)],
    "AMD"   => [(2,4),  (4,29), (7,29), (10,28)],
    "NFLX"  => [(1,21), (4,17), (7,17), (10,16)],
    "DVN"   => [(2,18), (5,6),  (8,5),  (11,4)],
    "PEP"   => [(2,4),  (4,24), (7,10), (10,8)],
    "KO"    => [(2,11), (4,29), (7,22), (10,21)],
    "PG"    => [(1,22), (4,23), (7,30), (10,18)],
    "JNJ"   => [(1,22), (4,15), (7,15), (10,14)],
    "IBM"   => [(1,29), (4,23), (7,23), (10,22)],
    "VZ"    => [(1,24), (4,22), (7,22), (10,21)],
    "T"     => [(1,22), (4,23), (7,23), (10,22)],
)

"""
    estimate_earnings_calendar(tickers; year=2025) -> EarningsCalendar

Per-company estimated quarterly earnings dates based on typical SEC filing patterns.
Uses KNOWN_EARNINGS_PATTERNS for well-known names, falls back to generic
quarterly dates (Feb 1, May 1, Aug 1, Nov 1) for others.
"""
function estimate_earnings_calendar(tickers::Vector{String}; year::Int=2025)::EarningsCalendar
    dates = Dict{String, Vector{Date}}()
    for ticker in tickers
        if haskey(KNOWN_EARNINGS_PATTERNS, ticker)
            dates[ticker] = [Date(year, m, d) for (m, d) in KNOWN_EARNINGS_PATTERNS[ticker]]
        else
            dates[ticker] = [
                Date(year, 2, 1), Date(year, 5, 1),
                Date(year, 8, 1), Date(year, 11, 1)
            ]
        end
    end
    return EarningsCalendar(dates)
end

"""
    fetch_earnings_dates(ticker; year=2025) -> Vector{Date}

Attempt to fetch actual earnings dates from Yahoo Finance calendarEvents API.
Returns empty vector on failure. Self-designed (no direct course reference).
"""
function fetch_earnings_dates(ticker::String; year::Int=2025)::Vector{Date}
    try
        url = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/$(ticker)?modules=calendarEvents"
        resp = HTTP.get(url; headers=["User-Agent" => "Mozilla/5.0"], connect_timeout=5, readtimeout=10)
        data = JSON3.read(String(resp.body))
        earnings = data[:quoteSummary][:result][1][:calendarEvents][:earnings]
        raw_date = get(earnings, :earningsDate, nothing)
        raw_date === nothing && return Date[]
        if raw_date isa AbstractVector
            return [Date(unix2datetime(d[:raw])) for d in raw_date if haskey(d, :raw)]
        end
        return Date[]
    catch e
        @debug "fetch_earnings_dates failed for $ticker: $e"
        return Date[]
    end
end

"""
    near_earnings(cal, ticker, date, buffer_days) -> Bool

Returns true if `date` falls within `buffer_days` of any earnings date for `ticker`.
"""
function near_earnings(cal::EarningsCalendar, ticker::String, date::Date, buffer::Int)::Bool
    !haskey(cal.dates, ticker) && return false
    for ed in cal.dates[ticker]
        abs(Dates.value(date - ed)) <= buffer && return true
    end
    return false
end
