using Dates

const TARGET_TICKERS = Set([
    "PEP","KO","PG","JNJ","CME","CMCSA","VZ","T","IBM","MO",
    "PM","MDLZ","EXC","KMB","PAYX","TROW","PFG","SO","DUK","ED",
    "LNT","GIS","CAG","REG","CPB",
    "TSLA","NVDA","AMD","AAPL","MSFT","AMZN","GOOG","META","NFLX","DVN"
])

const INPUT_FILE = raw"C:\Users\aaron\Desktop\2014-2026 option price.csv"
const OUTPUT_DIR = joinpath(@__DIR__, "data")
const OUTPUT_PRICES = joinpath(OUTPUT_DIR, "option_prices.csv")
const OUTPUT_IV = joinpath(OUTPUT_DIR, "wrds_iv_surface.csv")

function preprocess()
    println("Reading: $INPUT_FILE")
    println("Target tickers: $(length(TARGET_TICKERS))")

    header_line = ""
    col_indices = Dict{String, Int}()
    needed_cols = ["secid","date","exdate","cp_flag","strike_price",
                   "best_bid","best_offer","impl_volatility","delta",
                   "gamma","vega","theta","volume","open_interest",
                   "ticker","exercise_style"]

    lines_read = 0
    lines_kept = 0

    open(OUTPUT_PRICES, "w") do out_prices
        open(OUTPUT_IV, "w") do out_iv
            write(out_prices, "ticker,date,exdate,cp_flag,strike_price,best_bid,best_offer,impl_volatility,delta,gamma,vega,theta,volume,open_interest\n")
            write(out_iv, "ticker,date,days,delta,impl_volatility,cp_flag\n")

            open(INPUT_FILE, "r") do f
                header_line = readline(f)
                cols = split(header_line, ",")
                for (i, c) in enumerate(cols)
                    col_indices[strip(c)] = i
                end

                println("Columns found: $(length(cols))")
                for nc in needed_cols
                    if !haskey(col_indices, nc)
                        println("  WARNING: column '$nc' not found!")
                    end
                end

                idx_ticker = get(col_indices, "ticker", 0)
                idx_date = get(col_indices, "date", 0)
                idx_exdate = get(col_indices, "exdate", 0)
                idx_cp = get(col_indices, "cp_flag", 0)
                idx_strike = get(col_indices, "strike_price", 0)
                idx_bid = get(col_indices, "best_bid", 0)
                idx_offer = get(col_indices, "best_offer", 0)
                idx_iv = get(col_indices, "impl_volatility", 0)
                idx_delta = get(col_indices, "delta", 0)
                idx_gamma = get(col_indices, "gamma", 0)
                idx_vega = get(col_indices, "vega", 0)
                idx_theta = get(col_indices, "theta", 0)
                idx_vol = get(col_indices, "volume", 0)
                idx_oi = get(col_indices, "open_interest", 0)

                while !eof(f)
                    line = readline(f)
                    lines_read += 1

                    if lines_read % 5_000_000 == 0
                        println("  Processed $(lines_read ÷ 1_000_000)M lines, kept $lines_kept ...")
                    end

                    fields = split(line, ",")
                    length(fields) < maximum(values(col_indices)) && continue

                    ticker = strip(fields[idx_ticker])
                    ticker in TARGET_TICKERS || continue

                    date_str = strip(fields[idx_date])
                    (date_str >= "2024-01-01" && date_str <= "2025-12-31") || continue

                    exdate_str = strip(fields[idx_exdate])
                    cp = strip(fields[idx_cp])
                    strike = strip(fields[idx_strike])
                    bid = strip(fields[idx_bid])
                    offer = strip(fields[idx_offer])
                    iv = strip(fields[idx_iv])
                    delta_val = strip(fields[idx_delta])
                    gamma_val = strip(fields[idx_gamma])
                    vega_val = strip(fields[idx_vega])
                    theta_val = strip(fields[idx_theta])
                    vol_val = strip(fields[idx_vol])
                    oi_val = strip(fields[idx_oi])

                    write(out_prices, "$ticker,$date_str,$exdate_str,$cp,$strike,$bid,$offer,$iv,$delta_val,$gamma_val,$vega_val,$theta_val,$vol_val,$oi_val\n")
                    lines_kept += 1

                    if iv != "" && delta_val != ""
                        try
                            y1, m1, d1 = parse(Int, date_str[1:4]), parse(Int, date_str[6:7]), parse(Int, date_str[9:10])
                            y2, m2, d2 = parse(Int, exdate_str[1:4]), parse(Int, exdate_str[6:7]), parse(Int, exdate_str[9:10])
                            dte = Dates.value(Date(y2,m2,d2) - Date(y1,m1,d1))
                            if dte > 0
                                write(out_iv, "$ticker,$date_str,$dte,$delta_val,$iv,$cp\n")
                            end
                        catch
                        end
                    end
                end
            end
        end
    end

    println("\nDone!")
    println("  Total lines read: $lines_read")
    println("  Lines kept: $lines_kept")
    println("  Option prices: $OUTPUT_PRICES")
    println("  IV surface:    $OUTPUT_IV")
end

preprocess()
