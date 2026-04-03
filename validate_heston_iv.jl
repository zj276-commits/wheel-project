"""
validate_heston_iv.jl — Compare Heston model IV vs WRDS real market IV

Reads:
  data/heston_params.csv   — calibrated Heston parameters (from calibrate_heston.jl)
  data/wrds_iv_surface.csv — real market IV from WRDS OptionMetrics

Outputs:
  Terminal tables: per-ticker, per-DTE-bucket, per-delta-bucket error statistics
  data/iv_validation_results.csv — full comparison dataset

Usage:
    julia validate_heston_iv.jl            # sample every 100th row (~55K points)
    julia validate_heston_iv.jl 50         # sample every 50th row (~110K points)
"""

include(joinpath(@__DIR__, "Include.jl"))

const WRDS_IV_FILE = joinpath(_PATH_TO_DATA, "wrds_iv_surface.csv")
const R = 0.045
const SAMPLE_EVERY = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100

println("═══════════════════════════════════════════════════")
println("  Heston IV vs WRDS Market IV Validation")
println("  Sampling every $(SAMPLE_EVERY)th row")
println("═══════════════════════════════════════════════════\n")

if !isfile(WRDS_IV_FILE)
    error("wrds_iv_surface.csv not found in data/")
end
if !isfile(HESTON_CAL_PATH)
    error("heston_params.csv not found. Run calibrate_heston.jl first.")
end

# ── Step 1: Load Heston calibration ──────────────────────────────────────────

println("  Step 1: Loading Heston parameters...")
heston_ts = load_heston_params(HESTON_CAL_PATH)
heston_tickers = Set(keys(heston_ts))
println("  Tickers with Heston calibration: $(length(heston_tickers))")

# ── Step 2: Download price data for date → S mapping ────────────────────────

println("\n  Step 2: Loading price data for S lookup...")
cal_start = Date(2024, 1, 1)
cal_end   = Date(2025, 12, 31)
price_data = download_all_prices(collect(heston_tickers), cal_start, cal_end; cache_year=2025)
div_data   = download_all_dividends(collect(heston_tickers), cal_start, cal_end; cache_year=2025)

price_lookup = Dict{String, Dict{Date, Float64}}()
for (tk, df) in price_data
    pd = Dict{Date, Float64}()
    for row in eachrow(df)
        pd[row.date] = row.adj_close
    end
    price_lookup[tk] = pd
end
println("  Price data: $(length(price_lookup)) tickers")

# ── Step 3: Compute per-stock RV regime ──────────────────────────────────────

println("\n  Step 3: Computing per-stock RV regime...")
stock_rolling_vol = compute_rolling_volatility(price_data; window=30)
rv_regime = compute_stock_rv_regime(stock_rolling_vol)
println("  Per-stock RV regime: $(length(rv_regime)) tickers")

# ── Step 4: Stream WRDS IV surface, compute Heston IV for each sample ───────

println("\n  Step 4: Streaming WRDS IV surface (sampling 1/$(SAMPLE_EVERY))...")

results = NamedTuple{(:ticker,:date,:dte,:delta,:wrds_iv,:heston_iv,:cp),
                      Tuple{String,Date,Int,Float64,Float64,Float64,String}}[]

line_count = 0
skipped_no_heston = 0
skipped_no_price = 0
matched = 0

open(WRDS_IV_FILE, "r") do f
    global line_count, skipped_no_heston, skipped_no_price, matched
    readline(f)  # skip header

    while !eof(f)
        line = readline(f)
        line_count += 1

        line_count % SAMPLE_EVERY != 0 && continue

        fields = split(line, ",")
        length(fields) < 6 && continue

        tk = String(Base.strip(fields[1]))
        tk ∉ heston_tickers && (skipped_no_heston += 1; continue)

        date_str = String(Base.strip(fields[2]))
        dte_raw = tryparse(Int, Base.strip(fields[3]))
        delta_raw = tryparse(Float64, Base.strip(fields[4]))
        iv_raw = tryparse(Float64, Base.strip(fields[5]))
        cp = String(Base.strip(fields[6]))

        (dte_raw === nothing || delta_raw === nothing || iv_raw === nothing) && continue
        dte_raw < 7 || dte_raw > 180 && continue
        iv_raw <= 0.0 || iv_raw > 5.0 && continue

        y = parse(Int, date_str[1:4])
        m = parse(Int, date_str[6:7])
        d = parse(Int, date_str[9:10])
        dt = Date(y, m, d)

        !haskey(price_lookup, tk) && (skipped_no_price += 1; continue)
        S = get(price_lookup[tk], dt, NaN)
        isnan(S) && (skipped_no_price += 1; continue)

        raw_params = lookup_heston_params(heston_ts, tk, dt)
        raw_params === nothing && (skipped_no_heston += 1; continue)

        tk_rv = get(rv_regime, tk, Dict{Date, Float64}())
        regime_scale = get(tk_rv, dt, 1.0)
        params = rv_adjusted_params(raw_params, regime_scale)

        T = dte_raw / 365.0
        abs_delta = abs(delta_raw)
        abs_delta < 0.01 || abs_delta > 0.99 && continue
        opt_type = cp == "C" ? :call : :put

        ddf = get(div_data, tk, DataFrame(ex_date=Date[], amount=Float64[]))
        q = trailing_dividend_yield(ddf, S, dt)

        K = strike_from_delta(S, T, R, sqrt(params.v0), abs_delta, opt_type; q=q)

        try
            heston_iv = heston_implied_vol(S, K, T, R, params; q=q, option_type=opt_type)
            heston_iv <= 0.0 || heston_iv > 5.0 && continue

            push!(results, (ticker=tk, date=dt, dte=dte_raw, delta=delta_raw,
                            wrds_iv=iv_raw, heston_iv=heston_iv, cp=cp))
            matched += 1
        catch
        end

        if matched % 5000 == 0 && matched > 0
            println("    $(matched) comparisons computed...")
        end
    end
end

println("\n  Streaming complete:")
println("    Total rows scanned:   $(line_count)")
println("    Sampled & matched:    $(matched)")
println("    Skipped (no Heston):  $(skipped_no_heston)")
println("    Skipped (no price):   $(skipped_no_price)")

if matched == 0
    println("\n  ⚠ No comparison points — check data alignment.")
    exit()
end

# ── Step 5: Compute error statistics ────────────────────────────────────────

println("\n  Step 5: Computing error statistics...\n")

errors = [r.heston_iv - r.wrds_iv for r in results]
abs_errors = abs.(errors)
pct_errors = abs_errors ./ max.([r.wrds_iv for r in results], 0.01) .* 100.0

println("══════════════════════════════════════════════════════")
println("  OVERALL STATISTICS  (N = $(matched))")
println("══════════════════════════════════════════════════════")
println("  Bias (mean error):     $(round(mean(errors), digits=4))")
println("  MAE:                   $(round(mean(abs_errors), digits=4))")
println("  RMSE:                  $(round(sqrt(mean(errors.^2)), digits=4))")
println("  MAPE:                  $(round(mean(pct_errors), digits=1))%")
println("  Median Abs Error:      $(round(median(abs_errors), digits=4))")
println("  Correlation:           $(round(cor([r.wrds_iv for r in results], [r.heston_iv for r in results]), digits=4))")
println("══════════════════════════════════════════════════════\n")

# ── 5a: Per-ticker breakdown ─────────────────────────────────────────────────

ticker_stats = DataFrame(
    Ticker=String[], N=Int[], Bias=Float64[], MAE=Float64[],
    RMSE=Float64[], MAPE=Float64[], Corr=Float64[]
)

all_tickers_sorted = sort(unique(r.ticker for r in results))
for tk in all_tickers_sorted
    tk_results = filter(r -> r.ticker == tk, results)
    n = length(tk_results)
    n < 5 && continue
    errs = [r.heston_iv - r.wrds_iv for r in tk_results]
    wrds = [r.wrds_iv for r in tk_results]
    hest = [r.heston_iv for r in tk_results]
    ae = abs.(errs)
    pe = ae ./ max.(wrds, 0.01) .* 100.0
    c = length(unique(wrds)) > 1 && length(unique(hest)) > 1 ? cor(wrds, hest) : NaN
    push!(ticker_stats, (tk, n, round(mean(errs), digits=4),
          round(mean(ae), digits=4), round(sqrt(mean(errs.^2)), digits=4),
          round(mean(pe), digits=1), round(c, digits=3)))
end
sort!(ticker_stats, :RMSE)

println("-- Per-Ticker Error Statistics --")
pretty_table(ticker_stats)

# ── 5b: By DTE bucket ───────────────────────────────────────────────────────

dte_buckets = [(7, 14, "7-14d"), (15, 30, "15-30d"), (31, 60, "31-60d"), (61, 90, "61-90d"), (91, 180, "91-180d")]
dte_stats = DataFrame(DTE_Bucket=String[], N=Int[], Bias=Float64[], MAE=Float64[], RMSE=Float64[], MAPE=Float64[])

for (lo, hi, label) in dte_buckets
    bucket = filter(r -> lo <= r.dte <= hi, results)
    n = length(bucket)
    n < 5 && continue
    errs = [r.heston_iv - r.wrds_iv for r in bucket]
    ae = abs.(errs)
    pe = ae ./ max.([r.wrds_iv for r in bucket], 0.01) .* 100.0
    push!(dte_stats, (label, n, round(mean(errs), digits=4),
          round(mean(ae), digits=4), round(sqrt(mean(errs.^2)), digits=4),
          round(mean(pe), digits=1)))
end

println("\n-- Error by DTE Bucket --")
pretty_table(dte_stats)

# ── 5c: By delta bucket (moneyness) ─────────────────────────────────────────

delta_buckets = [
    (0.05, 0.15, "Deep OTM (5-15Δ)"),
    (0.15, 0.30, "OTM (15-30Δ)"),
    (0.30, 0.45, "Slight OTM (30-45Δ)"),
    (0.45, 0.55, "ATM (45-55Δ)"),
    (0.55, 0.70, "Slight ITM (55-70Δ)"),
    (0.70, 0.95, "ITM (70-95Δ)")
]
delta_stats = DataFrame(Delta_Bucket=String[], N=Int[], Bias=Float64[], MAE=Float64[], RMSE=Float64[], MAPE=Float64[])

for (lo, hi, label) in delta_buckets
    bucket = filter(r -> lo <= abs(r.delta) <= hi, results)
    n = length(bucket)
    n < 5 && continue
    errs = [r.heston_iv - r.wrds_iv for r in bucket]
    ae = abs.(errs)
    pe = ae ./ max.([r.wrds_iv for r in bucket], 0.01) .* 100.0
    push!(delta_stats, (label, n, round(mean(errs), digits=4),
          round(mean(ae), digits=4), round(sqrt(mean(errs.^2)), digits=4),
          round(mean(pe), digits=1)))
end

println("\n-- Error by Delta Bucket (Moneyness) --")
pretty_table(delta_stats)

# ── 5d: Put vs Call ──────────────────────────────────────────────────────────

cp_stats = DataFrame(Type=String[], N=Int[], Bias=Float64[], MAE=Float64[], RMSE=Float64[])
for cp_flag in ["P", "C"]
    bucket = filter(r -> r.cp == cp_flag, results)
    n = length(bucket)
    n < 5 && continue
    errs = [r.heston_iv - r.wrds_iv for r in bucket]
    ae = abs.(errs)
    push!(cp_stats, (cp_flag == "P" ? "Put" : "Call", n, round(mean(errs), digits=4),
          round(mean(ae), digits=4), round(sqrt(mean(errs.^2)), digits=4)))
end

println("\n-- Put vs Call --")
pretty_table(cp_stats)

# ── Step 6: Save full comparison dataset ────────────────────────────────────

out_df = DataFrame(
    ticker = [r.ticker for r in results],
    date = [r.date for r in results],
    dte = [r.dte for r in results],
    delta = [round(r.delta, digits=4) for r in results],
    wrds_iv = [round(r.wrds_iv, digits=6) for r in results],
    heston_iv = [round(r.heston_iv, digits=6) for r in results],
    error = [round(r.heston_iv - r.wrds_iv, digits=6) for r in results],
    abs_error = [round(abs(r.heston_iv - r.wrds_iv), digits=6) for r in results],
    cp = [r.cp for r in results]
)

out_path = joinpath(_PATH_TO_DATA, "iv_validation_results.csv")
CSV.write(out_path, out_df)
println("\n  -> Full comparison saved to $(out_path) ($(nrow(out_df)) rows)")

# ── Step 7: Scatter plot ────────────────────────────────────────────────────

println("\n  Step 7: Generating scatter plot...")
try
    wrds_vals = [r.wrds_iv for r in results]
    hest_vals = [r.heston_iv for r in results]

    sample_idx = if length(wrds_vals) > 5000
        sort(randperm(length(wrds_vals))[1:5000])
    else
        1:length(wrds_vals)
    end

    p = scatter(wrds_vals[sample_idx], hest_vals[sample_idx];
                xlabel="WRDS Market IV", ylabel="Heston Model IV",
                title="Heston IV vs Market IV (N=$(matched), corr=$(round(cor(wrds_vals, hest_vals), digits=3)))",
                label="", alpha=0.15, ms=2, color=:steelblue,
                size=(800, 600), dpi=150)

    max_iv = max(maximum(wrds_vals[sample_idx]), maximum(hest_vals[sample_idx]))
    plot!(p, [0, max_iv], [0, max_iv]; label="Perfect fit", color=:red, lw=2, ls=:dash)

    plot_path = joinpath(_PATH_TO_DATA, "iv_validation_scatter.png")
    savefig(p, plot_path)
    println("  -> Scatter plot saved to $(plot_path)")
catch e
    @warn "Could not generate plot: $e"
end

println("\n  Done! Heston IV validation complete.")
