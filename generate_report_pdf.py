"""Generate Wheel Strategy System Report as PDF using fpdf2."""
import os
from fpdf import FPDF

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120)
            self.cell(0, 5, "Wheel Strategy ETF Fund - System Report 2025", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, num, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(26, 54, 93)
        self.ln(4)
        self.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(43, 108, 176)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(45, 55, 72)
        self.ln(2)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def formula_box(self, text):
        self.set_fill_color(235, 248, 255)
        self.set_draw_color(49, 130, 206)
        self.set_font("Courier", "", 9)
        self.set_text_color(30)
        x = self.get_x()
        self.rect(x, self.get_y(), 190, 5 + text.count("\n") * 4.5 + 4, style="DF")
        self.set_x(x + 4)
        self.ln(2)
        for line in text.split("\n"):
            self.set_x(x + 4)
            self.cell(0, 4.5, line, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def warning_box(self, text):
        self.set_fill_color(255, 255, 240)
        self.set_draw_color(214, 158, 46)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 100, 0)
        w = 190
        lines = text.split("\n")
        h = 5 + len(lines) * 4
        self.rect(10, self.get_y(), w, h, style="DF")
        self.ln(1)
        for line in lines:
            self.set_x(14)
            self.cell(0, 4, line, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(45, 55, 72)
        self.set_text_color(255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8)
        self.set_text_color(30)
        for ri, row in enumerate(rows):
            if ri % 2 == 0:
                self.set_fill_color(247, 250, 252)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 5, str(cell), border=1, fill=True)
            self.ln()
        self.ln(2)

    def add_chart(self, filename, caption):
        path = os.path.join(DATA, filename)
        if not os.path.exists(path):
            self.body_text(f"[Chart not found: {filename}]")
            return
        if self.get_y() > 200:
            self.add_page()
        self.image(path, x=10, w=190)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(113, 128, 150)
        self.cell(0, 5, caption, new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(3)


pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ── Title Page ──
pdf.add_page()
pdf.ln(50)
pdf.set_font("Helvetica", "B", 32)
pdf.set_text_color(26, 54, 93)
pdf.cell(0, 15, "Wheel Strategy ETF Fund", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Helvetica", "", 18)
pdf.set_text_color(74, 85, 104)
pdf.cell(0, 10, "Complete System Report - 2025 Backtest", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(20)
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(113, 128, 150)
for line in [
    "Cornell University - CHEME-5660 Final Project",
    "Framework: Julia | Pricing: CRR American Binomial Lattice",
    "Initial NAV: $600,000,000 | Universe: 35 Names",
    "Report Generated: March 2026",
]:
    pdf.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")

# ── Section 1: Architecture ──
pdf.add_page()
pdf.section_title(1, "System Architecture")
pdf.body_text("The system consists of 8 Julia source modules + 2 entry-point scripts, loaded in dependency order by Include.jl.")
pdf.add_table(
    ["File", "Role"],
    [
        ["Include.jl", "Environment setup: activate project, load packages, include modules"],
        ["src/Files.jl", "Data loaders: JLD2 market data, SAGBM CSV, Finviz CSV"],
        ["src/DataDownload.jl", "Yahoo Finance API: prices & dividends with CSV caching"],
        ["src/Compute.jl", "Log-growth matrix, rolling vol, div yield, IV calibration, correlation"],
        ["src/OptionPricing.jl", "CRR American option pricing, Greeks, strike_from_delta, IV solver"],
        ["src/EarningsCalendar.jl", "Earnings date mgmt: CSV/estimation/Yahoo, near_earnings detection"],
        ["src/MonteCarloSim.jl", "GBM, regime-switching GBM, correlated GBM, HMM, stress testing"],
        ["src/WheelEngine.jl", "Core backtest engine: state machine, roll, repair, risk overlays, NAV"],
        ["Backtest.jl", "BACKTEST ENTRY: universe -> data -> simulation -> reporting"],
        ["Wheel Strategy Code", "LIVE TEMPLATE: daily signal generation for real-time ops"],
    ],
    [40, 150],
)

# ── Section 2: Data Pipeline ──
pdf.section_title(2, "Data Pipeline")
pdf.sub_title("Data Sources")
pdf.add_table(
    ["Source", "Description", "Loader"],
    [
        ["JLD2 Market Data", "S&P 500 2025 daily OHLC (Polygon.io)", "Files.jl -> MyTestingMarketDataSet()"],
        ["SAGBM CSV", "Pre-computed drift/vol per ticker (2014-2024)", "Files.jl -> load_sagbm_parameters()"],
        ["Finviz CSV", "Div yield, market cap, volume", "Files.jl -> load_finviz_screener()"],
        ["Yahoo Finance Prices", "Daily OHLC + adj_close", "DataDownload.jl -> download_price_data()"],
        ["Yahoo Finance Divs", "Ex-dividend dates + amounts", "DataDownload.jl -> download_dividends()"],
        ["VIX", "VIX index daily (reference)", "DataDownload.jl"],
        ["SPY", "S&P 500 ETF for benchmark", "DataDownload.jl"],
    ],
    [38, 90, 62],
)

# ── Section 3: Universe Construction ──
pdf.section_title(3, "Universe Construction (Varner PDF Sec.3)")
pdf.sub_title("Two-Sleeve Portfolio")
pdf.add_table(
    ["Sleeve", "# Names", "Allocation", "Per-Name Wt", "Selection Criteria"],
    [
        ["Safe", "25", "60%", "2.4%", "Low vol, high div yield (>=3%), deep options"],
        ["Aggressive", "10", "40%", "4.0%", "Top-quintile vol, deep options, sector diversity"],
    ],
    [25, 18, 22, 22, 103],
)
pdf.body_text("Initial NAV: $600,000,000 | Per-name cap: <=5% NAV | Sector cap: <=25%")

pdf.sub_title("Two-Block Inventory per Ticker")
pdf.add_table(
    ["Block", "Allocation", "Purpose"],
    [
        ["Block A (Hold)", "50%", "Buy-and-hold for dividends + capital appreciation. No options."],
        ["Block B (Wheel)", "50%", "Wheel cycle: Sell Put->Assigned->Sell Call->Called Away"],
    ],
    [35, 25, 130],
)

pdf.sub_title("Safe Tickers (25)")
pdf.body_text("PEP, KO, PG, JNJ, CME, CMCSA, VZ, T, IBM, MO, PM, MDLZ, EXC, KMB, PAYX, TROW, PFG, SO, DUK, ED, LNT, GIS, CAG, REG, CPB")
pdf.sub_title("Aggressive Tickers (10)")
pdf.body_text("TSLA, NVDA, AMD, AAPL, MSFT, AMZN, GOOG, META, NFLX, DVN")

# ── Section 4: Preprocessing Calculations ──
pdf.add_page()
pdf.section_title(4, "Preprocessing Calculations")

pdf.sub_title("4.1 Log Growth Matrix (CHEME-5660 Week 5b)")
pdf.formula_box("mu_{t,t-1}(r_f) = (1/dt) * ln(S_t / S_{t-1}) - r_f\nwhere dt = 1/252, r_f = risk-free rate")
pdf.body_text("Computes excess log returns for all tickers from JLD2 dataset. Output: (days x tickers) matrix for cross-sectional volatility estimation.")

pdf.sub_title("4.2 30-Day Rolling Realized Volatility")
pdf.warning_box("SELF-DESIGNED - no direct course reference")
pdf.formula_box("sigma_RV(t) = std(r_{t-29}, ..., r_t) * sqrt(252)")
pdf.body_text("Annualized rolling volatility using trailing 30-day window of daily log returns. Floor at 1%.")

pdf.sub_title("4.3 Dividend Yield")
pdf.formula_box("q = (sum of dividends) / (avg price) / lookback_years")
pdf.body_text("Continuous dividend yield passed to CRR lattice for accurate American option valuation around ex-dividend dates.")

pdf.sub_title("4.4 IV Calibration (VRP Model)")
pdf.warning_box("SELF-DESIGNED - Entire IV calibration module is original work.\nAcademic motivation: Carr & Wu (2009), Bollerslev, Tauchen & Zhou (2009).\nOur specific parametric model (VRP x term x skew x volvol) is original.")
pdf.formula_box("sigma_IV = sigma_RV x VRP x (1 + term_adj) x (1 + skew_adj) x (1 + volvol_adj)\n\nVRP multiplier:  Safe = 1.15x,  Aggressive = 1.25x\nterm_adj  = term_slope * max(0, (30/365 - T) / (30/365))\nskew_adj  = skew_slope * max(0, -moneyness)\nvolvol_adj = vol_of_vol * max(0, sigma_RV - 0.30)")
pdf.body_text("Data needed for better calibration: (1) Real options chain bid/ask -> extract actual IV; (2) VIX historical data -> time-varying VRP; (3) CBOE DataShop / OptionMetrics for IV surfaces.")

pdf.sub_title("4.5 Return Correlation Matrix (CHEME-5660 Week 6)")
pdf.formula_box("rho_{ij} = cor(r_i, r_j) on pairwise-complete observations\nZ_corr = L * Z_indep,  L = cholesky(rho).L")

# ── Section 5: Option Pricing ──
pdf.add_page()
pdf.section_title(5, "Option Pricing Model")

pdf.sub_title("5.1 CRR Binomial Lattice - American (CHEME-5660 Week 10)")
pdf.formula_box("dt = T / N\nu = exp(sigma * sqrt(dt)),  d = 1/u\np = (exp((r-q)*dt) - d) / (u - d)    risk-neutral prob with div yield q\n\nBackward induction (American):\nV_j = max(intrinsic_j,  exp(-r*dt) * [p*V_{j+1} + (1-p)*V_j])")
pdf.body_text("N = 50 steps. All pricing uses American exercise exclusively. European pricing is never used because the Wheel trades American-style equity options.")

pdf.sub_title("5.2 Greeks (CHEME-5660 Weeks 11-12)")
pdf.add_table(
    ["Greek", "Method", "Formula"],
    [
        ["Delta", "CRR first level", "D = (V_u - V_d) / (S*u - S*d)"],
        ["Gamma", "Central finite diff", "G = (V(S+e) - 2V(S) + V(S-e)) / e^2"],
        ["Theta", "Forward diff", "Th = V(T - 1/365) - V(T)"],
        ["Vega", "Central finite diff", "v = (V(s+0.01) - V(s-0.01)) / 0.02"],
    ],
    [25, 45, 120],
)

pdf.sub_title("5.3 strike_from_delta (SELF-DESIGNED)")
pdf.warning_box("SELF-DESIGNED - no course or textbook reference")
pdf.body_text("Input: target |Delta| (e.g. 0.25), S, sigma, T. Method: Binary search over K in [0.5S, 2.0S], compute CRR delta at each midpoint, converge when K_hi - K_lo < $0.005.")

pdf.sub_title("5.4 estimate_implied_vol (SELF-DESIGNED)")
pdf.warning_box("SELF-DESIGNED - CRR-based alternative to BSM IV solver")
pdf.body_text("Input: market option price, S, K, T. Method: Binary search over sigma in [0.01, 3.0], price via CRR at each midpoint, converge when sigma_hi - sigma_lo < 1e-6.")

# ── Section 6: Backtest Engine ──
pdf.add_page()
pdf.section_title(6, "Backtest Engine")

pdf.sub_title("6.1 State Machine per Ticker")
pdf.formula_box("SELLING_PUT -> (put assigned: S < K) -> HOLDING_SHARES -> (call ITM: S > K) -> SELLING_PUT")
pdf.body_text("Each ticker's Block B has 1-3 ladder slots, each running an independent Wheel cycle for temporal diversification of expiry dates.")

pdf.sub_title("6.2 Daily Simulation Loop")
pdf.add_table(
    ["Step", "Action", "Ref"],
    [
        ["1", "Deduct daily mgmt fee (0.68% annual / 252)", "PDF S5"],
        ["2", "Compute trailing VaR/ES -> throttle if exceeded", "PDF S4,S6"],
        ["3", "Compute drawdown -> feed into adaptive delta", "PDF S3"],
        ["4", "Per ticker x slot: deduct borrow cost (0.5% ann)", "PDF S7A"],
        ["5", "Check dividends -> credit Block A + B held shares", "PDF S2"],
        ["6", "Check option expiry -> assignment or call-away", "PDF S2"],
        ["7", "Check 4 roll triggers -> roll if triggered", "PDF S3"],
        ["8", "If no option: check risk/earn/caps -> open_option!", "PDF S3-6"],
        ["9", "Check cost-basis repair (loss >=10% -> avg down)", "PDF S6"],
        ["10", "Record DailyRecord: NAV, Greeks, cumulative totals", "-"],
    ],
    [12, 130, 48],
)

pdf.sub_title("6.3 Roll Triggers (4 Conditions)")
pdf.body_text("1. Time-based: DTE in [3,5] and OTM\n2. Premium decay: current value / original < 20% (80% captured)\n3. Moneyness band: |moneyness| > 5%\n4. Breakeven breach: price beyond BE +/- 2%")

pdf.sub_title("6.4 Trading Cost Model")
pdf.add_table(
    ["Component", "Amount"],
    [
        ["Commission", "$0.65 / contract"],
        ["Exchange fee", "$0.03 / contract"],
        ["Clearing fee", "$0.02 / contract"],
        ["Bid-ask slippage", "Volatility-adjusted spread model"],
        ["Borrow fee", "0.5% annual on held shares"],
        ["Management fee", "0.68% annual on NAV"],
    ],
    [60, 130],
)

pdf.sub_title("6.5 Earnings Policy")
pdf.add_table(
    ["Policy", "Behavior"],
    [
        [":avoid", "Skip opening options within +/-5 days of earnings"],
        [":widen", "Open but reduce delta (more OTM)"],
        [":reduce_size", "Open but reduce contracts to 50%"],
    ],
    [30, 160],
)

# ── Section 7: Monte Carlo ──
pdf.add_page()
pdf.section_title(7, "Monte Carlo & Stress Testing")

pdf.sub_title("7.1 GBM (CHEME-5660 Week 5b)")
pdf.formula_box("S_{t+dt} = S_t * exp[(mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z],  Z ~ N(0,1)")

pdf.sub_title("7.2 Correlated Multi-Asset GBM (CHEME-5660 Week 6)")
pdf.formula_box("Z_corr = L * Z_indep,  L = cholesky(rho).L\nEach asset j: S_j(t+dt) = S_j(t) * exp[(mu_j - sigma_j^2/2)*dt + sigma_j*sqrt(dt)*Z_corr_j]")

pdf.sub_title("7.3 Regime-Switching GBM (SELF-DESIGNED)")
pdf.body_text("2-state Markov chain switching between normal (mu1, sigma1) and stressed (mu2, sigma2) regimes with transition probabilities p12 and p21. Motivated by Varner PDF Section 7B.")

pdf.sub_title("7.4 HMM 2-State (CHEME-5660 Week 13)")
pdf.formula_box("EM (Baum-Welch): Forward (alpha) + Backward (beta) -> gamma, xi\nM-step: Update mu_k, sigma^2_k, transition matrix A\nOutput: (mu_normal, sigma_normal, mu_stressed, sigma_stressed, p12, p21)")

pdf.sub_title("7.5 Stress Scenarios")
pdf.add_table(
    ["Scenario", "Vol Mult", "Drift Adj", "Gap"],
    [
        ["Normal", "1.0x", "0", "-"],
        ["Vol Spike", "2.0x", "0", "-"],
        ["Bear Market", "1.5x", "-20%", "-"],
        ["Flash Crash", "2.0x", "0", "-10% d30"],
        ["Name Blowup", "1.5x", "0", "-30% d60"],
        ["Bull Squeeze", "1.5x", "+30%", "+15% d45"],
    ],
    [50, 30, 30, 80],
)

# ── Section 8: Charts ──
pdf.add_page()
pdf.section_title(8, "Results & Charts")

charts = [
    ("nav_curve_2025.png", "Chart 1: NAV Curve vs SPY Benchmark"),
    ("drawdown_2025.png", "Chart 2: Drawdown from Peak (Wheel vs SPY)"),
    ("income_decomposition_2025.png", "Chart 3: Cumulative Income Decomposition (Premium + Divs - Costs)"),
    ("greeks_delta_2025.png", "Chart 4: Portfolio Delta over Time"),
    ("greeks_gamma_2025.png", "Chart 5: Portfolio Gamma over Time"),
    ("greeks_vega_2025.png", "Chart 6: Portfolio Vega over Time"),
    ("nav_composition_2025.png", "Chart 7: NAV Composition (Cash / Block A / Block B)"),
    ("return_distribution_2025.png", "Chart 8: Daily Return Distribution (Wheel vs SPY)"),
    ("rolling_sharpe_2025.png", "Chart 9: Rolling 60-Day Sharpe Ratio"),
    ("premium_by_ticker_2025.png", "Chart 10: Top 15 Tickers by Premium Income"),
    ("option_mtm_2025.png", "Chart 11: Short Option Mark-to-Market Liability"),
    ("monthly_returns_2025.png", "Chart 12: Monthly Returns"),
]

for fname, caption in charts:
    if pdf.get_y() > 170:
        pdf.add_page()
    pdf.add_chart(fname, caption)

# ── Section 9: Output Files ──
pdf.add_page()
pdf.section_title(9, "Output Files")

pdf.sub_title("CSV Data Files")
pdf.add_table(
    ["File", "Key Columns", "Purpose"],
    [
        ["daily_nav_2025.csv", "Date,NAV,Cash,Delta,Gamma,Vega,DailyReturn,SPY...", "Day-level full dataset for downstream analysis"],
        ["ticker_performance_2025.csv", "Ticker,Premium,Dividends,Costs,Assigns,Trades...", "Per-ticker attribution and performance"],
    ],
    [45, 85, 60],
)

pdf.sub_title("Chart Files (12 PNGs)")
pdf.add_table(
    ["#", "File", "Analysis Purpose"],
    [
        ["1", "nav_curve_2025.png", "Core performance: did strategy beat benchmark?"],
        ["2", "drawdown_2025.png", "Risk: max drawdown depth & recovery"],
        ["3", "income_decomposition_2025.png", "Income source decomposition, Distribution Yield"],
        ["4-6", "greeks_*.png", "Directional, convexity, volatility risk monitoring"],
        ["7", "nav_composition_2025.png", "Asset allocation visualization over time"],
        ["8", "return_distribution_2025.png", "Return profile characterization"],
        ["9", "rolling_sharpe_2025.png", "Time-varying risk-adjusted performance"],
        ["10", "premium_by_ticker_2025.png", "Per-name contribution attribution"],
        ["11", "option_mtm_2025.png", "Outstanding short option exposure"],
        ["12", "monthly_returns_2025.png", "Monthly performance review"],
    ],
    [10, 60, 120],
)

# ── Section 10: Course Reference ──
pdf.add_page()
pdf.section_title(10, "Course Reference Mapping")

pdf.sub_title("CHEME-5660 Content Implemented")
pdf.add_table(
    ["Week", "Topic", "Implementation"],
    [
        ["5b", "GBM / SAGBM parameter estimation", "log_growth_matrix(), simulate_gbm()"],
        ["6", "Multi-asset GBM + Cholesky", "simulate_correlated_gbm(), compute_return_correlation()"],
        ["10", "CRR Binomial - American Options", "crr_price() with dividend yield q"],
        ["11", "Greeks (D, G, Th, v)", "crr_delta/gamma/theta/vega(), option_greeks()"],
        ["12b", "Delta and hedging", "crr_delta() -> strike_from_delta()"],
        ["13", "HMM - Markov Models", "fit_two_state_hmm(), classify_regime()"],
    ],
    [15, 65, 110],
)

pdf.sub_title("Self-Designed Components (not from course)")
pdf.add_table(
    ["Component", "Description"],
    [
        ["IV Calibration (VRP)", "sigma_RV->sigma_IV via Variance Risk Premium+term+skew+volvol"],
        ["strike_from_delta", "Bisection search: find K producing target |Delta| under CRR"],
        ["estimate_implied_vol", "Bisection on CRR to find sigma matching market price"],
        ["Bid-ask spread model", "Volatility-adjusted half-spread estimation"],
        ["Fill rate model", "Spread-based fill rate with vol penalty"],
        ["Regime-switching GBM", "2-state Markov x GBM for regime-aware simulation"],
        ["Earnings calendar", "SEC filing pattern-based quarterly date estimation"],
        ["Cost-basis repair", "Automated average-down when shares >=10% underwater"],
        ["Adaptive tenor/delta", "Vol-regime-aware selection of tenor and delta target"],
        ["Rolling IV computation", "Convert time-series sigma_RV to sigma_IV for all tickers"],
    ],
    [48, 142],
)

# ── Section 11: Improvements ──
pdf.add_page()
pdf.section_title(11, "Improvements & Future Work")

pdf.sub_title("Priority Improvements")
pdf.add_table(
    ["Priority", "Improvement", "Current State", "Method"],
    [
        ["HIGH", "IS/OOS + Walk-Forward", "Only in-sample 2025", "2014-2024 train, 2025 OOS. PDF S7A required."],
        ["HIGH", "Integrate HMM to backtest", "HMM done, not integrated", "Daily classify_regime -> adjust delta/tenor"],
        ["HIGH", "Real IV Data", "Fixed VRP multipliers", "CBOE/OptionMetrics/broker API for IV surface"],
        ["MED", "Correlated Stress Test", "Per-ticker / uniform", "simulate_correlated_gbm + correlations"],
        ["MED", "Execution Realism", "Close-price execution", "Open/VWAP/next-day-open fill simulation"],
        ["MED", "Dynamic Rebalance", "Weights fixed", "Monthly/quarterly rebalance Block A/B"],
        ["LOW", "Heston Model", "Rolling realized vol", "Replace GBM with stochastic vol"],
        ["LOW", "TWAP/VWAP Execution", "Fixed slippage", "Real L2 data + optimal execution"],
        ["LOW", "Live Broker API", "Template only", "Interactive Brokers TWS API (IBApi.jl)"],
    ],
    [22, 42, 48, 78],
)

pdf.sub_title("Varner PDF Requirements vs Implementation")
pdf.add_table(
    ["Section", "Requirement", "Status"],
    [
        ["S2", "Wheel cycle: Short Put -> Assigned -> CC -> Called Away", "DONE"],
        ["S3", "Safe/Aggressive, 5% name cap, sector cap, laddering", "DONE"],
        ["S4", "Distribution Yield, Premium Capture, Sharpe, VaR/ES", "DONE"],
        ["S5", "Commissions, exchange fees, borrow fees", "DONE"],
        ["S6", "Delta range, tenor, laddering, earnings, roll, repair", "DONE"],
        ["S7A", "IS/OOS split, walk-forward", "NOT DONE"],
        ["S7A", "Parameter sweep", "DONE (flag)"],
        ["S7B", "Regime-switching GBM, gap, vol spike", "DONE (flag)"],
        ["S7B", "IV surface calibration", "PARTIAL (VRP model)"],
    ],
    [20, 110, 60],
)

# ── Save ──
out_path = os.path.join(ROOT, "Wheel_Strategy_Report_2025.pdf")
pdf.output(out_path)
print(f"PDF saved to: {out_path}")
