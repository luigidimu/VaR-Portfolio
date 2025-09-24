# 📊 Value-at-Risk (VaR) — Multi-Asset, Interactive

Interactive Python tool to compute 1-day VaR for multi-asset portfolios (equities, ETFs, commodity futures, FX, crypto) using Yahoo Finance data.
All series are converted to a base currency you choose (default EUR), then four VaR methods are compared on two horizons (2024→today and 2015→today).

---

## ✅ What the code does

- Downloads prices for user-provided tickers.
- Converts every series to the chosen base currency via Yahoo FX.
- Aligns data to a common calendar (Business Days by default).
- Computes portfolio log-returns from user amounts/weights.
- Calculates VaR 95% & 99% with four methods:
  - Parametric (Normal)
  - Historical Simulation
  - Monte Carlo (Normal multivariate)
  - Monte Carlo (t-Student multivariate) with df estimated from excess kurtosis
- Produces charts with VaR lines, CSV summary, and plain-language messages.
---
## 🛠️ How it was built (step-by-step)

Input (CLI, interactive).
1. Ask user for: tickers, base currency (ISO-3), amounts in base currency.
2. Download prices.
   1. yfinance.download(..., auto_adjust=True) to get close prices (dividends/splits adjusted).
   2. Detect quote currency.
   3. For each ticker, read fast_info["currency"].
   4. FX conversion to base.
   5. If quote currency ≠ base, download the FX pair (BASEQUOTE=X) and convert price: -> price_in_base = price_in_quote / FX(BASE/QUOTE).
3. Calendar alignment.
4. Reindex to Business Days (“B”) and forward-fill gaps (crypto are 24/7, others aren’t).
5. Portfolio returns.
6. Log-returns per asset → weight by user amounts → single portfolio return series.
7. VaR engines.
   1. Parametric: VaRα = −(μ + zα·σ)·Capital
   2. Historical: empirical α-quantile of returns (left tail)
   3. MC Normal: draw from N(μ, Σ) preserving historical covariance
   4. MC t-Student: scale-mixture t_df(μ, Σ); df estimated from portfolio excess kurtosis
8. Outputs.
Save VaR table (CSV), render 8 charts (method × horizon), print human-readable messages.
---
## 📚 Libraries used (and why)

- yfinance – free Yahoo Finance access (prices & FX).
- pandas – time-series handling, joins, resampling, CSV export.
- numpy – vector math, random draws, percentiles.
- scipy.stats – Normal PDF/quantiles; chi-square for t-mixture.
- matplotlib – histograms + VaR lines for visual inspection.

---
## 🚀 Quick start

Python 3.9+

pip install numpy pandas matplotlib scipy yfinance
python code/var_interactive_multiasset.py

You will be prompted for:

Tickers (comma-separated):
- e.g., NVDA,AAPL,TLT,GLD,BTC-USD,EURUSD=X
- Base currency (ISO-3): EUR (default), USD, GBP, …
- Amounts in base currency (same order as tickers): e.g., 2000,1500,1000,500,300,200
---
## 📤 Outputs

CSV summary → outputs/var_comparison_interactive_multiasset.csv

(VaR 95/99 by method & horizon on your total capital)

Charts → plots/
Console messages (interpretation):

“In the worst 1% of days, you could lose more than €X in one day.”

---
## 🧾 Supported instruments (Yahoo tickers)

Equities / ETFs: AAPL, TLT, EUNL.DE, IWDA.AS, ISP.MI, …

Commodity futures (continuous): CL=F (WTI), GC=F (Gold), SI=F, NG=F, ZC=F, …

FX: EURUSD=X, USDJPY=X, …

Crypto: BTC-USD, ETH-USD, SOL-USD, …

⚠️ Use the correct exchange suffix: .MI (Borsa Italiana), .L (LSE), .DE (Xetra), .PA (Paris), .AS (Amsterdam), etc.

---
## 🧩 Troubleshooting

“FX not found …”
Ensure you typed an ISO-3 base currency (e.g., EUR) and a valid ticker; EUR-quoted assets don’t need FX.

Ticker not found / delisted
Verify spelling and the exchange suffix (.MI, .L, .DE, …).

Weird weekend behavior
Data are aligned to Business Days. To include weekends, set CALENDAR_FREQ = "D" in the script.

I typed amounts where base currency was requested
Re-run and enter just the ISO-3 code (e.g., EUR) at that prompt.

---
## 📝 Notes & Disclaimer

Data source: Yahoo Finance, auto-adjusted prices (splits/dividends).

VaR is a statistical estimate; extreme losses can exceed VaR. Educational use only.

Crypto are 24/7; downsampled to business days by default for comparability.

---
## 📫 Contact
- LinkedIn: [Luigi Di Muzio](https://linkedin.com/in/luigidimuzio)  
- Email: [luigidimu@gmail.com](mailto:luigidimu@gmail.com)
