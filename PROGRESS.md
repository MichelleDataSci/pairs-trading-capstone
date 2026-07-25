# Project Progress — Pairs Trading Analytics
**Last updated:** 2026-05-09
**Status:** Phase 2 complete. Phase 3 next.

---

## 1. Project Overview and Goals

This project builds an end-to-end quantitative pair trading strategy for 6 large-cap S&P 500 technology stocks. The goal is to identify cointegrated stock pairs, develop a mean-reversion trading strategy around the spread between each pair, overlay a sentiment filter, and surface everything through an interactive web dashboard.

**Universe:** MSFT, GOOGL, NVDA, AAPL, AMZN, META
**Data period:** 2018-01-01 to 2025-12-31 (8 years, ~2,009 trading days)
**Benchmarks:** S&P 500 (^GSPC), VIX (^VIX)

### Five-phase structure

| Phase | File(s) | Goal | Status |
|-------|---------|------|--------|
| 1 | `phase1_data.py` | Master data creation — download, clean, merge | COMPLETE |
| 2 | `phase2_eda.py` | Exploratory data analysis — 8 charts + 2 reports | COMPLETE |
| 3 | `phase3_cointegration.py`, `phase3_strategy.py` | Cointegration screening + backtesting engine | NOT STARTED |
| 4 | `phase4_sentiment.py` | News sentiment scoring + strategy overlay | NOT STARTED |
| 5 | `app/app.py` | Streamlit interactive dashboard | STUB ONLY |

---

## 2. Phase 1 — Master Data Creation (COMPLETE)

**Script:** `src/phase1_data.py`

### What was built

A five-step pipeline that:
1. Downloads OHLC + Volume for all 6 tickers from yfinance and saves each as a raw CSV
2. Computes daily returns using the formula `(Xt - Xt-1) / Xt-1 * 100` from adjusted close prices
3. Downloads S&P 500 (^GSPC) and VIX (^VIX) and computes their daily returns
4. Adds `Year`, `Quarter`, and `Month` integer columns as time indicators
5. Merges all series into a single master CSV, dropping the first NaN row from the return shift

### Files created

**`data/raw/`** — 8 raw OHLC CSVs (one per ticker + benchmarks):
- `MSFT_raw.csv`, `GOOGL_raw.csv`, `NVDA_raw.csv`, `AAPL_raw.csv`, `AMZN_raw.csv`, `META_raw.csv`
- `sp500_raw.csv`, `vix_raw.csv`
- Each file: 2,010 rows, columns: Open, High, Low, Close, Volume

**`data/processed/master_data.csv`** — single merged file:

| Property | Value |
|----------|-------|
| Rows | 2,009 trading days |
| Columns | 11 |
| Date range | 2018-01-03 to 2025-12-30 |
| Null values | 0 |

**Columns in master_data.csv:**

| Column | Description |
|--------|-------------|
| `MSFT_return` | Microsoft daily return (%) |
| `GOOGL_return` | Alphabet daily return (%) |
| `NVDA_return` | NVIDIA daily return (%) |
| `AAPL_return` | Apple daily return (%) |
| `AMZN_return` | Amazon daily return (%) |
| `META_return` | Meta daily return (%) |
| `sp500_return` | S&P 500 daily return (%) |
| `vix_return` | VIX daily return (%) |
| `Year` | Calendar year integer |
| `Quarter` | Quarter integer (1–4) |
| `Month` | Month integer (1–12) |

---

## 3. Phase 2 — Exploratory Data Analysis (COMPLETE)

**Script:** `src/phase2_eda.py`

### Charts produced (`outputs/charts/`)

| File | Description |
|------|-------------|
| `hist_returns.png` | 2x3 grid of return distributions with KDE overlay, one panel per stock |
| `correlation_heatmap.png` | 6x6 annotated Pearson correlation heatmap of daily returns |
| `cumulative_returns.png` | Base-100 cumulative return index for all 6 stocks (2018-2025) |
| `rolling_volatility.png` | 30-day rolling annualised volatility for all 6 stocks |
| `boxplot_by_year.png` | 2x3 grid of annual return boxplots (2018-2025), one panel per stock |
| `pairwise_price_scatter.png` | 3x5 grid of all 15 price-level scatter plots with regression R values |
| `beta_analysis.png` | Bar chart of each stock's beta vs S&P 500 |

### Reports produced (`outputs/reports/`)

| File | Description |
|------|-------------|
| `summary_stats.csv` | Mean, std, min, max, skewness, kurtosis per ticker |
| `beta_analysis.csv` | Beta, alpha, R-squared, p-value, std error vs S&P 500 per ticker |
| `phase2_eda_report.md` | Full written analytical report — findings and Phase 3 implications |

### Key findings

#### Return statistics

| Ticker | Mean (%) | Std (%) | Min (%) | Max (%) | Skewness | Ex. Kurtosis |
|--------|----------|---------|---------|---------|----------|--------------|
| MSFT   | 0.107    | 1.786   | -14.74  | 14.22   | +0.08    | 7.09         |
| GOOGL  | 0.107    | 1.952   | -11.63  | 10.22   | -0.01    | 3.86         |
| NVDA   | 0.233    | 3.232   | -18.76  | 24.37   | +0.13    | 4.84         |
| AAPL   | 0.114    | 1.940   | -12.86  | 15.33   | +0.15    | 6.45         |
| AMZN   | 0.091    | 2.168   | -14.05  | 13.54   | +0.10    | 4.26         |
| META   | 0.100    | 2.612   | -26.39  | 23.28   | -0.33    | 18.12        |

#### Beta vs S&P 500

| Ticker | Beta  | Alpha (daily %) | R-squared |
|--------|-------|-----------------|-----------|
| NVDA   | 1.822 | 0.134           | 0.485     |
| META   | 1.318 | 0.028           | 0.388     |
| AAPL   | 1.216 | 0.048           | 0.599     |
| AMZN   | 1.178 | 0.027           | 0.450     |
| MSFT   | 1.176 | 0.043           | 0.661     |
| GOOGL  | 1.156 | 0.045           | 0.535     |

All p-values = 0.000 (all betas are statistically significant).

#### Return correlations (ranked)

Highest: MSFT/GOOGL (0.709), MSFT/AAPL (0.700), MSFT/AMZN (0.682)
Lowest: NVDA/META (0.525), AAPL/META (0.533), NVDA/AAPL (0.572)

#### Cumulative returns (2018-2025)

NVDA: +3,705% | AAPL: +577% | MSFT: +517% | GOOGL: +489% | AMZN: +291% | META: +270%

#### Rolling volatility (mean annualised)

NVDA: 48.3% | META: 38.2% | AMZN: 32.4% | GOOGL: 29.6% | AAPL: 28.3% | MSFT: 26.0%

#### Priority pairs for Phase 3 (from EDA)

Based on return correlation, price-level R², beta similarity, and visual co-movement:

1. **MSFT / GOOGL** — highest return correlation (0.709), similar betas (1.176 vs 1.156), clean price scatter (R=0.935)
2. **MSFT / AAPL** — second highest return correlation (0.700), similar betas (1.176 vs 1.216), tightest price scatter in universe (R=0.966)
3. **GOOGL / AAPL** — strong correlation (0.622), similar betas, clean scatter (R=0.927)

**Caution flags:**
- NVDA decoupled from all peers from mid-2023 onward due to AI-driven price surge — test over 2018-2022 sub-period only
- META has structural regime break in 2022 (-75% drawdown) — single-day kurtosis events inflate spread risk
- 2022 is the critical stress year for all backtests

---

## 4. Outstanding Work — Phases 3, 4, and 5

### Phase 3a — Cointegration Screening (`phase3_cointegration.py`)

- [ ] Load price levels from `data/raw/` for all 6 tickers
- [ ] Run ADF test on each individual price series to confirm non-stationarity (I(1))
- [ ] Run Engle-Granger cointegration test on all 15 pairs
- [ ] Run Johansen test on priority pairs for robustness
- [ ] Test NVDA pairs over restricted 2018-2022 window
- [ ] Rank passing pairs by ADF test statistic on residuals
- [ ] Save cointegration results table to `outputs/reports/cointegration_results.csv`
- [ ] Plot spread (residual) series for all passing pairs

### Phase 3b — Pairs Trading Strategy (`phase3_strategy.py`)

- [ ] Estimate hedge ratio (OLS) for each cointegrated pair
- [ ] Construct spread series: `spread = price_A - hedge_ratio * price_B`
- [ ] Compute rolling z-score of spread (30-day window)
- [ ] Implement signal logic: enter long/short at z > ±2, exit at z = 0, stop at z > ±3
- [ ] Backtest: simulate daily P&L with transaction costs
- [ ] Compute performance metrics: Sharpe ratio, max drawdown, win rate, total return
- [ ] Walk-forward validation with annual parameter re-estimation
- [ ] Stress test over 2022 bear market period
- [ ] Save performance summary to `outputs/reports/strategy_results.csv`
- [ ] Save equity curve and z-score charts to `outputs/charts/`

### Phase 4 — Sentiment Analysis (`phase4_sentiment.py`)

- [ ] Define news sources and scraping targets (financial news headlines)
- [ ] Fetch headlines using `requests` + `BeautifulSoup`
- [ ] Score each headline with NLTK VADER sentiment analyser
- [ ] Aggregate daily sentiment scores per ticker
- [ ] Overlay sentiment on spread z-score — identify signal conflicts
- [ ] Evaluate whether sentiment improves entry timing or reduces false signals
- [ ] Save sentiment scores to `outputs/reports/sentiment_scores.csv`
- [ ] Plot sentiment time series alongside spread chart

### Phase 5 — Streamlit Dashboard (`app/app.py`)

Currently a stub. To be built once Phases 3 and 4 are complete.

- [ ] Sidebar: ticker pair selector, date range slider, z-score threshold inputs
- [ ] Tab 1 — EDA: display Phase 2 charts from `outputs/charts/`
- [ ] Tab 2 — Cointegration: show test results table and spread plots
- [ ] Tab 3 — Strategy: equity curve, z-score signal chart, performance metrics
- [ ] Tab 4 — Sentiment: daily sentiment vs spread overlay
- [ ] Deploy locally with `streamlit run app/app.py`

---

## 5. Important Notes and Decisions

| Decision | Detail |
|----------|--------|
| **Date range corrected** | Initially set to 2015-2024. Corrected to 2018-01-01 / 2025-12-31 for an 8-year window as per project spec. Phase 1 was re-run after the fix. |
| **Ticker scope trimmed** | `utils.py` originally scaffolded with 64 tickers across 8 sectors. Trimmed to 6 tech stocks (MSFT, GOOGL, NVDA, AAPL, AMZN, META) before any data was downloaded. |
| **`auto_adjust=True`** | yfinance downloads use `auto_adjust=True` throughout. This means the `Close` column already reflects dividend and split adjustments — no manual adjustment step is needed. |
| **MultiIndex flattening** | yfinance >= 0.2.x returns a MultiIndex column header when downloading a single ticker. All scripts call `df.columns.get_level_values(0)` to flatten before use. |
| **Return formula** | Daily returns computed as `(Xt - Xt-1) / Xt-1 * 100` (percentage, not decimal). This is consistent across all scripts. Benchmarks use the same formula. |
| **First-row NaN drop** | The shift(1) return calculation produces a NaN on row 0. Phase 1 detects and drops only that row using `first_valid_index()` rather than a blanket `dropna()`. |
| **Windows encoding** | Unicode characters (arrows, ellipsis) in print statements caused `cp1252` codec errors on Windows. All print statements use plain ASCII only. |
| **Matplotlib backend** | `matplotlib.use("Agg")` is set at the top of `phase2_eda.py` to prevent display errors when running headlessly on Windows without a display server. |
| **NVDA sub-period note** | NVDA's price increased ~37x over the full period. Any cointegration test run on the full 2018-2025 window for NVDA pairs is expected to fail. Restrict to 2018-2022 for NVDA. |
| **2022 stress year** | All backtests must be evaluated through the 2022 bear market (META -75%, NVDA -55%, AMZN -50%). This is the critical period for strategy robustness. |
