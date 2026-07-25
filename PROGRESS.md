# Project Progress — Pairs Trading Analytics
**Last updated:** 2026-07-25
**Status:** Phase 3 complete (cointegration screening + strategy backtest). Phase 4 next.

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
| 3a | `phase3_cointegration.py` | Cointegration screening — all 15 pairs, EG + Johansen | COMPLETE |
| 3b | `phase3_strategy.py` | Pairs trading strategy — AMZN/META, z-score backtest | COMPLETE |
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
- NVDA decoupled from all peers from mid-2023 onward due to AI-driven price surge — sub-period check (2018-2022) was run in Phase 3 and all NVDA pairs still failed both tests
- META has structural regime break in 2022 (-75% drawdown) — single-day kurtosis events inflate spread risk
- 2022 is the critical stress year for all backtests

---

## 4. Phase 3 — Cointegration Screening (COMPLETE)

**Scripts:** `src/phase3_cointegration.py`, `src/phase3_strategy.py`
**Full report:** `outputs/reports/phase3_report.md`

### Phase 3a — Cointegration Screening

**Method:** OLS regression on log prices in both directions (A~B and B~A); direction with lower ADF p-value on residuals is selected. Engle-Granger test (`statsmodels.tsa.stattools.coint`) and Johansen test applied to all 15 pairs. Pairs ranked by EG p-value.

**Result: Only 1 pair passed both tests at 5% significance — AMZN/META.**

| Rank | Pair | OLS Direction | Hedge Ratio | EG p-value | EG | Johansen |
|------|------|---------------|-------------|------------|-----|----------|
| 1 | AMZN/META | AMZN~META | 0.5976 | 0.0144 | PASS | PASS |
| 2 | NVDA/AMZN | AMZN~NVDA | 0.2392 | 0.1220 | FAIL | FAIL |
| 3 | GOOGL/AMZN | AMZN~GOOGL | 0.6671 | 0.1844 | FAIL | FAIL |
| 4 | MSFT/AAPL | MSFT~AAPL | 0.8711 | 0.2032 | FAIL | FAIL |
| 15 | MSFT/GOOGL | MSFT~GOOGL | 1.1214 | 0.7091 | FAIL | FAIL |

**Key findings:**
- High return correlation (MSFT/GOOGL = 0.709) does NOT imply cointegration — their price levels diverged over 8 years
- All 5 NVDA pairs fail over the full 2018-2025 window — NVDA's 37x price gain from mid-2023 (AI/GPU surge) breaks any long-run equilibrium
- NVDA sub-period check (2018-2022 only) also run to remove the structural break: all 5 pairs still fail (EG p-values worsen — NVDA/AMZN rises from 0.122 to 0.755, NVDA/META from 0.469 to 0.945). NVDA had no cointegrating relationship with peers even before the AI surge
- AMZN/META Johansen trace stat = 17.52 vs 5% critical value = 15.49

### Phase 3b — Pairs Trading Strategy (AMZN/META)

**Setup:** Spread = log(AMZN) − 1.0405 × log(META) + 0.8742 | 30-day rolling z-score | Entry ±2σ, exit at 0, stop loss ±3σ | 10 bps per leg | Train: 2018-2021, Test: 2022-2025

**Execution convention:** signal observed at day i close, trade executes at day i+1 (next-day execution — no same-day look-ahead).

**Out-of-sample test performance (2022-2025):**

| Metric | Value |
|--------|-------|
| Total P&L | −0.154 |
| Annualised Sharpe | −0.12 |
| Max drawdown | −0.653 |
| Win rate | 52.3% |
| Time in market | ~59% |

**Walk-forward results (4 of 7 years profitable):**

| Year | Hedge Ratio | R² | Sharpe | P&L |
|------|-------------|-----|--------|-----|
| 2019 | 0.051 | 0.003 | +0.04 | +0.006 |
| 2020 | 0.296 | 0.098 | +0.62 | +0.138 |
| 2021 | 1.125 | 0.698 | +0.68 | +0.137 |
| 2022 | 1.041 | 0.834 | −0.17 | −0.084 |
| 2023 | 0.790 | 0.621 | −0.36 | −0.077 |
| 2024 | 0.727 | 0.612 | −0.17 | −0.030 |
| 2025 | 0.629 | 0.706 | +0.58 | +0.109 |

**Average annual Sharpe: +0.18. Total walk-forward P&L: +0.199.**

**Key findings:**
- Strategy is marginally positive on a walk-forward basis (4 of 7 years profitable, total PnL +0.199)
- Performance strongest in 2020-2021 when hedge ratio stabilised (R² ≥ 0.70) and weakest in 2022-2024 when regime shifted
- The cointegrating relationship only became reliable after 2020 — in 2018-2019 R² was near zero
- Win rate consistently above 50% — direction signal works; Sharpe is modest due to small per-day P&L magnitude
- 2022 losses contained (Sharpe −0.17) despite META −65% and AMZN −50%
- Sentiment overlay (Phase 4) may help filter entries during macro stress periods

### Phase 3 Output Files

| File | Description |
|------|-------------|
| `outputs/reports/cointegration_results.csv` | Full EG + Johansen results, all 15 pairs |
| `outputs/reports/strategy_results.csv` | Backtest + walk-forward performance metrics |
| `outputs/reports/phase3_report.md` | Full written analytical report |
| `outputs/charts/spread_series_all_pairs.png` | OLS residual plots, all 15 pairs |
| `outputs/charts/cointegration_ranking.png` | EG p-value ranking bar chart |
| `outputs/charts/strategy_zscore_equity.png` | Z-score signals + equity curve (test period) |
| `outputs/charts/strategy_walkforward.png` | Annual Sharpe + P&L bar charts |
| `outputs/charts/strategy_zscore_full.png` | Full period z-score (2018-2025) |
| `outputs/charts/strategy_annual_equity.png` | Annual equity curves (walk-forward) |

---

## 5. Outstanding Work — Phases 4 and 5

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

## 6. Important Notes and Decisions

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
| **NVDA sub-period note** | NVDA's price increased ~37x over the full period. Phase 3 tested all 5 NVDA pairs over both the full 2018-2025 window and the restricted 2018-2022 window. All pairs failed both EG and Johansen in both windows. Results in `outputs/reports/nvda_subperiod_results.csv`. |
| **2022 stress year** | All backtests must be evaluated through the 2022 bear market (META -75%, NVDA -55%, AMZN -50%). This is the critical period for strategy robustness. |
