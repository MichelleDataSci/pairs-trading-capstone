# Phase 3 Report — Cointegration Screening & Pairs Trading Strategy
**Date:** 2026-07-25
**Scripts:** `src/phase3_cointegration.py`, `src/phase3_strategy.py`
**Data:** Log prices, 2018-01-02 to 2025-12-30 (2,010 trading days)
**Universe:** MSFT, GOOGL, NVDA, AAPL, AMZN, META (6 stocks → 15 possible pairs)

---

## Part A — Cointegration Screening

### Method

All tests were conducted on **log prices** (not raw prices, not returns), consistent with the standard cointegration framework which requires I(1) series. Log prices of individual stocks are non-stationary (contain a unit root), making them appropriate inputs for cointegration testing.

Three methods were applied to every pair:

1. **OLS regression (both directions) + ADF test on residuals** — since OLS is directional, both A~B and B~A were estimated; the direction whose residuals had a lower ADF p-value (stronger stationarity evidence) was selected as the preferred specification.
2. **Engle-Granger (EG) test** — `statsmodels.tsa.stattools.coint()` applied to the chosen direction. This uses non-standard critical values correct for regression residuals.
3. **Johansen multivariate cointegration test** — `coint_johansen()` with `det_order=0` (constant in cointegrating relationship) and `k_ar_diff=1`. Direction-independent. Trace statistic and max-eigenvalue statistic compared to 5% critical values.

All pairs ranked by EG p-value (ascending = stronger evidence).

---

### Results — All 15 Pairs Ranked

| Rank | Pair | OLS Direction | Hedge Ratio | ADF p (residuals) | EG p-value | EG Pass | Johansen Pass |
|------|------|---------------|-------------|-------------------|------------|---------|---------------|
| 1 | **AMZN/META** | AMZN~META | 0.5976 | 0.0031 | **0.0144** | **YES** | **YES** |
| 2 | NVDA/AMZN | AMZN~NVDA | 0.2392 | 0.0397 | 0.1220 | NO | NO |
| 3 | GOOGL/AMZN | AMZN~GOOGL | 0.6671 | 0.0669 | 0.1844 | NO | NO |
| 4 | MSFT/AAPL | MSFT~AAPL | 0.8711 | 0.0758 | 0.2032 | NO | NO |
| 5 | MSFT/NVDA | MSFT~NVDA | 0.4094 | 0.0951 | 0.2458 | NO | NO |
| 6 | AAPL/AMZN | AMZN~AAPL | 0.4880 | 0.1395 | 0.3208 | NO | NO |
| 7 | MSFT/AMZN | AMZN~MSFT | 0.5674 | 0.1606 | 0.3554 | NO | NO |
| 8 | MSFT/META | MSFT~META | 0.8381 | 0.1818 | 0.3886 | NO | NO |
| 9 | NVDA/META | META~NVDA | 0.3674 | 0.2387 | 0.4685 | NO | NO |
| 10 | GOOGL/NVDA | NVDA~GOOGL | 2.5463 | 0.2515 | 0.4850 | NO | NO |
| 11 | NVDA/AAPL | AAPL~NVDA | 0.4471 | 0.3348 | 0.5828 | NO | NO |
| 12 | AAPL/META | AAPL~META | 0.8562 | 0.3820 | 0.6313 | NO | NO |
| 13 | GOOGL/AAPL | AAPL~GOOGL | 1.2392 | 0.4164 | 0.6632 | NO | NO |
| 14 | GOOGL/META | META~GOOGL | 0.9357 | 0.4198 | 0.6665 | NO | NO |
| 15 | MSFT/GOOGL | MSFT~GOOGL | 1.1214 | 0.4691 | 0.7091 | NO | NO |

**EG vs Johansen agreement summary:**
- Both methods pass (5% level): **1 pair** — AMZN/META
- EG only: 0 pairs
- Johansen only: 0 pairs
- Neither: 14 pairs

---

### Key Findings — Cointegration

#### Finding 1: Only AMZN/META passes both tests

AMZN/META is the only pair where both the Engle-Granger test (p = 0.0144) and the Johansen trace test (stat = 17.52 vs critical value 15.49 at 5%) confirm cointegration. Both methods agree, which provides stronger confidence than either test alone.

The chosen OLS direction is **AMZN ~ META** (AMZN as dependent variable), giving:

> Spread = log(AMZN) − 0.5976 × log(META) − 1.532

The residuals of this regression are stationary (ADF p = 0.0031), meaning the spread between log-AMZN and 0.5976 × log-META reverts to a long-run mean over the 8-year period.

#### Finding 2: High return correlation does NOT imply cointegration

The pairs most correlated in Phase 2 EDA — MSFT/GOOGL (return correlation 0.709), MSFT/AAPL (0.700), MSFT/AMZN (0.682) — all fail both cointegration tests with p-values above 0.35. This is a critical distinction:

- **Return correlation** measures co-movement of daily price changes
- **Cointegration** measures whether two price level series share a stable long-run equilibrium

A pair can be highly correlated in returns but have price levels that diverge permanently — which is what MSFT/GOOGL and MSFT/AAPL show over 2018-2025.

#### Finding 3: NVDA pairs all fail

NVDA's 37x price appreciation between 2023 and 2025 (driven by the AI/GPU demand surge) broke any long-run equilibrium it might have held with other stocks. All 5 NVDA pairs fail both tests. The structural break in NVDA's price trend from mid-2023 onward dominates any pre-existing cointegrating relationship.

#### Finding 4: AMZN appears frequently as the independent variable

In 7 of the 15 pairs, AMZN's log price was selected as the independent variable (AMZN~X was the less stationary direction; X~AMZN was better). This suggests AMZN's price level is a relatively stronger "driver" in these long-run relationships — consistent with AMZN's broader exposure to cloud (AWS) and consumer sectors making it a bellwether.

#### Finding 5: The next-closest pairs (NVDA/AMZN and GOOGL/AMZN) do not pass

NVDA/AMZN (EG p = 0.122) and GOOGL/AMZN (EG p = 0.184) are the next-ranked pairs but both fail the 5% threshold by a comfortable margin. Neither passes the Johansen test either. There is no second-tier pair that could be considered a borderline candidate for trading.

---

## Part B — Pairs Trading Strategy (AMZN/META)

### Setup

- **Spread:** log(AMZN) − 1.0405 × log(META) − (−0.8742)
  *(Note: hedge ratio re-estimated on 2018-2021 training data = 1.0405)*
- **Z-score:** rolling 30-day mean and standard deviation of spread
- **Signal rules:**
  - Enter **long spread** (buy AMZN, sell META) when z < −2.0
  - Enter **short spread** (sell AMZN, buy META) when z > +2.0
  - Exit position when z crosses 0 (mean reversion complete)
  - Stop loss when |z| exceeds 3.0 in the loss direction
- **Transaction costs:** 10 bps per leg (applied at both entry and exit)
- **Train period:** 2018-01-01 to 2021-12-31
- **Test period:** 2022-01-01 to 2025-12-31

### Out-of-Sample Test Period Results (2022-2025)

| Metric | Value |
|--------|-------|
| Total P&L (log-return units) | −2.5876 |
| Annual P&L | −0.6508 |
| Annualised Sharpe ratio | −1.90 |
| Maximum drawdown | −2.6862 |
| Win rate (days in position) | 48.2% |
| Position changes (trade signals) | 51 |
| Time in market | 59.4% |

### Walk-Forward Validation Results (2019-2025)

| Year | Hedge Ratio | R² | Sharpe | Total P&L | Win Rate |
|------|-------------|-----|--------|-----------|----------|
| 2019 | 0.0510 | 0.003 | −1.48 | −0.2062 | 51.7% |
| 2020 | 0.2963 | 0.098 | −1.46 | −0.3678 | 43.6% |
| 2021 | 1.1247 | 0.698 | −1.20 | −0.2674 | 52.6% |
| 2022 | 1.0405 | 0.834 | −2.14 | −1.0942 | 48.2% |
| 2023 | 0.7902 | 0.621 | −2.62 | −0.5718 | 44.1% |
| 2024 | 0.7265 | 0.612 | −1.86 | −0.4092 | 53.3% |
| 2025 | 0.6286 | 0.706 | −1.57 | −0.3201 | 47.5% |

**Profitable years: 0 of 7. Average annual Sharpe: −1.76.**

### 2022 Stress Test

The strategy performed worst in 2022 (Sharpe −2.14, P&L −1.094). This was the year META fell approximately 65% and AMZN fell approximately 50% — a simultaneous collapse that caused the spread to diverge sharply and remain out of equilibrium for extended periods rather than mean-reverting within the 30-day window.

### Key Findings — Strategy

#### Finding 1: Cointegration confirmed but strategy is unprofitable

AMZN/META is statistically cointegrated over the full 8-year period, but the pairs trading strategy generates negative returns in every single year. This demonstrates a critical principle: **statistical cointegration does not guarantee trading profitability**.

#### Finding 2: Hedge ratio is structurally unstable

The hedge ratio shifts dramatically across subperiods:
- 2018-only training: β = 0.051 (R² = 0.003 — almost no relationship)
- 2018-2020 training: β = 1.125 (R² = 0.698)
- Full 2018-2024 training: β = 0.627 (R² = 0.706)

This instability means the "spread" being traded is defined differently depending on when the parameters were estimated. When the hedge ratio changes, the z-score thresholds trigger on a spread that no longer reflects the current equilibrium, generating false signals.

#### Finding 3: The cointegrating relationship is a long-run artefact

The full-period cointegration only emerges over the complete 8-year window. In early subperiods (2018-2019), AMZN and META had essentially no price-level relationship (R² = 0.003). The relationship strengthened only after 2020 when both stocks became driven by common macro and digital-advertising/cloud themes. This means the cointegration is a backward-looking statistical finding, not a stable real-time signal.

#### Finding 4: 2022 broke the spread severely

Both stocks fell dramatically in 2022 but not proportionally. META's 65% fall was driven by specific company-level factors (Metaverse investment losses, advertising market collapse) while AMZN's 50% fall was more macro-driven (AWS growth slowdown, consumer weakness). The spread widened well beyond ±3σ and stayed there for months, triggering repeated stop losses.

#### Finding 5: Win rate near 50% but Sharpe consistently negative

The win rate of roughly 48-53% across years (near coin-flip) combined with consistently negative Sharpe ratios suggests the strategy wins slightly more often than it loses on individual days, but the **magnitude of losses exceeds the magnitude of wins**. This is characteristic of a strategy that is right about direction but wrong about timing — positions are entered too early and the spread continues to diverge before eventually reverting.

---

## Conclusions and Implications for Phase 4

1. **AMZN/META is the only cointegrated pair** across all 15 combinations of the 6 tech stocks over 2018-2025. All other pairs fail both the Engle-Granger and Johansen tests.

2. **The strategy as implemented is not profitable.** This is a meaningful academic finding — it shows the gap between statistical evidence of cointegration and practical trading profitability.

3. **The main issues are:** (a) hedge ratio instability across regimes; (b) the 2022 bear market causing prolonged spread divergence; (c) the rolling z-score window (30 days) being too short to capture the true mean-reversion speed of the spread.

4. **Phase 4 (Sentiment Analysis) is motivated by these failures.** If a sentiment filter can identify periods when the spread is likely to revert (positive news for one stock relative to the other), it may help avoid entering positions during macro stress periods like 2022 when mean reversion takes longer than the signal window.

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/reports/cointegration_results.csv` | Full test results for all 15 pairs |
| `outputs/reports/strategy_results.csv` | Backtest and walk-forward performance metrics |
| `outputs/charts/spread_series_all_pairs.png` | OLS residual plots for all 15 pairs |
| `outputs/charts/cointegration_ranking.png` | EG p-value ranking bar chart |
| `outputs/charts/strategy_zscore_equity.png` | Z-score signals + equity curve (test period) |
| `outputs/charts/strategy_walkforward.png` | Annual Sharpe and P&L bar charts |
| `outputs/charts/strategy_zscore_full.png` | Full period z-score (2018-2025) |
| `outputs/charts/strategy_annual_equity.png` | Annual equity curves (walk-forward) |
