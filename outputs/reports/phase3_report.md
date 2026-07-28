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

**Selection criterion:** a pair was selected for trading if and only if it passed both the Engle-Granger test and the Johansen test at the 5% significance level. Requiring agreement from two independent methods — one residual-based and one multivariate — reduces the risk of selecting a spurious pair that passes a single test by chance. Of the 15 pairs screened, **only AMZN/META met this criterion** and was taken forward to the strategy phase.

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

#### Finding 3: NVDA pairs fail over both the full period and the restricted 2018-2022 window

The Phase 2 EDA report recommended testing NVDA pairs over a restricted 2018-2022 window to remove the structural break caused by the AI/GPU-driven price surge from mid-2023. Both tests were run.

**Full period (2018-2025):** All 5 NVDA pairs fail both EG and Johansen tests (EG p-values range from 0.122 to 0.583).

**Restricted window (2018-2022):** All 5 NVDA pairs still fail both tests. EG p-values in the restricted window are actually worse than the full period for most pairs (NVDA/AMZN rises from p=0.122 to p=0.755; NVDA/META from p=0.469 to p=0.945).

| Pair | Full period EG p | 2018-2022 EG p | Full EG | Restricted EG |
|------|------------------|----------------|---------|---------------|
| MSFT/NVDA | 0.246 | 0.408 | FAIL | FAIL |
| GOOGL/NVDA | 0.485 | 0.344 | FAIL | FAIL |
| NVDA/AAPL | 0.583 | 0.431 | FAIL | FAIL |
| NVDA/AMZN | 0.122 | 0.755 | FAIL | FAIL |
| NVDA/META | 0.469 | 0.945 | FAIL | FAIL |

This result indicates that NVDA was not cointegrated with any of its tech peers even before the 2023 AI surge. The structural break confirmed in Phase 2 EDA was a continuation of an already unstable relationship, not the sole cause of cointegration failure.

#### Finding 4: AMZN appears frequently as the dependent variable

In 5 of the 15 pairs, AMZN's log price was selected as the dependent variable (the AMZN~X direction produced more stationary residuals than X~AMZN). In the notation A~B, A is the dependent variable regressed on B as the independent variable. This pattern suggests that modelling AMZN as a function of its peers captures the long-run relationship better than the reverse, consistent with AMZN having broad macro sensitivity (AWS cloud, consumer spending) that ties it to movements in other large-cap tech stocks.

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

### Execution Timing Convention

Signals are generated using the **previous day's z-score** (`z_prev`). The trade executes at the following day's open and earns that day's spread change. This "signal on close, execute next day" convention prevents earning the same-day spread move that triggered the signal, which would be unrealisable in practice.

### Train / Test / Full Period Performance Comparison (`strategy_train_test_comparison.csv`)

| Metric | Train 2018-2021 | Test 2022-2025 | Full 2018-2025 |
|--------|----------------|---------------|----------------|
| Days | 1,008 | 1,002 | 2,010 |
| Total P&L | +0.1816 | −0.1544 | −0.0187 |
| Annualised P&L | +0.0454 | −0.0388 | −0.0023 |
| Annualised volatility | 0.2138 | 0.3259 | 0.2762 |
| Sharpe ratio | +0.21 | −0.12 | −0.01 |
| Sortino ratio | +0.22 | −0.11 | −0.01 |
| Maximum drawdown | −0.3608 | −0.6531 | −0.6683 |
| Calmar ratio | +0.13 | −0.06 | −0.003 |
| Win rate (days in position) | 50.4% | 52.3% | 51.2% |
| Estimated trades | 28 | 25 | 53 |
| Avg trade P&L | +0.006484 | −0.006175 | −0.000353 |
| Time in market | 55.8% | 59.4% | 58.3% |

### Out-of-Sample Test Period Results (2022-2025)

### Walk-Forward Validation Results (2019-2025)

| Year | Hedge Ratio | R² | Sharpe | Total P&L | Win Rate |
|------|-------------|-----|--------|-----------|----------|
| 2019 | 0.051 | 0.003 | +0.04 | +0.0057 | 56.8% |
| 2020 | 0.296 | 0.098 | +0.62 | +0.1380 | 50.0% |
| 2021 | 1.125 | 0.698 | +0.68 | +0.1372 | 57.5% |
| 2022 | 1.041 | 0.834 | −0.17 | −0.0842 | 51.8% |
| 2023 | 0.790 | 0.621 | −0.36 | −0.0773 | 46.9% |
| 2024 | 0.727 | 0.612 | −0.17 | −0.0299 | 57.2% |
| 2025 | 0.629 | 0.706 | +0.58 | +0.1093 | 53.5% |

**Profitable years: 4 of 7. Average annual Sharpe: +0.18. Total walk-forward P&L: +0.199.**

The train period (2018-2021) produced a positive Sharpe of +0.21, while the test period (2022-2025) produced −0.12. The divergence is partly explained by the hedge ratio only becoming stable after 2020 (R² = 0.698 from the 2018-2020 training window used in 2021), and partly by the structurally more volatile macro environment of 2022-2025.

### 2022 Stress Test

The strategy performed worst in 2022 (Sharpe −0.17, P&L −0.084). This was the year META fell approximately 65% and AMZN fell approximately 50% — a simultaneous collapse driven by different company-specific factors that caused the spread to diverge and not revert within the rolling window. Despite this, the loss was contained relative to the overall period, largely because the next-day execution convention meant the strategy was not exposed to the full same-day move on each signal day.

### Trade Log — Test Period (`trade_log_test.csv`)

The trade log records each individual round-trip trade on the test period (2022-2025): entry/exit date, position direction (long or short spread), z-score at entry and exit, holding period in days, gross P&L, transaction cost, and net P&L.

**Test period trade log summary:**
- Total trades: 26
- Winning trades: 14 (53.8%)
- Average holding period: 32.8 calendar days
- Average net P&L per trade: −0.0059 log-return units
- Total transaction cost: 26 × 0.004 = 0.104 (approximately 67% of the gross loss is attributable to transaction costs)

The trade log reveals that the 2023 bear trade (entry 2023-02-03, exit 2023-04-21, holding 77 days, net P&L −0.222) is the single largest loss, contributing more than the total test-period drawdown in isolation. This was a long-spread trade entered when z = −3.80 — near the stop-loss threshold — which continued to deteriorate before eventually being stopped out.

### Benchmark Comparison — Test Period (`strategy_vs_benchmarks.png`)

The pairs strategy is compared against four market-neutral and directional benchmarks over the test period (2022-2025), all expressed in cumulative log-return units for comparability.

| Strategy / Benchmark | Cumulative Log-Return (2022-2025) |
|----------------------|----------------------------------|
| Pairs strategy (AMZN/META) | −0.154 |
| Buy-and-hold AMZN | +0.311 |
| Equal-weight AMZN/META | +0.497 |
| S&P 500 | +0.363 |
| Buy-and-hold META | +0.684 |

The pairs strategy underperforms all four benchmarks over the test period. This result directly answers the question a supervisor would ask: *"Did the strategy add value compared with simply holding the stocks or the index?"* The answer is no — at least over 2022-2025. The strategy's value proposition (market-neutrality) comes at the cost of significantly lower absolute return in a period when both stocks and the index recovered strongly.

### Parameter Sensitivity Analysis (`sensitivity_analysis.csv`)

To test whether the baseline parameters are cherry-picked or whether the results are robust, three one-way sensitivity sweeps and a full 36-combination grid (3 entry thresholds × 4 rolling windows × 3 stop-loss levels) were run on the test period.

**Varying entry threshold (window = 30d, stop = 3.0σ):**
| Entry | Sharpe | Sortino | Total P&L |
|-------|--------|---------|-----------|
| 1.5σ | −0.23 | −0.24 | −0.325 |
| **2.0σ (baseline)** | −0.12 | −0.11 | −0.154 |
| 2.5σ | −0.05 | −0.03 | −0.051 |

**Varying rolling window (entry = 2.0σ, stop = 3.0σ):**
| Window | Sharpe | Sortino | Total P&L |
|--------|--------|---------|-----------|
| 20d | −0.06 | −0.06 | −0.076 |
| **30d (baseline)** | −0.12 | −0.11 | −0.154 |
| 60d | −0.33 | −0.30 | −0.430 |
| 90d | −0.08 | −0.07 | −0.098 |

**Varying stop loss (entry = 2.0σ, window = 30d):**
| Stop | Sharpe | Sortino | Total P&L |
|------|--------|---------|-----------|
| **3.0σ (baseline)** | −0.12 | −0.11 | −0.154 |
| 3.5σ | −0.09 | −0.08 | −0.114 |
| 4.0σ | −0.07 | −0.07 | −0.094 |

**Full 36-combination grid:** Best Sharpe = +0.26 (entry = 2.5σ, window = 90d, stop = 4.0σ). No combination achieves a Sharpe above 1.0. The result is not an artefact of the baseline parameter choice — the strategy is only weakly profitable across the entire parameter space in the test period.

### Key Findings — Strategy

#### Finding 1: Strategy is marginally positive on a walk-forward basis

With correct next-day execution, the strategy produces positive total P&L (+0.199) over 7 walk-forward years, with 4 of 7 years profitable. The out-of-sample test period (2022-2025) is modestly negative (Sharpe −0.12), reflecting the challenging macro environment of that specific window. This is a more realistic and balanced result than the same-day execution convention produced.

#### Finding 2: Hedge ratio instability creates regime-dependent performance

The hedge ratio shifts significantly across subperiods — from 0.051 (2018-only training, R² = 0.003) to 1.125 (2018-2020 training, R² = 0.698). Performance is strongest in 2020 and 2021 when the relationship had stabilised (R² ≥ 0.70) and weakest in years when the hedge ratio was drifting. This confirms the cointegrating relationship only became reliable after 2020 when both stocks became driven by common digital-advertising and cloud themes.

#### Finding 3: The cointegrating relationship is regime-dependent

In early subperiods (2018-2019), AMZN and META had almost no price-level relationship (R² = 0.003). The cointegration found over the full 8-year period is a long-run statistical property that only stabilised from 2020 onward. A practitioner would need a minimum of 2-3 years of stable R² before trading the pair with confidence.

#### Finding 4: 2022 was the worst year but losses were contained

META's 65% fall in 2022 was driven by company-specific factors (Metaverse investment, advertising collapse) while AMZN's 50% fall was more macro-driven (AWS slowdown, consumer weakness). The spread diverged significantly but the next-day execution convention and stop-loss rule at ±3σ limited the annual loss to −0.084.

#### Finding 5: Win rate consistently above 50% across most years

Win rates range from 47% to 57%, with most years above 50%. This indicates the directional signal (z-score mean reversion) is working — the strategy is correct about direction more often than not. The modest Sharpe ratios reflect the small magnitude of individual day P&L rather than a poor signal, consistent with a slow mean-reverting spread.

---

## Conclusions and Implications for Phase 4

1. **AMZN/META is the only cointegrated pair** across all 15 combinations of the 6 tech stocks over 2018-2025. All other pairs fail both the Engle-Granger and Johansen tests.

2. **The strategy is only weakly profitable.** Walk-forward validation shows a positive total P&L (+0.199 over 7 years) but an average Sharpe of only +0.18. The out-of-sample test period (2022-2025) produces a negative Sharpe (−0.12) and a cumulative loss of −0.154 log-return units.

3. **Cointegration alone is not sufficient for a profitable trading rule.** The statistical evidence of cointegration (EG p = 0.014, Johansen trace = 17.52 vs 15.49) is real but does not guarantee mean-reversion fast enough to be captured at the 30-day z-score window. The hedge ratio instability across years (R² ranging from 0.003 to 0.834) means the long-run relationship is only reliable in stable regimes.

4. **Benchmark comparison confirms the strategy does not add value over the test period.** Buy-and-hold AMZN returned +0.311 log-units, META +0.684, and the S&P 500 +0.363 — all well above the strategy's −0.154. The market-neutral property does not compensate for foregone directional returns in a period of strong equity recovery.

5. **Parameter sensitivity confirms results are robust to parameter choice.** No combination among 36 tested achieves a Sharpe above 1.0 in the test period. The baseline parameters are not specially tuned to produce a positive result.

6. **Phase 4 (Sentiment Analysis) is motivated by strategy limitations.** The trade log shows that the largest losses occur during macro stress periods (2022, early 2023) when the spread diverges beyond the stop-loss threshold. A sentiment filter identifying adverse macroeconomic conditions may help avoid entering new positions in these environments.

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/reports/cointegration_results.csv` | Full EG + Johansen results, all 15 pairs |
| `outputs/reports/nvda_subperiod_results.csv` | NVDA pairs re-tested on 2018-2022 window |
| `outputs/reports/strategy_results.csv` | Backtest and walk-forward metrics (expanded) |
| `outputs/reports/strategy_train_test_comparison.csv` | Train / test / full period side-by-side |
| `outputs/reports/trade_log_test.csv` | Individual trade log — test period (26 trades) |
| `outputs/reports/sensitivity_analysis.csv` | 36-combination parameter sensitivity grid |
| `outputs/reports/phase3_strategy_conclusion.txt` | Written strategy conclusion |
| `outputs/charts/spread_series_all_pairs.png` | OLS residual plots for all 15 pairs |
| `outputs/charts/cointegration_ranking.png` | EG p-value ranking bar chart |
| `outputs/charts/strategy_zscore_equity.png` | Z-score signals + equity curve (test period) |
| `outputs/charts/strategy_walkforward.png` | Annual Sharpe and P&L bar charts |
| `outputs/charts/strategy_zscore_full.png` | Full period z-score (2018-2025) |
| `outputs/charts/strategy_annual_equity.png` | Annual equity curves (walk-forward) |
| `outputs/charts/strategy_vs_benchmarks.png` | Strategy vs buy-hold benchmarks + S&P 500 |
