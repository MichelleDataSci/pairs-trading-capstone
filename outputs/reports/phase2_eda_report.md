# Phase 2 EDA — Analytical Report

**Universe:** MSFT, GOOGL, NVDA, AAPL, AMZN, META
**Period:** 2018-01-03 to 2025-12-30
**N:** 2,009 trading days
**Analyses:** 14 (11 charts, 3 reports, 1 conclusion file)

---

## 1. Summary Statistics (`summary_stats.csv`)

**What it shows.** Mean daily return, standard deviation, min/max daily moves, skewness, and excess kurtosis for each of the six stocks.

**Key observations.**

| Ticker | Mean % | Std % | Skew  | Ex. Kurtosis |
|--------|--------|-------|-------|--------------|
| MSFT   | 0.107  | 1.79  | +0.08 | 7.07         |
| GOOGL  | 0.107  | 1.95  | -0.01 | 3.85         |
| NVDA   | 0.233  | 3.23  | +0.13 | 4.83         |
| AAPL   | 0.114  | 1.94  | +0.15 | 6.43         |
| AMZN   | 0.091  | 2.17  | +0.10 | 4.25         |
| META   | 0.100  | 2.61  | -0.33 | 18.07        |

NVDA has roughly twice the daily standard deviation of MSFT, making it the noisiest stock in the universe. All six stocks have positive excess kurtosis — every distribution has fatter tails than a normal distribution, meaning extreme daily moves occur far more often than a Gaussian model would predict. META's kurtosis of 18.07 is extreme and is almost entirely driven by its single-day -26.4% crash in February 2022 and its subsequent +23.3% recovery day. Skewness values are all near zero, meaning gains and losses are roughly symmetric in magnitude.

**Implications for Phase 3.** Fat tails mean the spread between any pair can gap violently on a single day. Mean-reversion signals that assume Gaussian noise will underestimate how often the spread blows past the exit threshold. Consider robust z-score thresholds (e.g. ±2 standard deviations) and hard stop-losses.

---

## 2. Histogram Grid with KDE Overlay (`hist_returns.png`)

**What it shows.** One subplot per stock: a density histogram of all daily returns over the full period, with a kernel density estimate overlaid to show the smooth return distribution shape.

**Key observations.** All six distributions are sharply peaked around zero (leptokurtic) with visible heavy tails. The KDE curve has a much taller, narrower centre than a normal bell curve, with noticeably thicker tails extending beyond ±3%. NVDA's histogram is the widest by a clear margin. META's histogram features the most extreme outlier bars at the far tails, consistent with its kurtosis of 18. MSFT and AAPL have the most compact, bell-like shapes, though they still deviate from normality.

**Implications for Phase 3.** No stock's daily returns are normally distributed — spread z-score models should be calibrated on empirical quantiles rather than assuming normality. Positions sized using Gaussian VaR will underestimate actual tail risk.

---

## 3. Correlation Heatmap (`correlation_heatmap.png`)

**What it shows.** The lower-triangle Pearson correlation matrix of daily returns for all 6 stocks, annotated and color-scaled from green (high correlation) to red (lower). The upper triangle is masked to avoid redundancy — each of the 15 pairs appears exactly once.

**Key observations.** All pairwise return correlations are positive, ranging from 0.53 (NVDA/META) to 0.71 (MSFT/GOOGL). MSFT is the most correlated stock with the rest of the universe. NVDA and META consistently sit at the bottom of correlation rankings, meaning their daily returns are most idiosyncratic relative to peers.

**Implications for Phase 3.** High return correlation is necessary but not sufficient for cointegration. MSFT/GOOGL and MSFT/AAPL are the strongest candidates based on return co-movement and should be prioritised in cointegration tests.

---

## 4. Ranked Pairwise Correlation Table (`pair_correlations.csv`)

**What it shows.** All 15 possible pairs of the 6 stocks, ranked from highest to lowest Pearson return correlation. Saved as a CSV for direct use in Phase 3 pair prioritisation.

**Key observations.**

| Rank | Pair          | Correlation |
|------|---------------|-------------|
| 1    | MSFT / GOOGL  | 0.709       |
| 2    | MSFT / AAPL   | 0.700       |
| 3    | MSFT / AMZN   | 0.682       |
| 4    | MSFT / NVDA   | 0.662       |
| 5    | GOOGL / AMZN  | 0.641       |
| 6    | GOOGL / AAPL  | 0.622       |
| 7    | AMZN / META   | 0.611       |
| 8    | GOOGL / META  | 0.610       |
| 9    | MSFT / META   | 0.609       |
| 10   | AAPL / AMZN   | 0.595       |
| 11   | NVDA / AMZN   | 0.584       |
| 12   | GOOGL / NVDA  | 0.576       |
| 13   | NVDA / AAPL   | 0.572       |
| 14   | AAPL / META   | 0.533       |
| 15   | NVDA / META   | 0.525       |

MSFT dominates the top of the ranking — it appears in 5 of the top 9 pairs. The correlation floor of ~0.53 across all 15 pairs means every combination has some systematic co-movement, so rejecting pairs on return correlation alone would be premature.

**Implications for Phase 3.** This ranked table serves as the first filter for Phase 3. The top 6 pairs (correlation ≥ 0.62) will be prioritised for Engle-Granger and Johansen cointegration testing. Pairs involving NVDA will be tested over the restricted 2018–2022 window only.

---

## 5. Cumulative Returns (`cumulative_returns.png`)

**What it shows.** All six stocks on a single chart, indexed to 100 at the start of 2018, showing the compounded growth of each through end of 2025.

**Key observations.**

| Ticker | Total Return |
|--------|-------------|
| NVDA   | +3,705%     |
| AAPL   | +577%       |
| MSFT   | +517%       |
| GOOGL  | +489%       |
| AMZN   | +291%       |
| META   | +270%       |

NVDA separates from the group decisively from mid-2023 onward (the AI-driven GPU demand surge). MSFT, AAPL, and GOOGL track each other closely through 2018–2021, diverge modestly during 2022, then re-converge. META collapsed ~75% in 2022 before recovering. The chart shows two clear regimes: a broad bull market (2018–2021), a sharp synchronised correction (2022), and a divergent recovery (2023–2025).

**Implications for Phase 3.** NVDA's exponential divergence means any pair involving NVDA will have a non-stationary spread over the full period. The MSFT/GOOGL/AAPL cluster is the most promising region — their price levels tracked together for much of the period, which is exactly the visual signature of a cointegrated pair.

---

## 6. Rolling 30-Day Annualised Volatility (`rolling_volatility.png`)

**What it shows.** For each stock, the 30-day rolling standard deviation of daily returns, annualised by multiplying by √252, plotted over the full period.

**Key observations.**

| Ticker | Mean Vol | Max Vol |
|--------|----------|---------|
| NVDA   | 48.3%    | 121.6%  |
| META   | 38.2%    | 96.6%   |
| AMZN   | 32.4%    | 68.4%   |
| GOOGL  | 29.6%    | 80.1%   |
| AAPL   | 28.3%    | 95.6%   |
| MSFT   | 26.0%    | 99.8%   |

Three distinct volatility spikes are visible across all stocks simultaneously: March 2020 (COVID crash), late 2022 (inflation/rate shock), and isolated earnings spikes in 2024–2025. Outside these events, vol is relatively stable for MSFT, AAPL, and GOOGL, while NVDA's vol remains structurally elevated from 2023 onward.

**Implications for Phase 3.** Rolling volatility informs z-score window sizing and position sizing. A volatility-normalised spread is more robust than a fixed-threshold approach. The 2022 bear market is the critical stress test period to examine in backtesting.

---

## 7. Boxplot by Year (`boxplot_by_year.png`)

**What it shows.** Six subplots, one per stock, showing box-and-whisker distributions of daily returns for each calendar year from 2018 to 2025.

**Key observations.** The median return line sits very close to zero across all stocks and all years. The interquartile range is the key signal: 2020 and 2022 boxes are visibly taller for every stock, reflecting elevated dispersion during the COVID crash and the 2022 rate-shock bear market. NVDA's boxes are consistently taller than peers from 2020 onward. META's 2022 box is the tallest single box in the entire grid. By contrast, 2019 and 2023–2025 boxes are compact and centred near zero.

**Implications for Phase 3.** The year-by-year variation in dispersion means a single set of backtest parameters trained across the full 8 years may not reflect current conditions. Consider walk-forward validation with annual parameter re-estimation. The 2022 stress year is the most important out-of-sample test period.

---

## 8. Pairwise Price Level Scatter Plots (`pairwise_price_scatter.png`)

**What it shows.** All 15 pair combinations plotted as scatter plots of raw closing prices, with a linear regression line and R value annotated on each panel.

**Key observations.**

| Pair           | Price R | Price R² |
|----------------|---------|----------|
| MSFT / AAPL    | 0.966   | 0.933    |
| NVDA / META    | 0.949   | 0.900    |
| MSFT / GOOGL   | 0.935   | 0.874    |
| GOOGL / AAPL   | 0.927   | 0.859    |
| AMZN / META    | 0.901   | 0.811    |
| MSFT / NVDA    | 0.890   | 0.792    |
| GOOGL / NVDA   | 0.897   | 0.804    |
| MSFT / AMZN    | 0.883   | 0.779    |
| MSFT / META    | 0.880   | 0.774    |
| GOOGL / AMZN   | 0.877   | 0.769    |
| GOOGL / META   | 0.868   | 0.753    |
| AAPL / AMZN    | 0.852   | 0.726    |
| NVDA / AAPL    | 0.833   | 0.694    |
| NVDA / AMZN    | 0.822   | 0.676    |
| AAPL / META    | 0.803   | 0.645    |

MSFT/AAPL shows the tightest linear relationship. Pairs involving NVDA post-2023 show a distinctive elbow or curved pattern, meaning the relationship is non-linear and unstable. META pairs show a characteristic hysteresis caused by META's 2022 collapse and recovery — two separate price regimes visible in the scatter.

**Implications for Phase 3.** High price-level R² is suggestive but not diagnostic for cointegration — two independent random walks can trend together. The Engle-Granger test will formally separate spurious trending relationships from true cointegration. MSFT/AAPL and MSFT/GOOGL are highest priority.

---

## 9. Pairwise Return Scatter Plots (`pairwise_return_scatter.png`)

**What it shows.** All 15 pair combinations plotted as scatter plots of daily returns (not price levels), with a linear regression line and R value on each panel. This directly visualises the day-to-day co-movement between stocks — the return-level relationship that underpins correlation.

**Key observations.** Return scatter plots are noisier than price scatter plots because individual daily returns contain a large idiosyncratic component, but the direction and strength of the linear relationship is clearly visible. The top-ranked pairs (MSFT/GOOGL, MSFT/AAPL) show the most compact scatter clouds around the regression line, confirming their high correlation. Pairs involving NVDA or META show wider, more dispersed clouds — consistent with their lower correlations and higher idiosyncratic volatility. Notably, extreme observations (large single-day moves) pull the regression line but cluster along it, indicating that co-movement holds even during market stress events. This is a positive sign for pairs trading — the relationship does not break down during volatility spikes.

**Implications for Phase 3.** The return scatter R values rank pairs identically to the correlation table in analysis 4, as expected. The visual confirms there are no obvious non-linearities in return co-movement (unlike the price scatter for NVDA pairs). All 15 pairs show a positive linear return relationship, which means every pair has some inherent mean-reversion potential — the cointegration test will determine which relationships are formally stationary.

---

## 10. Beta Analysis (`beta_analysis.png` + `beta_analysis.csv`)

**What it shows.** Each stock's beta relative to the S&P 500, computed by regressing daily stock returns on daily S&P 500 returns. Results shown as a bar chart and saved with full regression statistics.

**Key observations.**

| Ticker | Beta  | Alpha (daily %) | R²    |
|--------|-------|-----------------|-------|
| NVDA   | 1.822 | 0.134           | 0.485 |
| META   | 1.318 | 0.028           | 0.388 |
| AAPL   | 1.216 | 0.048           | 0.599 |
| AMZN   | 1.178 | 0.027           | 0.450 |
| MSFT   | 1.176 | 0.043           | 0.661 |
| GOOGL  | 1.156 | 0.045           | 0.535 |

Every stock has a beta above 1, meaning all six amplify S&P 500 moves. NVDA's beta of 1.82 means a 1% S&P day typically produces a ~1.8% NVDA move. Despite this, NVDA's R² is only 0.49 — nearly half its daily variance is idiosyncratic. MSFT has the highest R² (0.66), meaning it is the most market-like of the six. All alpha values are small but positive, consistent with the tech sector outperforming the broad market over this period.

**Implications for Phase 3.** Because all six stocks have similar betas (except NVDA), the spread between most pairs will be partially market-neutral by construction. MSFT/GOOGL and MSFT/AAPL — with nearly identical betas (1.176 vs 1.156, 1.176 vs 1.216) — are the most naturally beta-neutral combinations in the universe. NVDA's materially higher beta means any pair with NVDA will carry a residual long-market exposure unless an explicit beta hedge is applied.

---

## 11. Benchmark Comparison (`cumulative_vs_benchmark.png`)

**What it shows.** All six stocks plotted as base-100 cumulative return indices on the same chart as the S&P 500 (shown as a thick black dashed line), making it possible to directly assess whether each stock outperformed or underperformed the market benchmark over the 8-year period.

**Key observations.** Every stock in the universe outperformed the S&P 500 over 2018-2025, though the magnitude varies widely. NVDA's divergence from 2023 onward is the most striking feature — by end-2025 its index value is roughly 37 times the starting level, compared to approximately 4x for the S&P 500. MSFT, AAPL, and GOOGL track each other closely and approximately double the S&P 500's return over the period. AMZN and META underperform the other four tech stocks but still outperform the S&P 500. During 2022 all six stocks fall significantly below the S&P 500's starting-level reference line before recovering.

**Implications for Phase 3.** The benchmark comparison confirms that all six stocks carry excess return relative to the market — their spreads will have a positive drift component that a naive pairs strategy must account for. The OLS hedge ratio in Phase 3 removes the level relationship, but the relative drift between pairs over subperiods (visible in the chart) explains why the hedge ratio must be recalibrated periodically.

---

## 12. Rolling 60-Day Correlation — Top 6 Pairs (`rolling_correlation.png`)

**What it shows.** For each of the six pairs with the highest average return correlation, a time series of the 60-day rolling Pearson correlation between daily returns, with the overall period mean overlaid as a dashed line. This shows whether correlations between stocks are stable or highly regime-dependent.

**Key observations.** Rolling correlations are strongly regime-dependent for all six pairs. Every pair shows a sharp upward spike during March 2020 (COVID crash) and again during the 2022 bear market, as all stocks moved together under macro stress. In calmer periods (2018-2019, 2023-2025), correlations fall materially below their full-period means. NVDA-involving pairs show the most dramatic collapse in rolling correlation from mid-2023 onward as NVDA decoupled from peers on AI-driven demand. The MSFT/GOOGL and MSFT/AAPL pairs maintain the most stable above-mean rolling correlation throughout.

**Implications for Phase 3.** The regime-dependence of correlations is a direct warning for the spread strategy: a pair with a stable long-run EG cointegration result may still have unstable short-run dynamics. The 2022 spike confirms that macro stress increases co-movement — but the strategy relies on mean reversion of the *spread*, not on the stocks moving together.

---

## 13. VIX Sensitivity Analysis (`vix_sensitivity.png` + `vix_sensitivity.csv`)

**What it shows.** Three panels examining how each stock's returns and volatility behave in relation to the VIX (fear index). Panel 1 shows Pearson correlation of each stock's daily return with the daily VIX return. Panel 2 compares average daily returns on high-VIX days (VIX level > 25) versus low-VIX days. Panel 3 compares annualised volatility in each regime. High-VIX days numbered 349 out of 2,009 trading days (17.4%); low-VIX days numbered 1,660.

**Key observations.**

| Ticker | Corr w/ VIX | Avg Ret High-VIX | Avg Ret Low-VIX | Ann Vol High-VIX | Ann Vol Low-VIX |
|--------|-------------|-----------------|-----------------|-----------------|-----------------|
| MSFT   | −0.573      | −0.192%         | +0.169%         | 46.1%           | 22.8%           |
| GOOGL  | −0.542      | −0.251%         | +0.183%         | 44.2%           | 27.3%           |
| NVDA   | −0.529      | −0.311%         | +0.348%         | 68.9%           | 46.6%           |
| AAPL   | −0.545      | −0.256%         | +0.192%         | 50.5%           | 24.6%           |
| AMZN   | −0.499      | −0.307%         | +0.175%         | 49.3%           | 30.2%           |
| META   | −0.435      | −0.402%         | +0.205%         | 59.1%           | 36.5%           |

All six stocks are negatively correlated with VIX returns — stocks fall on days when the fear index rises. On high-VIX days, average daily returns are negative for every stock, while on low-VIX days all stocks show positive average returns. Annualised volatility is approximately double in the high-VIX regime compared to the low-VIX regime for MSFT, AAPL, and GOOGL. NVDA and META exhibit the most extreme volatility amplification.

**Implications for Phase 3.** High-VIX periods are the most dangerous environment for a pairs strategy. When VIX spikes, spread dynamics can deviate from historical patterns for extended periods, causing z-score entries to be false signals. A sentiment or VIX filter — avoiding new position entries when VIX > 25 — is a practical risk management addition that Phase 4 could implement.

---

## 14. EDA Conclusion (`phase2_eda_conclusion.txt`)

**What it shows.** A written synthesis of all EDA findings, translating the quantitative observations into a structured view of which pairs are most likely to pass formal cointegration tests, and why. Saved separately as a standalone text file.

**Key conclusions documented:** AMZN/META, MSFT/AAPL, and MSFT/GOOGL are the priority pairs for Phase 3; NVDA pairs are expected to fail due to structural break; 2022 is the critical stress year for all backtests; VIX sensitivity suggests a regime-aware strategy is preferable to an always-on approach.

---

## Summary: Heading into Phase 3

The EDA across all 14 analyses points clearly to **three priority pairs** for cointegration testing:

1. **MSFT / GOOGL** — highest return correlation (0.709), similar betas (1.176 vs 1.156), clean price scatter (R=0.935), compact return scatter
2. **MSFT / AAPL** — second highest return correlation (0.700), similar betas (1.176 vs 1.216), tightest price scatter in the universe (R=0.966)
3. **AMZN / META** — moderate return correlation (0.611), both exposed to consumer/advertising cycle, neither affected by NVDA's AI structural break

**Additional findings from analyses 11–14:**
- All six stocks outperformed the S&P 500 over 2018-2025; NVDA at 37x was the dominant outlier
- Rolling 60-day correlations are highly regime-dependent — all pairs spike during macro stress (2020, 2022) and compress in calm periods
- VIX sensitivity is strong and uniform: all stocks post negative average returns on high-VIX days (VIX > 25) and roughly double their volatility; AMZN and META show the largest average return deterioration in fear regimes
- The EDA conclusion (saved separately) identifies AMZN/META as the pair best positioned for cointegration given the absence of a NVDA-style structural break in either stock

**Caution flags:**
- NVDA decoupled from all peers from mid-2023 — restrict any NVDA cointegration test to 2018–2022
- META has a structural regime break in 2022 — single-day kurtosis events inflate spread risk
- 2022 is the critical stress year for all backtests — any strategy that survives 2022 has demonstrated genuine robustness
- High-VIX environments (349 days, 17.4% of the sample) are the most hostile for pairs mean-reversion strategies

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/reports/summary_stats.csv` | Mean, std, min, max, skewness, kurtosis per ticker |
| `outputs/reports/pair_correlations.csv` | Ranked pairwise return correlations, all 15 pairs |
| `outputs/reports/beta_analysis.csv` | Beta, alpha, R², p-value vs S&P 500 per ticker |
| `outputs/reports/vix_sensitivity.csv` | VIX correlation, avg returns, ann. vol by regime |
| `outputs/reports/phase2_eda_conclusion.txt` | Written EDA conclusion and pair-selection rationale |
| `outputs/charts/hist_returns.png` | Return distributions with KDE, 2×3 grid |
| `outputs/charts/correlation_heatmap.png` | Pearson correlation heatmap (lower triangle) |
| `outputs/charts/cumulative_returns.png` | Base-100 cumulative returns, all 6 stocks |
| `outputs/charts/cumulative_vs_benchmark.png` | All 6 stocks vs S&P 500 on same chart |
| `outputs/charts/rolling_volatility.png` | 30-day rolling annualised volatility |
| `outputs/charts/boxplot_by_year.png` | Annual return boxplots per stock |
| `outputs/charts/pairwise_price_scatter.png` | Price-level scatter plots, all 15 pairs |
| `outputs/charts/pairwise_return_scatter.png` | Return scatter plots, all 15 pairs |
| `outputs/charts/beta_analysis.png` | Beta bar chart vs S&P 500 |
| `outputs/charts/rolling_correlation.png` | 60-day rolling correlations, top 6 pairs |
| `outputs/charts/vix_sensitivity.png` | VIX sensitivity: correlation, returns, vol |
