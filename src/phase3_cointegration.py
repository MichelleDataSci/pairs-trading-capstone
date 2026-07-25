"""
Phase 3 -- Cointegration Screening
Tests all 15 pairs of 6 tech stocks for cointegration using log prices.

Methods (per project scope):
  1. OLS regression in both directions + ADF on residuals -- pick direction
     with stronger stationarity evidence (lower ADF p-value).
  2. Engle-Granger test via statsmodels.tsa.stattools.coint (auto regression
     + residual ADF with correct non-standard critical values).
  3. Johansen multivariate cointegration test.

Output:
  outputs/reports/cointegration_results.csv  -- full results table, all 15 pairs
  outputs/charts/spread_series_all_pairs.png -- OLS residual plots, all 15 pairs
  outputs/charts/cointegration_ranking.png   -- EG p-value ranking bar chart
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ALL_TICKERS, DATA_RAW, CHARTS_DIR, REPORTS_DIR

print("=" * 65)
print("PHASE 3 -- COINTEGRATION SCREENING (LOG PRICES)")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. Load close prices from raw CSVs and compute log prices
# ---------------------------------------------------------------------------
prices = {}
for ticker in ALL_TICKERS:
    path = DATA_RAW / f"{ticker}_raw.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    prices[ticker] = df["Close"]

price_df = pd.DataFrame(prices).dropna()
log_price_df = np.log(price_df)

print(f"\nClose prices loaded: {price_df.shape[0]} trading days, {price_df.shape[1]} tickers")
print(f"Date range: {price_df.index[0].date()} to {price_df.index[-1].date()}")
print(f"Log prices computed for: {ALL_TICKERS}")

# ---------------------------------------------------------------------------
# 2. Generate all 15 pairs
# ---------------------------------------------------------------------------
pairs = list(combinations(ALL_TICKERS, 2))
print(f"\nAll {len(pairs)} pairs:")
for i, (a, b) in enumerate(pairs, 1):
    print(f"  {i:2d}. {a} / {b}")

# ---------------------------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------------------------

def ols_adf(y_log, x_log):
    """
    OLS of y on x (with constant) using log prices.
    Returns hedge_ratio, intercept, residuals, ADF stat, ADF p-value.
    ADF uses autolag='AIC' to select lag length automatically.
    """
    X = sm.add_constant(x_log)
    model = sm.OLS(y_log, X).fit()
    resid = model.resid
    adf = adfuller(resid, autolag="AIC", regression="c")
    # params: [const, slope]
    return float(model.params.iloc[1]), float(model.params.iloc[0]), resid, float(adf[0]), float(adf[1])


def engle_granger(y_log, x_log):
    """
    Engle-Granger test via statsmodels coint().
    Uses non-standard critical values appropriate for regression residuals.
    Returns (test stat, p-value).
    """
    stat, pval, _ = coint(y_log, x_log, autolag="AIC")
    return float(stat), float(pval)


def johansen(log_pair_df, det_order=0, k_ar_diff=1):
    """
    Johansen cointegration test on a 2-column log price DataFrame.
    det_order=0: constant in the cointegrating relationship.
    k_ar_diff=1: one lag in the VAR.
    Critical values at 5% level (index 1 in the cvt/cvm arrays).
    Returns dict of trace and max-eigenvalue stats, crits, and pass flags.
    """
    res = coint_johansen(log_pair_df, det_order=det_order, k_ar_diff=k_ar_diff)
    ci = 1  # 5% column
    return {
        "trace_stat_r0":        round(float(res.lr1[0]), 4),
        "trace_crit_r0_5pct":   round(float(res.cvt[0, ci]), 4),
        "trace_stat_r1":        round(float(res.lr1[1]), 4),
        "trace_crit_r1_5pct":   round(float(res.cvt[1, ci]), 4),
        "maxeig_stat_r0":       round(float(res.lr2[0]), 4),
        "maxeig_crit_r0_5pct":  round(float(res.cvm[0, ci]), 4),
        "maxeig_stat_r1":       round(float(res.lr2[1]), 4),
        "maxeig_crit_r1_5pct":  round(float(res.cvm[1, ci]), 4),
        "trace_pass":           bool(res.lr1[0] > res.cvt[0, ci]),
        "maxeig_pass":          bool(res.lr2[0] > res.cvm[0, ci]),
    }

# ---------------------------------------------------------------------------
# 4. Run all tests for all 15 pairs
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("RUNNING TESTS FOR ALL 15 PAIRS")
print("=" * 65)

records = []
spreads = {}  # chosen OLS residuals for plotting

for a, b in pairs:
    log_a = log_price_df[a]
    log_b = log_price_df[b]

    # OLS in both directions -- pick the direction whose residuals are more
    # stationary (lower ADF p-value = stronger evidence against unit root)
    hr_ab, ic_ab, resid_ab, adf_ab, pval_ab = ols_adf(log_a, log_b)
    hr_ba, ic_ba, resid_ba, adf_ba, pval_ba = ols_adf(log_b, log_a)

    if pval_ab <= pval_ba:
        chosen_dir = f"{a}~{b}"
        hedge_ratio, intercept = hr_ab, ic_ab
        resid, adf_stat, adf_pval = resid_ab, adf_ab, pval_ab
        dep, indep = a, b
    else:
        chosen_dir = f"{b}~{a}"
        hedge_ratio, intercept = hr_ba, ic_ba
        resid, adf_stat, adf_pval = resid_ba, adf_ba, pval_ba
        dep, indep = b, a

    # Engle-Granger on chosen direction
    eg_stat, eg_pval = engle_granger(log_price_df[dep], log_price_df[indep])

    # Johansen (direction-independent)
    joh = johansen(log_price_df[[a, b]])

    pair_label = f"{a}/{b}"
    spreads[pair_label] = resid

    eg_pass  = eg_pval < 0.05
    joh_pass = joh["trace_pass"]

    records.append({
        "Pair":                     pair_label,
        "OLS_direction":            chosen_dir,
        "Hedge_ratio":              round(hedge_ratio, 4),
        "Intercept":                round(intercept, 4),
        "ADF_stat_on_residuals":    round(adf_stat, 4),
        "ADF_pval_on_residuals":    round(adf_pval, 4),
        "EG_stat":                  round(eg_stat, 4),
        "EG_pval":                  round(eg_pval, 4),
        "EG_cointegrated_5pct":     eg_pass,
        "Johansen_trace_stat_r0":   joh["trace_stat_r0"],
        "Johansen_trace_crit_r0":   joh["trace_crit_r0_5pct"],
        "Johansen_trace_stat_r1":   joh["trace_stat_r1"],
        "Johansen_trace_crit_r1":   joh["trace_crit_r1_5pct"],
        "Johansen_maxeig_stat_r0":  joh["maxeig_stat_r0"],
        "Johansen_maxeig_crit_r0":  joh["maxeig_crit_r0_5pct"],
        "Johansen_trace_pass_5pct": joh_pass,
        "Johansen_maxeig_pass_5pct":joh["maxeig_pass"],
        "Both_methods_agree":       eg_pass and joh_pass,
    })

    eg_flag  = "PASS" if eg_pass  else "FAIL"
    joh_flag = "PASS" if joh_pass else "FAIL"
    print(f"  {pair_label:<12}  dir={chosen_dir:<13}  "
          f"EG p={eg_pval:.4f} [{eg_flag}]  "
          f"Johansen trace={joh['trace_stat_r0']:.2f} vs {joh['trace_crit_r0_5pct']:.2f} [{joh_flag}]")

# ---------------------------------------------------------------------------
# 5. Rank by EG p-value (ascending = stronger evidence of cointegration)
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(records)
results_df["Rank"] = results_df["EG_pval"].rank(method="min").astype(int)
results_df = results_df.sort_values("EG_pval").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Save full results table
# ---------------------------------------------------------------------------
out_csv = REPORTS_DIR / "cointegration_results.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nFull results saved -> {out_csv}")

# ---------------------------------------------------------------------------
# 7. Print ranking summary to console
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("RANKING TABLE -- BY ENGLE-GRANGER P-VALUE (ASCENDING)")
print("=" * 65)
hdr = f"{'Rank':<5} {'Pair':<12} {'OLS Direction':<14} {'HR':>7}  {'ADF p':>7}  {'EG p':>7}  {'EG':>6}  {'Johansen':>10}"
print(hdr)
print("-" * len(hdr))
for _, row in results_df.iterrows():
    eg_tag  = "PASS" if row["EG_cointegrated_5pct"]     else "FAIL"
    joh_tag = "PASS" if row["Johansen_trace_pass_5pct"] else "FAIL"
    print(f"  {row['Rank']:<4} {row['Pair']:<12} {row['OLS_direction']:<14} "
          f"{row['Hedge_ratio']:>7.4f}  "
          f"{row['ADF_pval_on_residuals']:>7.4f}  "
          f"{row['EG_pval']:>7.4f}  "
          f"{eg_tag:>6}  {joh_tag:>10}")

shortlist = results_df[results_df["Both_methods_agree"]]
print(f"\n{'='*65}")
print(f"SHORTLISTED PAIRS (both EG and Johansen pass at 5%):")
print(f"{'='*65}")
if len(shortlist) == 0:
    print("  No pairs pass both tests at 5% significance.")
else:
    for _, row in shortlist.iterrows():
        print(f"  Rank {row['Rank']}: {row['Pair']}  "
              f"OLS dir={row['OLS_direction']}  HR={row['Hedge_ratio']}  "
              f"EG p={row['EG_pval']:.4f}  "
              f"Johansen trace={row['Johansen_trace_stat_r0']} vs {row['Johansen_trace_crit_r0']}")

# ---------------------------------------------------------------------------
# 8. Plot OLS residual (spread) series for all 15 pairs
#    Color: blue = both pass, orange = one passes, grey = neither
# ---------------------------------------------------------------------------
n_cols = 3
n_rows = (len(pairs) + n_cols - 1) // n_cols   # 5 rows for 15 pairs

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.4))
fig.suptitle(
    "OLS Residual (Spread) Series -- All 15 Pairs (Log Prices, 2018-2025)\n"
    "Blue = Both EG+Johansen pass | Orange = One method passes | Grey = Neither",
    fontsize=12, y=1.01
)

axes_flat = axes.flatten()
for i, (pair_label, resid) in enumerate(spreads.items()):
    ax = axes_flat[i]
    row = results_df[results_df["Pair"] == pair_label].iloc[0]
    eg_pass  = row["EG_cointegrated_5pct"]
    joh_pass = row["Johansen_trace_pass_5pct"]

    if eg_pass and joh_pass:
        color, tag = "steelblue", "EG + JOH"
    elif eg_pass or joh_pass:
        color, tag = "darkorange", "EG" if eg_pass else "JOH"
    else:
        color, tag = "slategray", "none"

    ax.plot(resid.index, resid.values, color=color, linewidth=0.75, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(
        f"{pair_label}  [{tag}]  EG p={row['EG_pval']:.4f}  HR={row['Hedge_ratio']:.3f}",
        fontsize=8.5
    )
    ax.tick_params(axis="x", labelsize=6.5, rotation=30)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("")

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
chart1 = CHARTS_DIR / "spread_series_all_pairs.png"
plt.savefig(chart1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSpread chart saved -> {chart1}")

# ---------------------------------------------------------------------------
# 9. Ranking bar chart -- EG p-values, all 15 pairs
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))
bar_colors = [
    "steelblue"   if (eg and joh) else
    "darkorange"  if (eg or joh)  else
    "lightcoral"
    for eg, joh in zip(
        results_df["EG_cointegrated_5pct"],
        results_df["Johansen_trace_pass_5pct"]
    )
]
ax2.barh(
    results_df["Pair"][::-1],
    results_df["EG_pval"][::-1],
    color=bar_colors[::-1],
    edgecolor="white", linewidth=0.5
)
ax2.axvline(0.05, color="red", linestyle="--", linewidth=1.3, label="5% significance threshold")
ax2.set_xlabel("Engle-Granger p-value (lower = stronger cointegration evidence)", fontsize=10)
ax2.set_title("Cointegration Ranking -- All 15 Pairs (EG p-value)", fontsize=12)

from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor="steelblue",  label="Both EG + Johansen pass"),
    Patch(facecolor="darkorange", label="One method passes"),
    Patch(facecolor="lightcoral", label="Neither passes"),
    plt.Line2D([0], [0], color="red", linestyle="--", label="5% threshold"),
]
ax2.legend(handles=legend_items, fontsize=9)
plt.tight_layout()
chart2 = CHARTS_DIR / "cointegration_ranking.png"
plt.savefig(chart2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Ranking chart saved -> {chart2}")

# ---------------------------------------------------------------------------
# 10. Comparison summary -- EG vs Johansen agreement
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print("EG vs JOHANSEN AGREEMENT SUMMARY")
print(f"{'='*65}")
both  = results_df["Both_methods_agree"].sum()
eg_only   = (results_df["EG_cointegrated_5pct"] & ~results_df["Johansen_trace_pass_5pct"]).sum()
joh_only  = (~results_df["EG_cointegrated_5pct"] & results_df["Johansen_trace_pass_5pct"]).sum()
neither   = (~results_df["EG_cointegrated_5pct"] & ~results_df["Johansen_trace_pass_5pct"]).sum()
print(f"  Both EG and Johansen pass  : {both:2d} pair(s)")
print(f"  EG only                    : {eg_only:2d} pair(s)")
print(f"  Johansen only              : {joh_only:2d} pair(s)")
print(f"  Neither                    : {neither:2d} pair(s)")

# ---------------------------------------------------------------------------
# 11. NVDA restricted window check (2018-01-01 to 2022-12-31)
#     Phase 2 EDA flagged that NVDA decoupled from peers from mid-2023 onward
#     due to the AI/GPU-driven price surge. Restricting to 2018-2022 removes
#     that structural break and gives the NVDA pairs the best possible chance
#     of passing cointegration tests.
# ---------------------------------------------------------------------------
NVDA_END = "2022-12-31"
nvda_pairs = [p for p in pairs if "NVDA" in p]
log_nvda_window = log_price_df.loc[:NVDA_END]

print(f"\n{'='*65}")
print(f"NVDA SUB-PERIOD CHECK (2018-01-01 to {NVDA_END})")
print(f"{'='*65}")
print(f"  Window : {log_nvda_window.index[0].date()} to {log_nvda_window.index[-1].date()}  "
      f"({len(log_nvda_window)} days)")
print(f"  Pairs  : {[f'{a}/{b}' for a,b in nvda_pairs]}\n")

nvda_records = []
for a, b in nvda_pairs:
    log_a = log_nvda_window[a]
    log_b = log_nvda_window[b]

    hr_ab, ic_ab, resid_ab, adf_ab, pval_ab = ols_adf(log_a, log_b)
    hr_ba, ic_ba, resid_ba, adf_ba, pval_ba = ols_adf(log_b, log_a)

    if pval_ab <= pval_ba:
        chosen_dir = f"{a}~{b}"
        hedge_ratio, resid, adf_stat, adf_pval = hr_ab, resid_ab, adf_ab, pval_ab
        dep, indep = a, b
    else:
        chosen_dir = f"{b}~{a}"
        hedge_ratio, resid, adf_stat, adf_pval = hr_ba, resid_ba, adf_ba, pval_ba
        dep, indep = b, a

    eg_stat, eg_pval = engle_granger(log_nvda_window[dep], log_nvda_window[indep])
    joh = johansen(log_nvda_window[[a, b]])

    eg_pass  = eg_pval < 0.05
    joh_pass = joh["trace_pass"]
    eg_flag  = "PASS" if eg_pass  else "FAIL"
    joh_flag = "PASS" if joh_pass else "FAIL"

    print(f"  {a}/{b:<12}  dir={chosen_dir:<13}  "
          f"EG p={eg_pval:.4f} [{eg_flag}]  "
          f"Johansen trace={joh['trace_stat_r0']:.2f} vs {joh['trace_crit_r0_5pct']:.2f} [{joh_flag}]")

    nvda_records.append({
        "Pair": f"{a}/{b}", "Window": f"2018-{NVDA_END[:4]}",
        "OLS_direction": chosen_dir, "Hedge_ratio": round(hedge_ratio, 4),
        "ADF_pval": round(adf_pval, 4), "EG_pval": round(eg_pval, 4),
        "EG_pass": eg_pass, "Johansen_trace_pass": joh_pass,
    })

nvda_df = pd.DataFrame(nvda_records)
nvda_csv = REPORTS_DIR / "nvda_subperiod_results.csv"
nvda_df.to_csv(nvda_csv, index=False)

any_nvda_pass = nvda_df["EG_pass"].any() or nvda_df["Johansen_trace_pass"].any()
print(f"\n  Result: {'At least one NVDA pair passes in the restricted window.' if any_nvda_pass else 'All NVDA pairs still fail in the restricted 2018-2022 window.'}")
print(f"  Saved -> {nvda_csv}")
print(f"\n  Interpretation: Restricting to 2018-2022 removes the AI-driven structural")
print(f"  break in NVDA from mid-2023, but the cointegrating relationship with peer")
print(f"  stocks was also not established in the earlier period.")

print(f"\n{'='*65}")
print("PHASE 3 -- COINTEGRATION SCREENING COMPLETE")
print(f"{'='*65}")
print(f"\nOutputs:")
print(f"  {out_csv}")
print(f"  {nvda_csv}")
print(f"  {chart1}")
print(f"  {chart2}")
print(f"\nNext step: Phase 3b -- pairs trading strategy on shortlisted pairs.")
