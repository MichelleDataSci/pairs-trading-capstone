"""
Phase 4 -- Testing Mean Reversion on Unseen Data
Pair: AMZN / META (identified as cointegrated in Phase 3a)

Procedure (per project brief):
  Step 1  Cointegrated pair already identified (Phase 3a) -- AMZN/META
  Step 2  Fit OLS on full training period log prices (2018-01-01 to 2025-12-31)
  Step 3  Compute FIXED mean and std of residuals from 2025 only (last year of training)
  Step 4  Download unseen data (2026-01-01 to 2026-07-31) and apply same model
  Step 5  Standardise testing-period residuals using the 2025 mean and std
  Step 6  Plot Z-scores over the testing period -- assess mean-reversion behaviour
  Step 7  Count long signals (Z < -2, exit Z >= 0) and short signals (Z > +2, exit Z <= 0)

Key differences from Phase 3b:
  - Training covers the FULL 2018-2025 history (not just 2018-2021)
  - Z-score uses a FIXED anchor (2025 mean/std) -- not a rolling window
  - Test data is genuinely unseen: 2026 was not available when the model was built
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import yfinance as yf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_RAW, CHARTS_DIR, REPORTS_DIR

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DEP          = "AMZN"
INDEP        = "META"
TRAIN_END    = "2025-12-31"
NORM_START   = "2025-01-01"   # last 12 months of training used for mean/std
TEST_START   = "2026-01-01"
TEST_END     = "2026-07-31"
Z_ENTRY      = 2.0
Z_EXIT       = 0.0

print("=" * 65)
print("PHASE 4 -- TESTING MEAN REVERSION ON UNSEEN DATA")
print("=" * 65)
print(f"\n  Pair           : {DEP} / {INDEP}")
print(f"  Training period: 2018-01-01 to {TRAIN_END}")
print(f"  Norm window    : {NORM_START} to {TRAIN_END} (mean and std for Z-score)")
print(f"  Test period    : {TEST_START} to {TEST_END} (unseen data)")
print(f"  Entry signal   : |Z| > {Z_ENTRY}")
print(f"  Exit signal    : Z crosses {Z_EXIT}")

# ---------------------------------------------------------------------------
# Step 2 -- Load full training history from raw CSVs and fit OLS
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print("STEP 2 -- FIT OLS ON FULL TRAINING PERIOD (2018-2025)")
print(f"{'='*65}")

train_prices = {}
for ticker in [DEP, INDEP]:
    path = DATA_RAW / f"{ticker}_raw.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    train_prices[ticker] = df["Close"]

train_df  = pd.DataFrame(train_prices).dropna()
train_df  = train_df.loc[:TRAIN_END]
log_train = np.log(train_df)

print(f"\n  Training data loaded: {len(train_df)} trading days")
print(f"  Date range: {train_df.index[0].date()} to {train_df.index[-1].date()}")

# OLS: log(AMZN) ~ log(META) + constant
X     = sm.add_constant(log_train[INDEP])
model = sm.OLS(log_train[DEP], X).fit()
beta  = float(model.params.iloc[1])   # hedge ratio
alpha = float(model.params.iloc[0])   # intercept
r2    = float(model.rsquared)

print(f"\n  OLS result (log({DEP}) ~ log({INDEP}) + constant):")
print(f"    Hedge ratio (beta) : {beta:.6f}")
print(f"    Intercept (alpha)  : {alpha:.6f}")
print(f"    R-squared          : {r2:.6f}")
print(f"    Spread = log({DEP}) - {beta:.4f} * log({INDEP}) - {alpha:.4f}")

# Compute training residuals (spread)
train_spread = log_train[DEP] - beta * log_train[INDEP] - alpha
print(f"\n  Training spread (full 2018-2025):")
print(f"    Mean : {train_spread.mean():.6f}")
print(f"    Std  : {train_spread.std():.6f}")

# ---------------------------------------------------------------------------
# Step 3 -- Fixed mean and std from 2025 residuals only
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"STEP 3 -- COMPUTE FIXED MEAN AND STD FROM {NORM_START} TO {TRAIN_END}")
print(f"{'='*65}")

norm_spread = train_spread.loc[NORM_START:TRAIN_END]
mu    = float(norm_spread.mean())
sigma = float(norm_spread.std())

print(f"\n  Normalisation window: {norm_spread.index[0].date()} to {norm_spread.index[-1].date()}")
print(f"  Number of days      : {len(norm_spread)}")
print(f"  Mean (mu)           : {mu:.6f}")
print(f"  Std  (sigma)        : {sigma:.6f}")
print(f"\n  These values are FIXED -- they will not update during the test period.")
print(f"  Z-score = (spread - {mu:.4f}) / {sigma:.4f}")

# ---------------------------------------------------------------------------
# Step 4 -- Download unseen 2026 data and compute out-of-sample residuals
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"STEP 4 -- DOWNLOAD UNSEEN DATA ({TEST_START} to {TEST_END})")
print(f"{'='*65}")

# yfinance end date is exclusive -- add one day to capture 31 July
test_end_exclusive = "2026-08-01"

test_prices = {}
for ticker in [DEP, INDEP]:
    print(f"  Downloading {ticker} ({TEST_START} to {TEST_END}) ...", end=" ")
    df_t = yf.download(
        ticker, start=TEST_START, end=test_end_exclusive,
        auto_adjust=True, progress=False
    )
    if df_t.empty:
        print("WARNING: no data returned.")
        continue
    if isinstance(df_t.columns, pd.MultiIndex):
        df_t.columns = df_t.columns.get_level_values(0)
    df_t.index.name = "Date"
    test_prices[ticker] = df_t["Close"]
    print(f"{len(df_t)} trading days")

if len(test_prices) < 2:
    print("\nERROR: could not download test data for both tickers. Exiting.")
    sys.exit(1)

test_df   = pd.DataFrame(test_prices).dropna()
log_test  = np.log(test_df)

print(f"\n  Test data loaded: {len(test_df)} trading days")
print(f"  Date range: {test_df.index[0].date()} to {test_df.index[-1].date()}")

# Apply SAME model (same beta and alpha from training)
test_spread = log_test[DEP] - beta * log_test[INDEP] - alpha

print(f"\n  Out-of-sample spread statistics:")
print(f"    Mean : {test_spread.mean():.6f}  (training mean was {mu:.6f})")
print(f"    Std  : {test_spread.std():.6f}  (training std  was {sigma:.6f})")

# ---------------------------------------------------------------------------
# Step 5 -- Standardise using FIXED training mean and std
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print("STEP 5 -- STANDARDISE TEST RESIDUALS (FIXED 2025 MEAN AND STD)")
print(f"{'='*65}")

zscore = (test_spread - mu) / sigma

print(f"\n  Z-score statistics over test period:")
print(f"    Min    : {zscore.min():.4f}")
print(f"    Max    : {zscore.max():.4f}")
print(f"    Mean   : {zscore.mean():.4f}  (should be near 0 if spread is still stationary)")
print(f"    Std    : {zscore.std():.4f}   (should be near 1 if spread variance is stable)")
print(f"    Days above +{Z_ENTRY} : {(zscore > Z_ENTRY).sum()}")
print(f"    Days below -{Z_ENTRY} : {(zscore < -Z_ENTRY).sum()}")

# ---------------------------------------------------------------------------
# Step 7 -- Count long and short trading signals
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print("STEP 7 -- TRADING SIGNALS")
print(f"{'='*65}")
print(f"  Long  spread: enter when Z < -{Z_ENTRY}, exit when Z >= {Z_EXIT}")
print(f"  Short spread: enter when Z >  {Z_ENTRY}, exit when Z <= {Z_EXIT}")

z_arr  = zscore.values
dates  = zscore.index
n      = len(z_arr)

position      = np.zeros(n, dtype=int)
pos           = 0
long_entries  = []
short_entries = []
long_exits    = []
short_exits   = []

signal_log = []   # full day-by-day record

for i in range(n):
    z = z_arr[i]

    if np.isnan(z):
        signal_log.append({
            "Date": str(dates[i].date()), "Z_score": None,
            "Position": 0, "Signal": "NaN"
        })
        continue

    action = "hold"

    # Check exit first
    if pos == 1 and z >= Z_EXIT:
        long_exits.append({"Date": dates[i], "Z_score": round(z, 4)})
        pos    = 0
        action = "exit long"
    elif pos == -1 and z <= Z_EXIT:
        short_exits.append({"Date": dates[i], "Z_score": round(z, 4)})
        pos    = 0
        action = "exit short"

    # Check entry (only if flat)
    if pos == 0:
        if z < -Z_ENTRY:
            long_entries.append({"Date": dates[i], "Z_score": round(z, 4)})
            pos    = 1
            action = "enter long"
        elif z > Z_ENTRY:
            short_entries.append({"Date": dates[i], "Z_score": round(z, 4)})
            pos    = -1
            action = "enter short"

    position[i] = pos
    signal_log.append({
        "Date":     str(dates[i].date()),
        "Z_score":  round(z, 4),
        "Position": pos,
        "Signal":   action,
    })

n_long  = len(long_entries)
n_short = len(short_entries)
n_total = n_long + n_short

print(f"\n  Results:")
print(f"    Long  entries (Z < -{Z_ENTRY}) : {n_long}")
print(f"    Short entries (Z >  {Z_ENTRY}) : {n_short}")
print(f"    Total signals                  : {n_total}")
print(f"    Days in market                 : {(position != 0).sum()} of {n} "
      f"({(position != 0).sum() / n * 100:.1f}%)")

if long_entries:
    print(f"\n  Long entry dates:")
    for e in long_entries:
        print(f"    {e['Date']}  Z={e['Z_score']:+.4f}")

if short_entries:
    print(f"\n  Short entry dates:")
    for e in short_entries:
        print(f"    {e['Date']}  Z={e['Z_score']:+.4f}")

# ---------------------------------------------------------------------------
# Save signal log CSV
# ---------------------------------------------------------------------------
signals_df = pd.DataFrame(signal_log)
signals_csv = REPORTS_DIR / "phase4_signals.csv"
signals_df.to_csv(signals_csv, index=False)
print(f"\n  Full signal log ({len(signals_df)} rows) saved -> {signals_csv}")

# Summary row for easy reporting
summary = {
    "Pair":                  f"{DEP}/{INDEP}",
    "Training_period":       f"2018-01-01 to {TRAIN_END}",
    "Norm_window":           f"{NORM_START} to {TRAIN_END}",
    "Test_period":           f"{TEST_START} to {TEST_END}",
    "Hedge_ratio":           round(beta, 6),
    "Intercept":             round(alpha, 6),
    "R2_training":           round(r2, 6),
    "Norm_mean_mu":          round(mu, 6),
    "Norm_std_sigma":        round(sigma, 6),
    "Test_days":             n,
    "Z_min":                 round(float(zscore.min()), 4),
    "Z_max":                 round(float(zscore.max()), 4),
    "Z_mean":                round(float(zscore.mean()), 4),
    "Z_std":                 round(float(zscore.std()), 4),
    "Long_signals":          n_long,
    "Short_signals":         n_short,
    "Total_signals":         n_total,
    "Days_in_market":        int((position != 0).sum()),
    "Pct_in_market":         round((position != 0).sum() / n * 100, 2),
}

summary_df = pd.DataFrame([summary])
summary_csv = REPORTS_DIR / "phase4_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"  Summary saved -> {summary_csv}")

# ---------------------------------------------------------------------------
# Step 6 -- Plot Z-scores over the test period
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print("STEP 6 -- CHART: Z-SCORES OVER TEST PERIOD (JAN-JUL 2026)")
print(f"{'='*65}")

fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(3, 1, hspace=0.45, figure=fig)

# ── Panel 1: Z-score with entry/exit bands and position shading ──
ax1 = fig.add_subplot(gs[0])

ax1.plot(dates, z_arr, color="steelblue", linewidth=1.0, label="Z-score", zorder=3)
ax1.axhline( Z_ENTRY, color="tomato",   linestyle="--", linewidth=1.1,
             label=f"+{Z_ENTRY} short entry")
ax1.axhline(-Z_ENTRY, color="seagreen", linestyle="--", linewidth=1.1,
             label=f"-{Z_ENTRY} long entry")
ax1.axhline( Z_EXIT,  color="black",    linestyle="-",  linewidth=0.6,
             label="0 exit line", alpha=0.6)

# Shade position periods
for i in range(1, n):
    if position[i] == 1:
        ax1.axvspan(dates[i - 1], dates[i], alpha=0.14, color="seagreen", linewidth=0)
    elif position[i] == -1:
        ax1.axvspan(dates[i - 1], dates[i], alpha=0.14, color="tomato",   linewidth=0)

# Mark entry points
for e in long_entries:
    ax1.axvline(e["Date"], color="seagreen", linewidth=1.0, linestyle=":", alpha=0.7)
for e in short_entries:
    ax1.axvline(e["Date"], color="tomato",   linewidth=1.0, linestyle=":", alpha=0.7)

ax1.set_title(
    f"AMZN/META Spread Z-score -- Unseen Test Period ({TEST_START} to {TEST_END})\n"
    f"Z = (spread - mu_2025) / sigma_2025   |   mu={mu:.4f}, sigma={sigma:.4f}   |   "
    f"Green shading = long, Red = short",
    fontsize=9.5
)
ax1.set_ylabel("Z-score")
ax1.legend(fontsize=8, loc="upper right", ncol=2)
ax1.set_xlim(dates[0], dates[-1])

# Add horizontal annotation for mean-reversion assessment
z_range = float(zscore.max() - zscore.min())
annotation = (
    f"Z-score stats: mean={zscore.mean():.3f}, std={zscore.std():.3f}\n"
    f"Signals: {n_long} long, {n_short} short ({n_total} total)"
)
ax1.text(0.01, 0.97, annotation, transform=ax1.transAxes,
         fontsize=8, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor="lightgrey", alpha=0.85))

# ── Panel 2: Raw spread (log-price units) ──
ax2 = fig.add_subplot(gs[1])

ax2.plot(dates, test_spread.values, color="darkorange", linewidth=0.9)
ax2.axhline(mu, color="black", linestyle="--", linewidth=0.8,
            label=f"2025 mean ({mu:.4f})")
ax2.fill_between(dates, test_spread.values, mu,
                 where=test_spread.values >= mu, alpha=0.18, color="tomato")
ax2.fill_between(dates, test_spread.values, mu,
                 where=test_spread.values <  mu, alpha=0.18, color="seagreen")

ax2.set_title(
    f"Out-of-Sample Spread = log({DEP}) - {beta:.4f} * log({INDEP}) - {alpha:.4f}",
    fontsize=9.5
)
ax2.set_ylabel("Spread (log units)")
ax2.legend(fontsize=8)
ax2.set_xlim(dates[0], dates[-1])

# ── Panel 3: Log prices of AMZN and META (context) ──
ax3 = fig.add_subplot(gs[2])

ax3_right = ax3.twinx()

ax3.plot(log_test.index, log_test[DEP].values,
         color="steelblue", linewidth=1.0, label=f"log({DEP})")
ax3_right.plot(log_test.index, log_test[INDEP].values,
               color="darkorange", linewidth=1.0, label=f"log({INDEP})", linestyle="--")

ax3.set_ylabel(f"log({DEP})", color="steelblue")
ax3_right.set_ylabel(f"log({INDEP})", color="darkorange")
ax3.tick_params(axis="y", labelcolor="steelblue")
ax3_right.tick_params(axis="y", labelcolor="darkorange")

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_right.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
ax3.set_title(f"log({DEP}) and log({INDEP}) price levels -- Test Period", fontsize=9.5)
ax3.set_xlim(dates[0], dates[-1])

chart_path = CHARTS_DIR / "phase4_zscore_2026.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart saved -> {chart_path}")

# ---------------------------------------------------------------------------
# Written conclusion
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print("PHASE 4 -- CONCLUSION")
print(f"{'='*65}")

z_mean_test = float(zscore.mean())
z_std_test  = float(zscore.std())

if abs(z_mean_test) < 0.5 and 0.5 < z_std_test < 2.0:
    mean_rev_verdict = (
        "SUPPORTS mean reversion. The Z-score mean ({:.3f}) is close to zero "
        "and the std ({:.3f}) is within a reasonable range of 1.0, indicating "
        "the AMZN/META cointegrating relationship established on 2018-2025 data "
        "continues to hold in the unseen 2026 period."
    ).format(z_mean_test, z_std_test)
elif abs(z_mean_test) > 1.0:
    mean_rev_verdict = (
        "WEAK evidence of mean reversion. The Z-score mean ({:.3f}) has drifted "
        "significantly from zero, suggesting the spread has shifted from its "
        "historical equilibrium in the 2026 test period."
    ).format(z_mean_test, z_std_test)
else:
    mean_rev_verdict = (
        "MODERATE evidence of mean reversion. The Z-score shows partial "
        "reversion to zero (mean={:.3f}, std={:.3f})."
    ).format(z_mean_test, z_std_test)

conclusion = (
    "\nPHASE 4 -- TESTING MEAN REVERSION ON UNSEEN DATA\n"
    "=================================================\n\n"
    "METHODOLOGY:\n"
    f"  Pair              : {DEP} / {INDEP}\n"
    f"  OLS fitted on     : 2018-01-01 to {TRAIN_END} ({len(train_df)} trading days)\n"
    f"  Hedge ratio (beta): {beta:.6f}\n"
    f"  Intercept (alpha) : {alpha:.6f}\n"
    f"  R-squared         : {r2:.6f}\n"
    f"  Norm window (mu/sigma): {NORM_START} to {TRAIN_END} ({len(norm_spread)} days)\n"
    f"  Fixed mu          : {mu:.6f}\n"
    f"  Fixed sigma       : {sigma:.6f}\n\n"
    "TEST PERIOD RESULTS (Jan-Jul 2026):\n"
    f"  Trading days : {n}\n"
    f"  Z-score mean : {z_mean_test:.4f}  (0.0 = perfect mean reversion anchor)\n"
    f"  Z-score std  : {z_std_test:.4f}   (1.0 = variance unchanged from training)\n"
    f"  Z-score min  : {zscore.min():.4f}\n"
    f"  Z-score max  : {zscore.max():.4f}\n\n"
    "TRADING SIGNALS:\n"
    f"  Long  entries (Z < -{Z_ENTRY}) : {n_long}\n"
    f"  Short entries (Z >  {Z_ENTRY}) : {n_short}\n"
    f"  Total signals                  : {n_total}\n"
    f"  Days in market                 : {(position != 0).sum()} ({(position != 0).sum() / n * 100:.1f}%)\n\n"
    "MEAN REVERSION ASSESSMENT:\n"
    f"  {mean_rev_verdict}\n\n"
    "IMPORTANT NOTE:\n"
    "  The Z-score in Phase 4 uses a fixed mu and sigma anchored to 2025.\n"
    "  This is stricter than Phase 3b (which used a rolling 30-day window).\n"
    "  A fixed anchor is more conservative -- any drift in the spread level\n"
    "  or variance in 2026 will show up as Z-score deviation, even if the\n"
    "  pair remains broadly cointegrated.\n"
)

print(conclusion)

conclusion_path = REPORTS_DIR / "phase4_conclusion.txt"
with open(conclusion_path, "w", encoding="utf-8") as f:
    f.write(conclusion)
print(f"  Conclusion saved -> {conclusion_path}")

print(f"\n{'='*65}")
print("PHASE 4 COMPLETE")
print(f"{'='*65}")
print(f"\nOutputs:")
print(f"  {signals_csv}")
print(f"  {summary_csv}")
print(f"  {chart_path}")
print(f"  {conclusion_path}")
print(f"\nNext step: Phase 5 -- Machine Learning for Predicting Spread (brief pending).")
