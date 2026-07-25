"""
Phase 3b -- Pairs Trading Strategy: AMZN / META
Constructs the spread (log prices), generates z-score signals,
backtests on train/test data, runs walk-forward validation, and
stress-tests over the 2022 bear market.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_RAW, CHARTS_DIR, REPORTS_DIR

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
DEP      = "AMZN"   # dependent variable (from Phase 3a cointegration direction)
INDEP    = "META"   # independent variable
Z_ENTRY  = 2.0      # open position when |z| > Z_ENTRY
Z_EXIT   = 0.0      # close position when z crosses Z_EXIT (mean reversion)
Z_STOP   = 3.0      # stop loss when |z| exceeds Z_STOP in the loss direction
LOOKBACK = 30       # rolling window (trading days) for z-score
COST_PER_LEG = 0.001  # 10 bps transaction cost per leg (applied at entry and exit)
TRAIN_END  = "2021-12-31"
TEST_START = "2022-01-01"

print("=" * 65)
print("PHASE 3B -- PAIRS TRADING STRATEGY: AMZN / META")
print("=" * 65)
print(f"\nStrategy parameters:")
print(f"  Pair            : {DEP} (dep) ~ {INDEP} (indep)")
print(f"  Entry threshold : |z| > {Z_ENTRY}")
print(f"  Exit threshold  : z crosses {Z_EXIT}")
print(f"  Stop loss       : |z| > {Z_STOP} in loss direction")
print(f"  Z-score window  : {LOOKBACK} days")
print(f"  Cost per leg    : {COST_PER_LEG*100:.1f} bps")
print(f"  Train period    : 2018-01-01 to {TRAIN_END}")
print(f"  Test period     : {TEST_START} to 2025-12-31")

# ---------------------------------------------------------------------------
# 1. Load close prices and compute log prices
# ---------------------------------------------------------------------------
prices = {}
for ticker in [DEP, INDEP]:
    path = DATA_RAW / f"{ticker}_raw.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    prices[ticker] = df["Close"]

price_df = pd.DataFrame(prices).dropna()
log_df   = np.log(price_df)

print(f"\nPrices loaded: {len(price_df)} days  "
      f"({price_df.index[0].date()} to {price_df.index[-1].date()})")

# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
train_log = log_df.loc[:TRAIN_END]
test_log  = log_df.loc[TEST_START:]
print(f"\nTrain : {len(train_log)} days  "
      f"({train_log.index[0].date()} to {train_log.index[-1].date()})")
print(f"Test  : {len(test_log)} days  "
      f"({test_log.index[0].date()} to {test_log.index[-1].date()})")

# ---------------------------------------------------------------------------
# 3. Estimate hedge ratio on training data (OLS: log(AMZN) ~ log(META))
# ---------------------------------------------------------------------------
def estimate_hr(y_log, x_log):
    """OLS with constant. Returns (hedge_ratio, intercept, R-squared)."""
    X   = sm.add_constant(x_log)
    mod = sm.OLS(y_log, X).fit()
    return float(mod.params.iloc[1]), float(mod.params.iloc[0]), float(mod.rsquared)

hr, ic, r2 = estimate_hr(train_log[DEP], train_log[INDEP])
print(f"\nTraining OLS  ({train_log.index[0].date()} - {train_log.index[-1].date()}):")
print(f"  Hedge ratio (beta) : {hr:.4f}")
print(f"  Intercept          : {ic:.4f}")
print(f"  R-squared          : {r2:.4f}")
print(f"  Spread = log({DEP}) - {hr:.4f} * log({INDEP}) - {ic:.4f}")

# ---------------------------------------------------------------------------
# 4. Build spread and z-score on full dataset
#    (uses train-period OLS parameters; rolling stats only look back LOOKBACK days)
# ---------------------------------------------------------------------------
def build_spread_zscore(log_prices, hedge_ratio, intercept, lookback):
    spread     = log_prices[DEP] - hedge_ratio * log_prices[INDEP] - intercept
    roll_mean  = spread.rolling(lookback).mean()
    roll_std   = spread.rolling(lookback).std()
    zscore     = (spread - roll_mean) / roll_std
    return spread, zscore

spread_full, zscore_full = build_spread_zscore(log_df, hr, ic, LOOKBACK)

train_spread_mean = float(spread_full.loc[:TRAIN_END].mean())
train_spread_std  = float(spread_full.loc[:TRAIN_END].std())
print(f"\nTraining spread statistics:")
print(f"  Mean : {train_spread_mean:.4f}")
print(f"  Std  : {train_spread_std:.4f}")

# ---------------------------------------------------------------------------
# 5. Backtesting engine
# ---------------------------------------------------------------------------
def backtest(zscore_series, spread_series, z_entry, z_exit, z_stop, cost_per_leg):
    """
    Simulate pairs trading on z-score signal.

    Position rules:
      +1 (long spread)  : entered when z < -z_entry  (buy DEP, sell INDEP)
      -1 (short spread) : entered when z > +z_entry  (sell DEP, buy INDEP)
       0 (flat)

    Exit rules:
      Long  (pos=+1): close when z >= z_exit  OR  z <= -z_stop (stop loss)
      Short (pos=-1): close when z <= z_exit  OR  z >= +z_stop (stop loss)

    P&L: position * daily_delta_spread - transaction_costs
    Transaction cost charged on each entry AND exit (2 legs each time).

    Returns DataFrame with: zscore, spread, position, daily_pnl,
                             trade_cost, cum_pnl.
    """
    z_arr  = zscore_series.values
    sp_arr = spread_series.values
    n      = len(z_arr)

    position  = np.zeros(n)
    daily_pnl = np.zeros(n)
    trade_cost= np.zeros(n)

    pos = 0

    for i in range(1, n):
        z_prev = z_arr[i - 1]   # signal observed at yesterday's close
        z_now  = z_arr[i]       # today's z-score (used only for P&L, not decisions)

        if np.isnan(z_prev) or np.isnan(z_now):
            continue

        # All entry/exit decisions use z_prev — signal observed at day i-1 close,
        # trade executed at day i open. This prevents earning the same-day spread
        # move that triggered the signal.

        # --- Check exit / stop ---
        if pos == 1:
            if z_prev >= z_exit or z_prev <= -z_stop:
                trade_cost[i] += 2 * cost_per_leg
                pos = 0
        elif pos == -1:
            if z_prev <= z_exit or z_prev >= z_stop:
                trade_cost[i] += 2 * cost_per_leg
                pos = 0

        # --- Check entry (only if flat) ---
        if pos == 0:
            if z_prev < -z_entry:
                trade_cost[i] += 2 * cost_per_leg
                pos = 1
            elif z_prev > z_entry:
                trade_cost[i] += 2 * cost_per_leg
                pos = -1

        position[i]  = pos
        delta_spread = sp_arr[i] - sp_arr[i - 1]
        daily_pnl[i] = pos * delta_spread - trade_cost[i]

    return pd.DataFrame({
        "zscore":     z_arr,
        "spread":     sp_arr,
        "position":   position,
        "daily_pnl":  daily_pnl,
        "trade_cost": trade_cost,
        "cum_pnl":    np.cumsum(daily_pnl),
    }, index=zscore_series.index)

# ---------------------------------------------------------------------------
# 6. Run out-of-sample backtest on test period
# ---------------------------------------------------------------------------
bt_test = backtest(
    zscore_full.loc[TEST_START:],
    spread_full.loc[TEST_START:],
    Z_ENTRY, Z_EXIT, Z_STOP, COST_PER_LEG
)

# ---------------------------------------------------------------------------
# 7. Performance metrics
# ---------------------------------------------------------------------------
def compute_metrics(bt_df, label=""):
    pnl = bt_df["daily_pnl"]
    cum = bt_df["cum_pnl"]
    n   = len(pnl)

    total_pnl    = float(cum.iloc[-1])
    annual_pnl   = total_pnl / (n / 252)
    sharpe       = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0

    running_max  = cum.cummax()
    drawdown     = cum - running_max
    max_dd       = float(drawdown.min())

    in_pos       = bt_df["position"] != 0
    pos_pnl      = bt_df.loc[in_pos, "daily_pnl"]
    win_rate     = float((pos_pnl > 0).sum() / len(pos_pnl)) if len(pos_pnl) > 0 else 0.0

    # Count trades as position changes
    pos_changes  = bt_df["position"].diff().abs()
    n_trades     = int((pos_changes > 0).sum())
    pct_in_mkt   = float(in_pos.sum() / n * 100)

    return {
        "Period":           label,
        "Days":             n,
        "Total_PnL":        round(total_pnl, 6),
        "Annual_PnL":       round(annual_pnl, 6),
        "Sharpe_ratio":     round(sharpe, 4),
        "Max_drawdown":     round(max_dd, 6),
        "Win_rate_pct":     round(win_rate * 100, 2),
        "Num_position_chg": n_trades,
        "Pct_in_market":    round(pct_in_mkt, 2),
    }

test_m = compute_metrics(bt_test, label="Test 2022-2025")

print(f"\n{'='*65}")
print("OUT-OF-SAMPLE TEST PERIOD PERFORMANCE (2022-2025)")
print(f"{'='*65}")
for k, v in test_m.items():
    print(f"  {k:<22}: {v}")

# ---------------------------------------------------------------------------
# 8. Walk-forward validation — re-estimate hedge ratio at start of each year
#    Training expands: year Y uses all data up to end of Y-1
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print("WALK-FORWARD VALIDATION (annual, expanding window)")
print(f"{'='*65}")
print(f"{'Year':<6} {'HR':>7} {'Intercept':>10} {'R2':>6}  "
      f"{'Sharpe':>8}  {'Total PnL':>10}  {'WinRate':>9}  {'Pos chg':>8}")
print("-" * 75)

wf_records  = []
wf_segments = {}   # year -> bt DataFrame (for plotting)

for year in range(2019, 2026):
    train_wf = log_df.loc[:f"{year-1}-12-31"]
    test_wf  = log_df.loc[f"{year}-01-01":f"{year}-12-31"]
    if len(test_wf) == 0:
        continue

    hr_wf, ic_wf, r2_wf = estimate_hr(train_wf[DEP], train_wf[INDEP])

    # Build spread on full history up to end of test year (no look-ahead:
    # HR estimated on prior years only; rolling z-score looks back LOOKBACK days)
    spread_wf = log_df[DEP] - hr_wf * log_df[INDEP] - ic_wf
    spread_to = spread_wf.loc[:f"{year}-12-31"]
    z_to = (spread_to - spread_to.rolling(LOOKBACK).mean()) / spread_to.rolling(LOOKBACK).std()

    z_yr  = z_to.loc[f"{year}-01-01":f"{year}-12-31"]
    sp_yr = spread_wf.loc[f"{year}-01-01":f"{year}-12-31"]

    bt_yr = backtest(z_yr, sp_yr, Z_ENTRY, Z_EXIT, Z_STOP, COST_PER_LEG)
    m = compute_metrics(bt_yr, label=str(year))
    m["HR"] = round(hr_wf, 4)
    m["R2"] = round(r2_wf, 4)

    wf_records.append(m)
    wf_segments[year] = bt_yr

    print(f"  {year:<4}  {hr_wf:>7.4f}  {ic_wf:>10.4f}  {r2_wf:>6.4f}  "
          f"{m['Sharpe_ratio']:>8.4f}  {m['Total_PnL']:>+10.6f}  "
          f"{m['Win_rate_pct']:>8.2f}%  {m['Num_position_chg']:>8}")

wf_df = pd.DataFrame(wf_records)

# ---------------------------------------------------------------------------
# 9. Stress test: 2022 bear market
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print("STRESS TEST -- 2022 BEAR MARKET")
print(f"{'='*65}")
m_2022 = compute_metrics(wf_segments[2022], label="2022 Bear Market")
for k, v in m_2022.items():
    print(f"  {k:<22}: {v}")
note = ("  Note: 2022 saw META -65%, AMZN -50%. Strategy should benefit if\n"
        "  the spread mean-reverted despite the macro drawdown.")
print(note)

# ---------------------------------------------------------------------------
# 10. Save results CSV
# ---------------------------------------------------------------------------
all_records = [test_m] + wf_records
results_df  = pd.DataFrame(all_records)
csv_out = REPORTS_DIR / "strategy_results.csv"
results_df.to_csv(csv_out, index=False)
print(f"\nResults saved -> {csv_out}")

# ---------------------------------------------------------------------------
# 11. Charts
# ---------------------------------------------------------------------------

# --- Chart 1: Z-score signals + spread level + equity curve (test period) ---
fig = plt.figure(figsize=(16, 13))
gs  = gridspec.GridSpec(3, 1, hspace=0.45, figure=fig)

# Panel 1 — Z-score with bands and position shading
ax1 = fig.add_subplot(gs[0])
z_test = zscore_full.loc[TEST_START:]
ax1.plot(z_test.index, z_test.values, color="steelblue", linewidth=0.85, label="Z-score")
ax1.axhline( Z_ENTRY, color="red",     linestyle="--", linewidth=1.0, label=f"+{Z_ENTRY} entry (short)")
ax1.axhline(-Z_ENTRY, color="green",   linestyle="--", linewidth=1.0, label=f"-{Z_ENTRY} entry (long)")
ax1.axhline( Z_EXIT,  color="black",   linestyle="-",  linewidth=0.5, label="0 exit")
ax1.axhline( Z_STOP,  color="darkred", linestyle=":",  linewidth=0.9, label=f"+{Z_STOP} stop loss")
ax1.axhline(-Z_STOP,  color="darkred", linestyle=":",  linewidth=0.9)

pos_arr = bt_test["position"].values
dates   = bt_test.index
for i in range(1, len(pos_arr)):
    if pos_arr[i] == 1:
        ax1.axvspan(dates[i-1], dates[i], alpha=0.13, color="green",  linewidth=0)
    elif pos_arr[i] == -1:
        ax1.axvspan(dates[i-1], dates[i], alpha=0.13, color="tomato", linewidth=0)

ax1.set_title(f"Spread Z-score — Test Period {TEST_START[:4]}-2025  "
              "(green shading = long spread, red = short spread)", fontsize=10)
ax1.set_ylabel("Z-score")
ax1.legend(fontsize=7.5, loc="upper right", ncol=3)
ax1.set_xlim(bt_test.index[0], bt_test.index[-1])

# Panel 2 — Spread level
ax2 = fig.add_subplot(gs[1])
sp_test = spread_full.loc[TEST_START:]
ax2.plot(sp_test.index, sp_test.values, color="darkorange", linewidth=0.85)
ax2.axhline(train_spread_mean, color="black", linestyle="--", linewidth=0.8,
            label=f"Training mean ({train_spread_mean:.4f})")
ax2.set_title(f"Spread = log({DEP}) − {hr:.4f}·log({INDEP}) − {ic:.4f}", fontsize=10)
ax2.set_ylabel("Spread (log-price units)")
ax2.legend(fontsize=8)
ax2.set_xlim(bt_test.index[0], bt_test.index[-1])

# Panel 3 — Cumulative P&L equity curve
ax3 = fig.add_subplot(gs[2])
ax3.plot(bt_test.index, bt_test["cum_pnl"].values, color="steelblue", linewidth=1.2)
ax3.fill_between(bt_test.index, bt_test["cum_pnl"].values, 0,
                 where=bt_test["cum_pnl"].values >= 0, alpha=0.25, color="green")
ax3.fill_between(bt_test.index, bt_test["cum_pnl"].values, 0,
                 where=bt_test["cum_pnl"].values <  0, alpha=0.25, color="tomato")
ax3.axhline(0, color="black", linewidth=0.6)
metrics_str = (f"Sharpe={test_m['Sharpe_ratio']:.2f}  "
               f"MaxDD={test_m['Max_drawdown']:.4f}  "
               f"WinRate={test_m['Win_rate_pct']:.1f}%  "
               f"Total PnL={test_m['Total_PnL']:+.4f}")
ax3.set_title(f"Cumulative P&L — Test Period  |  {metrics_str}", fontsize=9.5)
ax3.set_ylabel("Cumulative log-return P&L")
ax3.set_xlabel("Date")
ax3.set_xlim(bt_test.index[0], bt_test.index[-1])

chart1 = CHARTS_DIR / "strategy_zscore_equity.png"
plt.savefig(chart1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nChart saved -> {chart1}")

# --- Chart 2: Walk-forward — Sharpe + P&L per year ---
fig2, (ax_s, ax_p) = plt.subplots(1, 2, figsize=(14, 5))
yr_labels = [r["Period"] for r in wf_records]
sharpes   = [r["Sharpe_ratio"] for r in wf_records]
pnls      = [r["Total_PnL"]    for r in wf_records]
c_sharpe  = ["steelblue" if s >= 0 else "tomato" for s in sharpes]
c_pnl     = ["steelblue" if p >= 0 else "tomato" for p in pnls]

ax_s.bar(yr_labels, sharpes, color=c_sharpe, edgecolor="white", linewidth=0.5)
ax_s.axhline(0, color="black", linewidth=0.8)
ax_s.set_title("Walk-Forward Annual Sharpe Ratios", fontsize=11)
ax_s.set_ylabel("Annualised Sharpe ratio")
ax_s.set_xlabel("Year (expanding training window)")

ax_p.bar(yr_labels, pnls, color=c_pnl, edgecolor="white", linewidth=0.5)
ax_p.axhline(0, color="black", linewidth=0.8)
ax_p.set_title("Walk-Forward Annual P&L (log-return units)", fontsize=11)
ax_p.set_ylabel("Total P&L")
ax_p.set_xlabel("Year (expanding training window)")

plt.tight_layout()
chart2 = CHARTS_DIR / "strategy_walkforward.png"
plt.savefig(chart2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved -> {chart2}")

# --- Chart 3: Full period z-score (2018-2025) — context view ---
fig3, ax_full = plt.subplots(figsize=(16, 4))
ax_full.plot(zscore_full.index, zscore_full.values,
             color="steelblue", linewidth=0.7, alpha=0.9)
ax_full.axhline( Z_ENTRY,  color="red",     linestyle="--", linewidth=0.9)
ax_full.axhline(-Z_ENTRY,  color="green",   linestyle="--", linewidth=0.9)
ax_full.axhline( 0,        color="black",   linestyle="-",  linewidth=0.5)
ax_full.axhline( Z_STOP,   color="darkred", linestyle=":",  linewidth=0.8)
ax_full.axhline(-Z_STOP,   color="darkred", linestyle=":",  linewidth=0.8)
ax_full.axvspan(pd.Timestamp(TEST_START), zscore_full.index[-1],
                alpha=0.07, color="orange", label=f"Test period ({TEST_START[:4]}-2025)")
ax_full.axvspan(zscore_full.index[0], pd.Timestamp(TRAIN_END),
                alpha=0.07, color="skyblue", label=f"Train period (2018-{TRAIN_END[:4]})")
ax_full.set_title(f"AMZN/META Spread Z-score — Full Period 2018-2025 (Train + Test)", fontsize=11)
ax_full.set_ylabel("Z-score")
ax_full.legend(fontsize=9)
plt.tight_layout()
chart3 = CHARTS_DIR / "strategy_zscore_full.png"
plt.savefig(chart3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved -> {chart3}")

# --- Chart 4: Walk-forward equity curves stacked by year ---
fig4, axes4 = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
axes4_flat  = axes4.flatten()
for idx, (year, bt_yr) in enumerate(wf_segments.items()):
    ax = axes4_flat[idx]
    cum = bt_yr["cum_pnl"].values
    ax.plot(bt_yr.index, cum, color="steelblue", linewidth=1.0)
    ax.fill_between(bt_yr.index, cum, 0,
                    where=cum >= 0, alpha=0.2, color="green")
    ax.fill_between(bt_yr.index, cum, 0,
                    where=cum <  0, alpha=0.2, color="tomato")
    ax.axhline(0, color="black", linewidth=0.6)
    m_yr = next(r for r in wf_records if r["Period"] == str(year))
    ax.set_title(f"{year}  Sharpe={m_yr['Sharpe_ratio']:.2f}  "
                 f"PnL={m_yr['Total_PnL']:+.4f}", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=6.5, rotation=30)
    ax.tick_params(axis="y", labelsize=7)

for j in range(len(wf_segments), len(axes4_flat)):
    axes4_flat[j].set_visible(False)

fig4.suptitle("Walk-Forward Annual Equity Curves — AMZN/META Pairs Strategy", fontsize=12, y=1.01)
plt.tight_layout()
chart4 = CHARTS_DIR / "strategy_annual_equity.png"
plt.savefig(chart4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved -> {chart4}")

# ---------------------------------------------------------------------------
# 12. Summary to console
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print("WALK-FORWARD SUMMARY")
print(f"{'='*65}")
positive_yrs = sum(1 for r in wf_records if r["Total_PnL"] > 0)
total_wf_pnl = sum(r["Total_PnL"] for r in wf_records)
avg_sharpe   = float(wf_df["Sharpe_ratio"].mean())
print(f"  Profitable years (of {len(wf_records)}) : {positive_yrs}")
print(f"  Total walk-forward PnL       : {total_wf_pnl:+.6f}")
print(f"  Average annual Sharpe        : {avg_sharpe:.4f}")
print(f"  Best year  : {wf_df.loc[wf_df['Total_PnL'].idxmax(), 'Period']} "
      f"(PnL={wf_df['Total_PnL'].max():+.6f})")
print(f"  Worst year : {wf_df.loc[wf_df['Total_PnL'].idxmin(), 'Period']} "
      f"(PnL={wf_df['Total_PnL'].min():+.6f})")

print(f"\n{'='*65}")
print("PHASE 3B -- PAIRS TRADING STRATEGY COMPLETE")
print(f"{'='*65}")
print(f"\nOutputs:")
print(f"  {csv_out}")
print(f"  {chart1}")
print(f"  {chart2}")
print(f"  {chart3}")
print(f"  {chart4}")
print(f"\nNext step: Phase 4 -- Sentiment analysis overlay.")
