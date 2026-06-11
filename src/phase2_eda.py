"""
Phase 2 -- Exploratory Data Analysis
Loads master_data.csv and produces 8 analyses/charts saved to outputs/.
"""

import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ALL_TICKERS, DATA_RAW, DATA_PROCESSED, CHARTS_DIR, REPORTS_DIR

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="darkgrid", palette="tab10")
COLORS = sns.color_palette("tab10", len(ALL_TICKERS))
TICKER_COLOR = dict(zip(ALL_TICKERS, COLORS))

print("=" * 60)
print("PHASE 2 -- EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------------------------
# List all 15 pairs upfront
# ---------------------------------------------------------------------------
ALL_PAIRS = list(combinations(ALL_TICKERS, 2))
print(f"\nAll possible pairs ({len(ALL_PAIRS)} total):")
for i, (a, b) in enumerate(ALL_PAIRS, 1):
    print(f"  {i:2d}. {a} / {b}")

# ---------------------------------------------------------------------------
# Load master data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PROCESSED / "master_data.csv", index_col="Date", parse_dates=True)
ret_cols = [f"{t}_return" for t in ALL_TICKERS]
returns  = df[ret_cols].copy()
returns.columns = ALL_TICKERS          # short names for plotting

print(f"\nLoaded master_data.csv  shape={df.shape}  "
      f"range={df.index[0].date()} to {df.index[-1].date()}")

# ---------------------------------------------------------------------------
# 1 -- Summary statistics
# ---------------------------------------------------------------------------
print("\n[1/10] Summary statistics ...")

summary = pd.DataFrame({
    "Mean (%)":     returns.mean(),
    "Std (%)":      returns.std(),
    "Min (%)":      returns.min(),
    "Max (%)":      returns.max(),
    "Skewness":     returns.skew(),
    "Kurtosis":     returns.kurtosis(),
}).round(4)

summary.index.name = "Ticker"
summary.to_csv(REPORTS_DIR / "summary_stats.csv")
print(summary.to_string())
print(f"  Saved -> outputs/reports/summary_stats.csv")

# ---------------------------------------------------------------------------
# 2 -- Histogram grid with KDE overlay
# ---------------------------------------------------------------------------
print("\n[2/10] Histogram grid ...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Daily Return Distributions (2018-2025)", fontsize=14, fontweight="bold")

for ax, ticker in zip(axes.flat, ALL_TICKERS):
    data = returns[ticker].dropna()
    ax.hist(data, bins=80, density=True, color=TICKER_COLOR[ticker],
            alpha=0.45, edgecolor="none", label="Freq")
    kde_x = np.linspace(data.min(), data.max(), 300)
    kde   = stats.gaussian_kde(data)
    ax.plot(kde_x, kde(kde_x), color=TICKER_COLOR[ticker], lw=2, label="KDE")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title(ticker, fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

plt.tight_layout()
out = CHARTS_DIR / "hist_returns.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved -> outputs/charts/hist_returns.png")

# ---------------------------------------------------------------------------
# 3 -- Correlation heatmap
# ---------------------------------------------------------------------------
print("\n[3/10] Correlation heatmap ...")

corr = returns.corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle only
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
    vmin=0.4, vmax=1.0, linewidths=0.5,
    ax=ax, square=True, cbar_kws={"shrink": 0.8}
)
ax.set_title("Pearson Correlation -- Daily Returns (2018-2025)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = CHARTS_DIR / "correlation_heatmap.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved -> outputs/charts/correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 4 -- Ranked pairwise correlation table (all 15 pairs)
# ---------------------------------------------------------------------------
print("\n[4/10] Ranked pairwise correlation table ...")

pair_corr_records = []
for t1, t2 in ALL_PAIRS:
    r = returns[t1].corr(returns[t2])
    pair_corr_records.append({"Pair": f"{t1} / {t2}", "Stock A": t1, "Stock B": t2,
                               "Correlation": round(r, 4)})

pair_corr_df = (pd.DataFrame(pair_corr_records)
                  .sort_values("Correlation", ascending=False)
                  .reset_index(drop=True))
pair_corr_df.index += 1
pair_corr_df.index.name = "Rank"
pair_corr_df.to_csv(REPORTS_DIR / "pair_correlations.csv")
print(pair_corr_df[["Pair", "Correlation"]].to_string())
print(f"  Saved -> outputs/reports/pair_correlations.csv")

# ---------------------------------------------------------------------------
# 5 -- Cumulative returns (base-indexed to 100)
# ---------------------------------------------------------------------------
print("\n[5/10] Cumulative returns ...")

cum = (1 + returns / 100).cumprod() * 100

fig, ax = plt.subplots(figsize=(14, 6))
for ticker in ALL_TICKERS:
    ax.plot(cum.index, cum[ticker], label=ticker,
            color=TICKER_COLOR[ticker], lw=1.6)

ax.axhline(100, color="black", lw=0.8, ls="--", alpha=0.5)
ax.set_title("Cumulative Returns -- Base 100 at 2018-01-03", fontsize=13, fontweight="bold")
ax.set_ylabel("Index (Start = 100)")
ax.set_xlabel("Date")
ax.legend(loc="upper left", ncol=2)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
plt.tight_layout()
out = CHARTS_DIR / "cumulative_returns.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved -> outputs/charts/cumulative_returns.png")

# ---------------------------------------------------------------------------
# 5 -- Rolling 30-day annualised volatility
# ---------------------------------------------------------------------------
print("\n[6/10] Rolling 30-day volatility ...")

roll_vol = returns.rolling(30).std() * np.sqrt(252)

fig, ax = plt.subplots(figsize=(14, 6))
for ticker in ALL_TICKERS:
    ax.plot(roll_vol.index, roll_vol[ticker], label=ticker,
            color=TICKER_COLOR[ticker], lw=1.4, alpha=0.85)

ax.set_title("Rolling 30-Day Annualised Volatility (2018-2025)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Annualised Vol (%)")
ax.set_xlabel("Date")
ax.legend(loc="upper right", ncol=2)
plt.tight_layout()
out = CHARTS_DIR / "rolling_volatility.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved -> outputs/charts/rolling_volatility.png")

# ---------------------------------------------------------------------------
# 6 -- Boxplot by year
# ---------------------------------------------------------------------------
print("\n[7/10] Boxplot by year ...")

years = sorted(df["Year"].unique())
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Daily Return Distribution by Year", fontsize=14, fontweight="bold")

for ax, ticker in zip(axes.flat, ALL_TICKERS):
    data_by_year = [returns[ticker][df["Year"] == y].dropna().values for y in years]
    bp = ax.boxplot(data_by_year, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(TICKER_COLOR[ticker])
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(years) + 1))
    ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=8)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(ticker, fontweight="bold")
    ax.set_ylabel("Daily Return (%)")

plt.tight_layout()
out = CHARTS_DIR / "boxplot_by_year.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved -> outputs/charts/boxplot_by_year.png")

# ---------------------------------------------------------------------------
# 7 -- Pairwise scatter plots (price levels) for all 15 pairs
# ---------------------------------------------------------------------------
print("\n[8/10] Pairwise scatter plots (price levels) ...")

price_frames = {}
for ticker in ALL_TICKERS:
    raw = pd.read_csv(DATA_RAW / f"{ticker}_raw.csv", index_col="Date",
                      parse_dates=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    price_frames[ticker] = raw["Close"].rename(ticker)

prices = pd.concat(price_frames.values(), axis=1).dropna()

ncols = 5
nrows = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12))
fig.suptitle("Pairwise Price Level Scatter Plots (2018-2025)",
             fontsize=14, fontweight="bold")

for ax, (t1, t2) in zip(axes.flat, ALL_PAIRS):
    ax.scatter(prices[t1], prices[t2], s=2, alpha=0.35,
               color=TICKER_COLOR[t1])
    m, b, r, _, _ = stats.linregress(prices[t1], prices[t2])
    x_line = np.linspace(prices[t1].min(), prices[t1].max(), 200)
    ax.plot(x_line, m * x_line + b, color="crimson", lw=1.2,
            label=f"R={r:.2f}")
    ax.set_xlabel(t1, fontsize=9)
    ax.set_ylabel(t2, fontsize=9)
    ax.set_title(f"{t1} vs {t2}", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)

# hide unused axes (15 pairs, 15 subplots -- exact fit, but keep guard)
for ax in axes.flat[len(ALL_PAIRS):]:
    ax.set_visible(False)

plt.tight_layout()
out = CHARTS_DIR / "pairwise_price_scatter.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved -> outputs/charts/pairwise_price_scatter.png")

# ---------------------------------------------------------------------------
# 9 -- Pairwise return scatter plots for all 15 pairs
# ---------------------------------------------------------------------------
print("\n[9/10] Pairwise return scatter plots ...")

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle("Pairwise Daily Return Scatter Plots (2018-2025)",
             fontsize=14, fontweight="bold")

for ax, (t1, t2) in zip(axes.flat, ALL_PAIRS):
    x = returns[t1].dropna()
    y = returns[t2].dropna()
    common = x.align(y, join="inner")
    x_c, y_c = common[0], common[1]
    ax.scatter(x_c, y_c, s=2, alpha=0.3, color=TICKER_COLOR[t1])
    m, b, r, _, _ = stats.linregress(x_c, y_c)
    x_line = np.linspace(x_c.min(), x_c.max(), 200)
    ax.plot(x_line, m * x_line + b, color="crimson", lw=1.2,
            label=f"R={r:.2f}")
    ax.set_xlabel(f"{t1} Return (%)", fontsize=8)
    ax.set_ylabel(f"{t2} Return (%)", fontsize=8)
    ax.set_title(f"{t1} vs {t2}", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)

for ax in axes.flat[len(ALL_PAIRS):]:
    ax.set_visible(False)

plt.tight_layout()
out = CHARTS_DIR / "pairwise_return_scatter.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved -> outputs/charts/pairwise_return_scatter.png")

# ---------------------------------------------------------------------------
# 10 -- Beta analysis (each stock vs S&P 500)
# ---------------------------------------------------------------------------
print("\n[10/10] Beta analysis ...")

sp500 = df["sp500_return"].dropna()
beta_records = []

for ticker in ALL_TICKERS:
    common = returns[ticker].align(sp500, join="inner")
    y, x   = common[0].dropna(), common[1].loc[common[0].dropna().index]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    beta_records.append({
        "Ticker":    ticker,
        "Beta":      round(slope, 4),
        "Alpha (%)": round(intercept, 4),
        "R-squared": round(r_value ** 2, 4),
        "P-value":   round(p_value, 6),
        "Std Error": round(std_err, 4),
    })
    print(f"  {ticker:6s}  beta={slope:.3f}  alpha={intercept:.4f}  "
          f"R2={r_value**2:.3f}  p={p_value:.4f}")

beta_df = pd.DataFrame(beta_records).set_index("Ticker")
beta_df.to_csv(REPORTS_DIR / "beta_analysis.csv")
print(f"  Saved -> outputs/reports/beta_analysis.csv")

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(beta_df.index, beta_df["Beta"],
              color=[TICKER_COLOR[t] for t in beta_df.index],
              edgecolor="black", linewidth=0.6)
ax.axhline(1.0, color="crimson", lw=1.2, ls="--", label="Beta = 1 (market)")
ax.axhline(0.0, color="black",   lw=0.8, ls="--", alpha=0.4)
for bar, val in zip(bars, beta_df["Beta"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02, f"{val:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Beta vs S&P 500 (2018-2025)", fontsize=13, fontweight="bold")
ax.set_ylabel("Beta")
ax.set_xlabel("Ticker")
ax.legend()
plt.tight_layout()
out = CHARTS_DIR / "beta_analysis.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved -> outputs/charts/beta_analysis.png")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)
print("\nCharts saved to  outputs/charts/")
print("Reports saved to outputs/reports/")
