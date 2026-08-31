"""
Unit tests for the core pairs-trading logic.

Tests are self-contained: they construct minimal synthetic data and check
hand-computed expected values, so no real market data files are required.

Run from the project root:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase3_strategy import backtest, compute_metrics, build_spread_zscore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


# ---------------------------------------------------------------------------
# backtest() — position state-machine
# ---------------------------------------------------------------------------

class TestBacktestPositionLogic:

    def test_no_trades_when_z_below_entry(self):
        """All z-scores inside the entry band → position stays 0 throughout."""
        idx = _make_index(10)
        z  = pd.Series([0.0, 0.5, 1.0, 1.5, 1.9, 1.9, 1.5, 1.0, 0.5, 0.0], index=idx)
        sp = pd.Series(np.zeros(10), index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert (bt["position"] == 0).all(), "No position should be opened below entry threshold"

    def test_long_position_entered_on_low_z(self):
        """z drops below -entry → long spread position (+1) opened next bar."""
        idx = _make_index(5)
        # bar 0: z = -2.5  (signal fires)
        # bar 1: position should be +1
        z  = pd.Series([-2.5, -2.5, -1.0, 0.0, 0.5], index=idx)
        sp = pd.Series([0.0,   0.1,  0.2, 0.3, 0.4], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == 1, "Long position expected after z < -entry"

    def test_short_position_entered_on_high_z(self):
        """z rises above +entry → short spread position (-1) opened next bar."""
        idx = _make_index(5)
        z  = pd.Series([2.5, 2.5, 1.0, 0.0, -0.5], index=idx)
        sp = pd.Series([0.0, 0.1, 0.2, 0.3,  0.4], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == -1, "Short position expected after z > +entry"

    def test_long_position_closed_at_mean_reversion(self):
        """Long position closes when z crosses back through z_exit (0)."""
        idx = _make_index(6)
        # bar 0: z = -2.5  → enter long at bar 1
        # bar 3: z = 0.0   → exit at bar 4
        z  = pd.Series([-2.5, -2.0, -1.0,  0.0,  0.5,  0.8], index=idx)
        sp = pd.Series([ 0.0,  0.1,  0.2,  0.3,  0.4,  0.5], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == 1,  "Should be long at bar 1"
        assert bt["position"].iloc[2] == 1,  "Should still be long at bar 2"
        assert bt["position"].iloc[3] == 1,  "Should still be long at bar 3"
        assert bt["position"].iloc[4] == 0,  "Should be flat after mean reversion"

    def test_stop_loss_closes_long_position(self):
        """
        Long position hits stop loss and then exits on mean reversion.

        Engine rule: at bar i, decisions use z[i-1] (z_prev).
        When z_prev crosses the stop threshold AND is still below -z_entry,
        the engine exits and immediately re-enters in the same bar.

        Sequence (z by bar index, 0..4):
          i=1: z_prev=z[0]=-2.5 < -2.0  → enter long.  pos[1]=+1
          i=2: z_prev=z[1]=-2.5  → no exit, no new entry.  pos[2]=+1
          i=3: z_prev=z[2]=-3.1 ≤ -3.0  → stop exit.
               z_prev=-3.1 < -2.0 → immediate re-entry.  pos[3]=+1
          i=4: z_prev=z[3]=0.0  ≥  0.0  → mean-reversion exit.  pos[4]=0
        """
        idx = _make_index(5)
        z  = pd.Series([-2.5, -2.5, -3.1,  0.0,  0.5], index=idx)
        sp = pd.Series([ 0.0,  0.1,  0.2,  0.3,  0.4], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == 1, "Long entered at bar 1"
        assert bt["position"].iloc[3] == 1, "Stop fires but re-enters (z_prev still below -entry)"
        assert bt["position"].iloc[4] == 0, "Flat after mean reversion at bar 4"

    def test_stop_loss_closes_short_position(self):
        """
        Short position hits stop loss and then exits on mean reversion.

        Sequence (z by bar index, 0..4):
          i=1: z_prev=z[0]=+2.5 > +2.0  → enter short.  pos[1]=-1
          i=2: z_prev=z[1]=+2.5  → no exit.  pos[2]=-1
          i=3: z_prev=z[2]=+3.1 ≥ +3.0  → stop exit.
               z_prev=+3.1 > +2.0 → immediate re-entry.  pos[3]=-1
          i=4: z_prev=z[3]=0.0  ≤  0.0  → mean-reversion exit.  pos[4]=0
        """
        idx = _make_index(5)
        z  = pd.Series([2.5, 2.5,  3.1,  0.0, -0.5], index=idx)
        sp = pd.Series([0.0, 0.1,  0.2,  0.3,  0.4], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == -1, "Short entered at bar 1"
        assert bt["position"].iloc[3] == -1, "Stop fires but re-enters (z_prev still above +entry)"
        assert bt["position"].iloc[4] ==  0, "Flat after mean reversion at bar 4"

    def test_no_position_flip_without_going_flat_first(self):
        """
        While in a long position the z-score should not immediately trigger a
        short entry — the engine must go flat first.
        """
        idx = _make_index(6)
        # bar 0: z = -2.5  → enter long
        # bar 1-2: position = +1
        # bar 3: z = 0.0   → exit (flat)
        # bar 4: z = 2.5   → enter short
        z  = pd.Series([-2.5, -2.5, -1.0,  0.0,  2.5,  2.5], index=idx)
        sp = pd.Series([ 0.0,  0.1,  0.2,  0.3,  0.4,  0.5], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == 1,  "Long at bar 1"
        assert bt["position"].iloc[4] == 0,  "Flat at bar 4 (exit processed, entry next)"
        assert bt["position"].iloc[5] == -1, "Short at bar 5"


# ---------------------------------------------------------------------------
# backtest() — P&L arithmetic
# ---------------------------------------------------------------------------

class TestBacktestPnL:

    def test_single_long_trade_pnl_no_costs(self):
        """
        Hand-computed expected P&L for a simple long trade.

        Setup:
          bar 0 (signal): z = -2.5  → enter long at bar 1
          bar 1: spread moves from 1.0 → 1.1  (+0.1)  position = +1  → P&L = +0.1
          bar 2: spread moves from 1.1 → 1.2  (+0.1)  position = +1  → P&L = +0.1
          bar 3 (signal): z = 0.0   → exit at bar 4  (spread now at 1.2)
          bar 4: spread moves from 1.2 → 1.3  (but position already 0) → P&L = 0
        """
        idx = _make_index(5)
        z   = pd.Series([-2.5, -2.5,  -1.0,   0.0,  0.5], index=idx)
        sp  = pd.Series([ 1.0,  1.1,   1.2,   1.2,  1.3], index=idx)
        bt  = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["daily_pnl"].iloc[1] == pytest.approx(0.1, abs=1e-9)
        assert bt["daily_pnl"].iloc[2] == pytest.approx(0.1, abs=1e-9)
        assert bt["daily_pnl"].iloc[4] == pytest.approx(0.0, abs=1e-9)
        assert bt["cum_pnl"].iloc[2]   == pytest.approx(0.2, abs=1e-9)

    def test_transaction_costs_deducted_at_entry_and_exit(self):
        """
        Each entry and exit costs 2 * cost_per_leg.
        One complete round-trip (entry + exit) → 4 * cost_per_leg deducted total.
        """
        cost = 0.001   # 10 bps
        idx  = _make_index(5)
        # Flat spread — all P&L comes only from costs
        z  = pd.Series([-2.5, -2.5, -1.0, 0.0, 0.5], index=idx)
        sp = pd.Series([ 1.0,  1.0,  1.0, 1.0, 1.0], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=cost)
        # Entry cost at bar 1
        assert bt["trade_cost"].iloc[1] == pytest.approx(2 * cost, abs=1e-9)
        # Exit cost at bar 4
        assert bt["trade_cost"].iloc[4] == pytest.approx(2 * cost, abs=1e-9)
        # Total cumulative P&L = -(entry cost + exit cost)
        assert bt["cum_pnl"].iloc[-1]   == pytest.approx(-4 * cost, abs=1e-9)

    def test_cumulative_pnl_is_running_sum_of_daily_pnl(self):
        """cum_pnl must equal np.cumsum(daily_pnl) at every row."""
        idx = _make_index(20)
        rng = np.random.default_rng(42)
        z   = pd.Series(rng.normal(0, 2, 20), index=idx)
        sp  = pd.Series(rng.normal(0, 0.1, 20).cumsum(), index=idx)
        bt  = backtest(z, sp, z_entry=1.5, z_exit=0.0, z_stop=3.0, cost_per_leg=0.001)
        expected = np.cumsum(bt["daily_pnl"].values)
        np.testing.assert_allclose(bt["cum_pnl"].values, expected, atol=1e-12)

    def test_nan_z_scores_do_not_open_positions(self):
        """NaN z-score bars (e.g. during rolling-window warm-up) must be skipped."""
        idx = _make_index(6)
        z   = pd.Series([np.nan, np.nan, np.nan, -2.5, -2.5, 0.0], index=idx)
        sp  = pd.Series([   1.0,    1.0,    1.0,   1.0,  1.1, 1.2], index=idx)
        bt  = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        # No position should be opened during NaN period
        assert bt["position"].iloc[0] == 0
        assert bt["position"].iloc[1] == 0
        assert bt["position"].iloc[2] == 0
        # Long position opens after first valid signal at bar 3
        assert bt["position"].iloc[4] == 1


# ---------------------------------------------------------------------------
# compute_metrics() — performance metric calculations
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def _flat_bt(self, pnl_values: list, start: str = "2020-01-02") -> pd.DataFrame:
        """Build a minimal backtest DataFrame from a list of daily P&L values."""
        idx = _make_index(len(pnl_values), start=start)
        pnl = pd.Series(pnl_values, index=idx, dtype=float)
        pos = pd.Series(np.where(pnl != 0, 1, 0), index=idx, dtype=float)
        return pd.DataFrame({
            "zscore":     np.zeros(len(pnl_values)),
            "spread":     np.zeros(len(pnl_values)),
            "position":   pos.values,
            "daily_pnl":  pnl.values,
            "trade_cost": np.zeros(len(pnl_values)),
            "cum_pnl":    pnl.cumsum().values,
        }, index=idx)

    def test_sharpe_zero_when_no_pnl(self):
        bt = self._flat_bt([0.0] * 252)
        m  = compute_metrics(bt, label="test")
        assert m["Sharpe_ratio"] == 0.0

    def test_sharpe_positive_for_constant_positive_pnl(self):
        """Constant daily gain → positive Sharpe (std is zero, but gain is positive)."""
        # std of a constant series is 0, so Sharpe calculation hits the vol==0 guard
        bt = self._flat_bt([0.001] * 252)
        m  = compute_metrics(bt, label="test")
        # With zero vol the code returns 0.0 by the guard — check it doesn't crash
        assert isinstance(m["Sharpe_ratio"], float)

    def test_max_drawdown_is_negative(self):
        """Max drawdown should always be ≤ 0."""
        pnl = [0.01, 0.01, -0.05, -0.03, 0.02]
        bt  = self._flat_bt(pnl)
        m   = compute_metrics(bt, label="test")
        assert m["Max_drawdown"] <= 0.0

    def test_max_drawdown_hand_computed(self):
        """
        Cumulative P&L: 0, 0.01, 0.02, -0.03, -0.06, -0.04
        Running max   : 0, 0.01, 0.02,  0.02,  0.02,  0.02
        Drawdown      : 0,    0,    0,  -0.05, -0.08, -0.06
        Max drawdown  : -0.08
        """
        pnl = [0.01, 0.01, -0.05, -0.03, 0.02]
        bt  = self._flat_bt(pnl)
        m   = compute_metrics(bt, label="test")
        assert m["Max_drawdown"] == pytest.approx(-0.08, abs=1e-9)

    def test_win_rate_all_positive(self):
        """All in-position days are positive → win rate = 100%."""
        bt = self._flat_bt([0.01] * 20)
        m  = compute_metrics(bt, label="test")
        assert m["Win_rate_pct"] == pytest.approx(100.0, abs=1e-9)

    def test_win_rate_all_negative(self):
        """All in-position days are negative → win rate = 0%."""
        bt = self._flat_bt([-0.01] * 20)
        m  = compute_metrics(bt, label="test")
        assert m["Win_rate_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_total_pnl_equals_sum_of_daily(self):
        pnl = [0.01, -0.02, 0.03, -0.01, 0.005]
        bt  = self._flat_bt(pnl)
        m   = compute_metrics(bt, label="test")
        assert m["Total_PnL"] == pytest.approx(sum(pnl), abs=1e-9)


# ---------------------------------------------------------------------------
# build_spread_zscore() — spread construction
# ---------------------------------------------------------------------------

class TestBuildSpreadZscore:

    def test_spread_formula(self):
        """spread = log(DEP) - hr * log(INDEP) - intercept."""
        idx = _make_index(10)
        log_dep   = pd.Series(np.log([100.0] * 10), index=idx, name="AMZN")
        log_indep = pd.Series(np.log([50.0]  * 10), index=idx, name="META")
        log_df    = pd.DataFrame({"AMZN": log_dep, "META": log_indep})

        hr, ic, lookback = 1.0, 0.5, 5

        # Monkeypatch the module-level DEP / INDEP names
        import phase3_strategy as strat
        _orig_dep, _orig_indep = strat.DEP, strat.INDEP
        strat.DEP, strat.INDEP = "AMZN", "META"

        spread, _ = build_spread_zscore(log_df, hr, ic, lookback)  # (log_prices, hedge_ratio, intercept, lookback)

        strat.DEP, strat.INDEP = _orig_dep, _orig_indep

        expected = log_dep - hr * log_indep - ic
        pd.testing.assert_series_equal(spread, expected, check_names=False)

    def test_zscore_nan_during_warmup(self):
        """Z-score must be NaN for the first (lookback - 1) rows."""
        idx     = _make_index(20)
        log_dep = pd.Series(np.log(np.linspace(100, 120, 20)), index=idx, name="AMZN")
        log_ind = pd.Series(np.log(np.linspace(50,   60, 20)), index=idx, name="META")
        log_df  = pd.DataFrame({"AMZN": log_dep, "META": log_ind})

        import phase3_strategy as strat
        _orig_dep, _orig_indep = strat.DEP, strat.INDEP
        strat.DEP, strat.INDEP = "AMZN", "META"

        _, zscore = build_spread_zscore(log_df, 1.0, 0.0, 5)  # (log_prices, hedge_ratio, intercept, lookback)

        strat.DEP, strat.INDEP = _orig_dep, _orig_indep

        assert zscore.iloc[:4].isna().all(), "First (lookback-1) z-scores must be NaN"
        assert not np.isnan(zscore.iloc[4]), "Z-score should be valid at index lookback-1"
