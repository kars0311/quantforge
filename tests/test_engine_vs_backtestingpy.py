"""Correctness: the custom PythonEngine must agree with `backtesting.py` on a known strategy.

This is the proof the engine is right — the single most important rigor test. Run the SAME momentum
strategy through both and assert the equity curves / Sharpe match within tolerance.

Why the comparison is set up the way it is:

- **Single asset, long/flat.** `backtesting.py` is an event-driven, single-instrument framework;
  our engine is a vectorized multi-asset one. The overlap where both are exactly defined is one
  asset with weight 1 (in) or 0 (out) — so that is where we prove agreement.
- **Flat bars (Open=High=Low=Close).** The engine only sees closes; an event-driven framework
  also sees the intrabar path. Making every bar a point removes any path-dependent difference.
- **`trade_on_close=True`.** Our engine's no-look-ahead rule is `positions.shift(1)`: a weight
  decided at the close of day t earns day t+1's close-to-close return. `trade_on_close` makes
  backtesting.py fill the order placed on bar t at bar t's close — the identical convention.
  (The default — fill at bar t+1's open — would lag the engine by one day on flat bars.)
- **Huge cash.** backtesting.py trades whole shares; a large cash/price ratio makes the
  fractional-share remainder (the only unavoidable difference) ~1e-7 of equity per trade.
- **Zero costs both sides** (`cost_bps=0`, `commission=0`) so timing, not cost models, is tested.

Sharpe is computed by running OUR `compute_metrics` on both daily-return streams rather than
trusting backtesting.py's own Sharpe, whose annualization conventions differ; the point is that
the engines produce the same returns, not that two libraries share a formula.

TODO(week3): engine-vs-backtesting.py validation is DONE (this file, 2026-08-29). Still
remaining for the Week 3 rigor pass: dedicated no-look-ahead and cost-accounting pytest
coverage, and the documented survivorship caveat.
"""

import numpy as np
import pandas as pd
from backtesting import Backtest
from backtesting import Strategy as BtStrategy

from quantforge.engine.python_engine import PythonEngine
from quantforge.metrics.performance import compute_metrics
from quantforge.strategies.momentum import MomentumStrategy

LOOKBACK = 20


def _synthetic_prices(n_days: int = 400, seed: int = 7) -> pd.Series:
    """A seeded geometric random walk with drift — enough flips to force several round trips."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=0.0005, scale=0.01, size=n_days)
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    return pd.Series(50.0 * np.cumprod(1.0 + rets), index=idx, name="close")


class _BtMomentum(BtStrategy):
    """Long when trailing LOOKBACK-day return is positive, flat otherwise.

    Mirrors MomentumStrategy on one asset: with a single column, its positive-momentum mode
    yields weight 1.0 exactly when momentum > 0 and 0.0 otherwise (including warmup).
    """

    def init(self):
        pass

    def next(self):
        if len(self.data) <= LOOKBACK:
            return  # warmup: momentum undefined, stay flat (engine side holds 0.0 too)
        mom = self.data.Close[-1] / self.data.Close[-1 - LOOKBACK] - 1.0
        if mom > 0 and not self.position:
            self.buy()
        elif mom <= 0 and self.position:
            self.position.close()


def test_python_engine_matches_backtesting_py():
    close = _synthetic_prices()

    # --- our engine ---
    prices = close.to_frame("TICK")
    positions = MomentumStrategy().generate_signals(prices, {"lookback": LOOKBACK, "top_n": 0})
    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 0.0})
    ours = result.equity_curve / result.equity_curve.iloc[0]

    # The comparison is vacuous unless the strategy actually trades.
    n_flips = int(positions["TICK"].diff().abs().gt(0).sum())
    assert n_flips >= 4, f"strategy barely trades on this seed ({n_flips} position changes)"

    # --- backtesting.py on identical data ---
    ohlc = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000},
        index=close.index,
    )
    bt = Backtest(
        ohlc,
        _BtMomentum,
        cash=50_000_000,
        commission=0.0,
        trade_on_close=True,
        exclusive_orders=True,
    )
    stats = bt.run()
    assert stats["# Trades"] >= 1
    theirs = stats["_equity_curve"]["Equity"]
    theirs = theirs / theirs.iloc[0]

    # --- agreement ---
    assert len(ours) == len(theirs)
    np.testing.assert_allclose(ours.to_numpy(), theirs.to_numpy(), rtol=5e-4)

    their_metrics = compute_metrics(theirs.pct_change().fillna(0.0))
    assert abs(result.metrics["sharpe"] - their_metrics["sharpe"]) < 1e-2
    assert abs(result.metrics["total_return"] - their_metrics["total_return"]) < 5e-4
    assert abs(result.metrics["max_drawdown"] - their_metrics["max_drawdown"]) < 5e-4
