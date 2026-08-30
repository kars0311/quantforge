"""PythonEngine — the core custom vectorized backtester (REFERENCE IMPLEMENTATION).

Small, correct, fully vectorized (no per-row Python loops). This is the heart of the project and the
strongest rigor signal — understand every line.

Inputs (wide form, simplest for a vectorized backtest):
- prices:    DataFrame, index=date, columns=ticker, adjusted close.
- positions: DataFrame, index=date, columns=ticker, TARGET weights decided at the close of each date.

Rigor:
- **No look-ahead:** a weight decided at close of day t earns day t+1's return -> `positions.shift(1)`.
- **Transaction costs:** charged on turnover (sum of |weight changes|) at `cost_bps`.
- **Input validation:** malformed inputs raise `ValueError` up front instead of producing a
  silently-wrong equity curve — the failure modes below are exactly the ones that corrupt a
  backtest while still returning plausible-looking numbers.

Validated against `backtesting.py` in `tests/test_engine_vs_backtestingpy.py`.
"""

from __future__ import annotations

from typing import Any

from pandas.api.types import is_numeric_dtype

from ..metrics.performance import compute_metrics
from .base import BacktestResult, Engine

#: Tolerance on the gross-exposure check. Equal-weight books built from 1/n arithmetic land a
#: few ULPs above 1.0; 1e-9 forgives float noise while still catching any real leverage.
_GROSS_EPS = 1e-9


def _validate_inputs(prices, positions) -> None:
    """Reject inputs that would make the backtest silently wrong (all raise ValueError).

    Why each check exists:

    - **Duplicated dates** in the prices index: `sort_index()` cannot repair them, and
      `pct_change` across a repeated label fabricates a zero-return day while alignment
      double-counts it. No correct interpretation exists, so we refuse.
    - **Non-monotonic dates:** an out-of-order index means the caller's data pipeline is
      broken upstream (the loader and interchange contract always deliver sorted dates).
      Silently re-sorting would *mask* that bug — and with look-ahead prevention riding
      entirely on `shift(1)` over a time-ordered index, we insist the caller sends data
      already in causal order rather than guessing on their behalf.
    - **Non-numeric columns:** a stray string/object column (e.g. a ticker column that leaked
      in from long format) turns arithmetic into concatenation or NaN soup deep inside the
      run. Fail at the door with the column named.
    - **Gross exposure sum(|w|) > 1 + eps:** this engine models an unlevered book. Long-short
      is fine as long as |longs| + |shorts| stays within 1.0 (e.g. +0.5 / -0.5); weights whose
      absolute values exceed 1.0 in total imply borrowed money whose financing cost we do not
      model, so the resulting equity curve would be flattering fiction. This is the classic
      buggy-strategy symptom (weights normalized by sum instead of sum of |.|), caught here at
      the engine boundary rather than discovered in a too-good Sharpe.
    """
    if not prices.index.is_unique:
        dupes = prices.index[prices.index.duplicated()].unique()
        raise ValueError(
            f"prices index has duplicated dates (e.g. {dupes[0]!r}); "
            "sort_index() cannot repair duplicates — deduplicate upstream."
        )
    if not prices.index.is_monotonic_increasing:
        raise ValueError(
            "prices index is not monotonic increasing; the engine requires time-ordered data "
            "(an unsorted index signals a broken data pipeline upstream — sort it there)."
        )
    for frame_name, frame in (("prices", prices), ("positions", positions)):
        non_numeric = [str(col) for col in frame.columns if not is_numeric_dtype(frame[col])]
        if non_numeric:
            raise ValueError(
                f"{frame_name} contains non-numeric column(s) {non_numeric}; "
                "the engine needs purely numeric wide-form data."
            )
    gross_exposure = positions.abs().sum(axis=1)  # NaN weights ignored (they align to 0 later)
    over = gross_exposure[gross_exposure > 1.0 + _GROSS_EPS]
    if not over.empty:
        raise ValueError(
            f"positions has gross exposure sum(|w|) = {over.iloc[0]:.6f} > 1 on {over.index[0]!r} "
            f"({len(over)} row(s) total). This engine models an unlevered book: long-short within "
            "gross 1.0 is fine, levered books are not (financing costs are unmodeled)."
        )


class PythonEngine(Engine):
    name = "python"

    def run_backtest(
        self, prices, positions, params: dict[str, Any] | None = None
    ) -> BacktestResult:
        """Turn (prices, target weights) into net returns, an equity curve, and metrics.

        NaN policy (explicit, on purpose):
        - **NaN prices** (pre-IPO raggedness — a name not yet listed) contribute **0 return**:
          `pct_change` yields NaN where prices are missing and `fillna(0.0)` zeroes it. A stock
          that does not trade yet can neither earn nor lose, and any weight parked on a
          NaN-price day is therefore effectively flat — capital sits idle, which is the honest
          model (interpolating or back-filling a price would invent returns that never existed).
        - **NaN positions** after alignment to the price grid become **0.0** (flat): a strategy
          that emits no weight for a date/ticker holds nothing there, it does not carry stale
          weights forward.
        """
        params = params or {}
        cost_bps = float(params.get("cost_bps", 0.0))

        _validate_inputs(prices, positions)

        prices = prices.sort_index()
        positions = positions.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)

        asset_returns = prices.pct_change().fillna(0.0)
        held = positions.shift(1).fillna(0.0)  # decided at t, applied to t+1 (no look-ahead)
        gross = (held * asset_returns).sum(axis=1)

        turnover = held.diff().abs().sum(axis=1).fillna(0.0)
        costs = turnover * (cost_bps / 10_000.0)
        net = gross - costs

        equity = (1.0 + net).cumprod()
        metrics = compute_metrics(net)
        meta = {
            "engine": self.name,
            "cost_bps": cost_bps,
            "params": params,
            "start": str(prices.index[0]),
            "end": str(prices.index[-1]),
            "n_days": len(prices.index),
            "total_turnover": float(turnover.sum()),
        }
        return BacktestResult(equity_curve=equity, returns=net, metrics=metrics, meta=meta)
