"""PythonEngine — the core custom vectorized backtester (REFERENCE IMPLEMENTATION).

Small, correct, fully vectorized (no per-row Python loops). This is the heart of the project and the
strongest rigor signal — understand every line.

Inputs (wide form, simplest for a vectorized backtest):
- prices:    DataFrame, index=date, columns=ticker, adjusted close.
- positions: DataFrame, index=date, columns=ticker, TARGET weights decided at the close of each date.

Rigor:
- **No look-ahead:** a weight decided at close of day t earns day t+1's return -> `positions.shift(1)`.
- **Transaction costs:** charged on turnover (sum of |weight changes|) at `cost_bps`.

Validate against `backtesting.py` in tests (week 3).
"""

from __future__ import annotations

from typing import Any

from ..metrics.performance import compute_metrics
from .base import BacktestResult, Engine


class PythonEngine(Engine):
    name = "python"

    def run_backtest(self, prices, positions, params: dict[str, Any] | None = None) -> BacktestResult:
        params = params or {}
        cost_bps = float(params.get("cost_bps", 0.0))

        prices = prices.sort_index()
        positions = positions.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)

        asset_returns = prices.pct_change().fillna(0.0)
        held = positions.shift(1).fillna(0.0)              # decided at t, applied to t+1 (no look-ahead)
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
        }
        return BacktestResult(equity_curve=equity, returns=net, metrics=metrics, meta=meta)
