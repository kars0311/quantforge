"""Momentum: buy recent winners (REFERENCE IMPLEMENTATION). Strategy #1.

Signal: trailing return over `lookback` days; long the `top_n` names (equal weight), else long all
positive-momentum names (weight ∝ momentum). All windows are strictly backward-looking; the engine's
shift handles execution timing, so this just expresses target weights per date.

params: {lookback:int=126, top_n:int=0}  (top_n=0 -> weight all positive momentum)
"""

from __future__ import annotations

from typing import Any

from ..engine.base import Strategy


class MomentumStrategy(Strategy):
    name = "momentum"

    def generate_signals(self, prices, params: dict[str, Any] | None = None):
        params = params or {}
        lookback = int(params.get("lookback", 126))
        top_n = int(params.get("top_n", 0))

        prices = prices.sort_index()
        mom = prices / prices.shift(lookback) - 1.0  # NaN during the warmup window

        if top_n > 0:
            selected = (mom.rank(axis=1, ascending=False) <= top_n).astype(float)
            counts = selected.sum(axis=1)
            weights = selected.div(counts.where(counts != 0), axis=0).fillna(0.0)
        else:
            positive = mom.clip(lower=0.0)
            row_sum = positive.sum(axis=1)
            weights = positive.div(row_sum.where(row_sum != 0), axis=0).fillna(0.0)

        # Warmup rows (momentum undefined) -> no position.
        weights.loc[mom.isna().all(axis=1)] = 0.0
        return weights
