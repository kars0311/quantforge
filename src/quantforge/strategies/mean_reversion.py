"""Mean-reversion: fade short-term extremes. Strategy #2 (week 4).

Signal: z-score of price vs a rolling mean; long when oversold, short/flat when overbought. Strictly
backward-looking windows.

TODO: implement generate_signals(prices, params) with params {lookback, entry_z, exit_z}.
"""

from __future__ import annotations

from typing import Any

from ..engine.base import Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def generate_signals(self, prices, params: dict[str, Any]):
        raise NotImplementedError
