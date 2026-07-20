"""Pairs trading (STRETCH): statistical arbitrage on a cointegrated pair.

Signal: trade the spread between two cointegrated assets when its z-score diverges. The "quant-y" one;
good multiple-testing discussion (selecting pairs from many candidates inflates false positives).

TODO(stretch): cointegration screen, spread z-score, entry/exit; document the multiple-testing caveat.
"""

from __future__ import annotations

from typing import Any

from ..engine.base import Strategy


class PairsStrategy(Strategy):
    name = "pairs"

    def generate_signals(self, prices, params: dict[str, Any]):
        raise NotImplementedError
