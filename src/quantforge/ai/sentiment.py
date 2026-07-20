"""LLM news-sentiment alt-data signal (STRETCH).

Headline/earnings text -> Claude sentiment score -> a feature that feeds a strategy. A credible
"alternative data" angle (alt-data is a real quant-internship topic). Bounded cost: batch, cache by
(ticker, date), and route through budget.charge().

TODO(stretch): fetch headlines, score with the cheap model, cache, expose as a signal column.
"""

from __future__ import annotations


def sentiment_signal(tickers: list[str], start: str, end: str):
    """Return a date x ticker sentiment feature aligned to the prices schema. TODO."""
    raise NotImplementedError
