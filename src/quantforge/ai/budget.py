"""Token-spend accounting + hard kill-switch.

Reads AI_BUDGET_USD_DAILY / _TOTAL from the environment. charge() records the cost of each model call
(from the API usage fields + per-model pricing) into a persistent ledger shared across users/sessions.
When a cap is hit, allow() returns False and the app degrades gracefully to cached runs.

This is a HARD requirement, not polish — a public URL with paid hooks will get hammered.

TODO(week7): implement a persistent (file/db) ledger keyed by UTC day; price table per model;
allow()/charge()/remaining(); a clear "demo AI budget reached" signal for the UI.
"""

from __future__ import annotations


def allow(estimated_usd: float = 0.0) -> bool:
    """Return True only if a call of ~estimated_usd stays within the daily AND total caps. TODO."""
    raise NotImplementedError


def charge(usd: float, *, model: str, tokens_in: int, tokens_out: int) -> None:
    """Record actual spend into the shared ledger. TODO."""
    raise NotImplementedError


def remaining() -> dict[str, float]:
    """Return {'daily': ..., 'total': ...} USD remaining. TODO."""
    raise NotImplementedError
