"""Two kinds of guardrails in one place.

(1) STATISTICAL (anti-overfitting):
    split_data() -> (train, validation, holdout). The holdout is returned via an opaque handle the
    agent cannot read during iteration; score_holdout() may be called ONCE. Iteration cap enforced by
    the caller (agent.run_research max_iters).

(2) PUBLIC-DEMO SAFETY:
    - rate_limit(key): per-IP/session calls/hour (env AI_RATE_LIMIT_PER_HOUR).
    - PUBLIC_MODE (env): when on, only vetted strategies + whitelisted params are allowed, and
      assert_no_codegen() guarantees no LLM-generated code path can execute.
    - The global budget ledger lives in budget.py; this module gates on it.

TODO(week7-8): implement all of the below; cover with tests (holdout isolation, no-codegen-in-public).
"""

from __future__ import annotations

import os
from typing import Any

PUBLIC_MODE = os.getenv("PUBLIC_MODE", "off").lower() == "on"
VETTED_STRATEGIES = {"momentum", "mean_reversion"}


def split_data(prices, *, train=0.6, validation=0.2):
    """Return (train, validation, holdout_handle). Holdout is NOT directly readable by the agent. TODO."""
    raise NotImplementedError


def score_holdout(holdout_handle, strategy, params) -> dict[str, float]:
    """Score the chosen strategy on the untouched holdout exactly once. TODO: enforce single-use."""
    raise NotImplementedError


def rate_limit(key: str) -> bool:
    """Return True if this key is within its per-hour AI call budget. TODO."""
    raise NotImplementedError


def assert_no_codegen(action: dict[str, Any]) -> None:
    """In PUBLIC_MODE, reject anything that would execute LLM-generated code; allow params-only. TODO."""
    if PUBLIC_MODE:
        raise NotImplementedError  # TODO: validate action is a vetted-strategy + whitelisted-params call
