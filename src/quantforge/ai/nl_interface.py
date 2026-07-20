"""Natural-language interface (week 7): plain English -> tool calls -> plain-English explanation.

Example: "backtest momentum on tech stocks, 2015-2020, 10bps costs" -> parse to structured params via
Claude tool-use -> call the MCP tools -> summarize the result for a human. The demo crowd-pleaser.

Use the cheap model (Haiku) for parsing; prompt caching on. Every call goes through budget.charge()
and respects guardrails (rate limit, PUBLIC_MODE, budget ledger).

TODO(week7): implement parse() and explain().
"""

from __future__ import annotations

from typing import Any


def handle(query: str, *, public_mode: bool = False) -> dict[str, Any]:
    """Parse a NL query, run the pipeline via tools, and return result + a plain-English explanation."""
    raise NotImplementedError
