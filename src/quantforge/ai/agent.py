"""AI research agent (headline feature, week 8): propose -> backtest -> read metrics -> refine.

The marquee feature AND the rigor talking point. The agent drives the pipeline ONLY through MCP tools,
so it is engine-agnostic (works against Python/R/KNIME/future-MATLAB unchanged).

OVERFITTING GUARDRAILS ARE MANDATORY (see guardrails.split_data):
  - Data is split train / validation / UNTOUCHED holdout.
  - The agent iterates on train+val only, under a hard iteration cap.
  - The holdout is scored ONCE at the end and is NEVER visible to the agent during iteration.
  - An LLM looping against the test set is p-hacking — these guardrails are the whole point.

Cost: every model call goes through budget.charge(); stop when the cap is hit. Use Sonnet for
reasoning, prompt caching on (system prompt + tool schemas cached). Use the `claude-api` skill.

PUBLIC_MODE: parameter-only. The codegen variant (LLM writes new Strategy code) runs LOCAL/DEV ONLY
and must never execute on the public server.
"""

from __future__ import annotations

from typing import Any


def run_research(goal: str, *, max_iters: int = 8, public_mode: bool = False) -> dict[str, Any]:
    """Run the bounded research loop and return the best validated result + the single holdout score.

    TODO(week8): implement the tool-use loop against mcp_server; enforce guardrails + budget;
    in public_mode restrict to vetted strategies + whitelisted params (no codegen exec).
    """
    raise NotImplementedError
