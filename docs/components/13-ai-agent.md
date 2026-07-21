# Component 13 — `src/quantforge/ai/agent.py` (AI research agent — headline feature)

**Week:** 8 · **Status:** stub · **Depends on:** mcp_server, guardrails, budget

## Function

The propose → backtest → read metrics → refine loop, run entirely through the MCP tools on
train+validation data, under hard iteration and budget caps. When the loop ends, the *runner*
(not the agent) scores the winning configuration once on the holdout via
`guardrails.score_holdout`. The rigor story: an LLM iterating against the test set is p-hacking,
and this architecture makes that impossible, not just discouraged.

## Requirements satisfied

- **FR-9** — the research loop; engine-agnostic because it only sees tools. **RG-4/SF-8** — train+val
  only; one-shot holdout, scored outside the loop. **SF-1** — metered per iteration; loop stops
  when `allow()` says no. **SF-3** — in PUBLIC_MODE proposals are parameter-only (vetted strategy +
  whitelisted params; the codegen variant is local/dev only and OFF the public server). **SF-7** —
  `guardrails.MAX_AGENT_ITERS = 10` (product.md §5; supersedes the stub's `max_iters=8` default).
  **AR-6** — Sonnet for reasoning, prompt caching on (system prompt + tool schemas). **DL-3**.

## Interface (integral function)

```python
def run_research(goal: str, *, max_iters: int = guardrails.MAX_AGENT_ITERS) -> dict
    # goal: e.g. "find a momentum variant with Sharpe > 1 on validation".
    # Loop (Sonnet tool-use conversation, tools = the 4 MCP tools):
    #   propose {strategy, params} -> run_backtest on TRAIN dataset -> read metrics ->
    #   run_backtest on VALIDATION dataset -> agent reflects and refines or declares done.
    # Stop when: agent declares convergence · max_iters reached · budget.allow() denies.
    # THEN (outside the agent conversation): score_holdout(best) exactly once.
    # Returns: {"best": {"strategy", "params", "train_metrics", "val_metrics"},
    #           "holdout_metrics": dict,          # the one-shot, honest number
    #           "history": [per-iteration records],   # for the UI's iteration timeline
    #           "stopped_because": "converged" | "max_iters" | "budget",
    #           "spend_usd": float}
```

History record: `{iter, strategy, params, train_metrics, val_metrics, rationale}` — rendered as
the Research-mode timeline and saved with cached scenarios.

## Design notes

- "Best" = highest **validation** Sharpe (not train — that asymmetry is the anti-overfitting
  point, worth one docstring paragraph).
- The agent's context contains *metrics summaries only* — never price data, never holdout dates
  (`load_data` clamps them, layer two of SF-8).
- Honest-reporting rule: the UI must always show validation and holdout metrics side by side;
  a big val/holdout gap is *itself* the demo's teaching moment, not a failure to hide.

## Done when

- `test_holdout_isolation.py` green (opaque handle + single-use + tool clamp); a mocked-client
  loop test proves cap/budget stops and that exactly one holdout scoring happens; live run
  produces a coherent history on cached data.
