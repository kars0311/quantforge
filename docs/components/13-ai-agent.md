# Component 13 — `src/quantforge/ai/agent.py` (AI research agent — headline feature)

**Week:** 8 · **Status:** built / green (wk 8) — `tests/test_agent.py`, `test_holdout_isolation.py` (agent-loop clause), `test_public_mode_no_codegen.py` (agent clause) · **Depends on:** mcp_server, guardrails, budget

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
def run_research(goal: str, *,
                 max_iters: int = guardrails.MAX_AGENT_ITERS,
                 tickers: list[str] | None = None,
                 engine: str = "python",
                 cost_bps: float = 10.0,
                 client=None) -> dict
    # goal: e.g. "find a momentum variant with Sharpe > 1 on validation".
    # All arguments are validated BEFORE any spend (no client built, no data loaded, no call).
    # Data enters only through mcp_server.load_data on the loader's fixed train and validation
    # bounds; `tickers=None` means the whole universe. `client` is injectable for tests.
    # Loop (Sonnet tool-use conversation, one model call per iteration):
    #   model calls propose_experiment{strategy, params, rationale} -> runner executes
    #   run_backtest on TRAIN dataset, then on VALIDATION dataset -> both metric sets go back
    #   as the tool_result -> model refines, or calls declare_done{reason}.
    # Stop when: agent declares convergence · max_iters model calls used · budget.allow()
    #   denies · the SDK raises (api_error).
    # THEN (outside the agent conversation): score_holdout(best) exactly once.
    # Returns: {"best": {"strategy", "params", "train_metrics", "val_metrics"} | None,
    #           "holdout_metrics": dict | None,  # the one-shot, honest number
    #           "history": [per-iteration records],   # for the UI's iteration timeline
    #           "stopped_because": "converged" | "max_iters" | "budget" | "api_error",
    #           "spend_usd": float,
    #           "error": str | None}              # the SDK message when stopped_because == "api_error"
```

History record: `{iter, strategy, params, train_metrics, val_metrics, rationale, error}` — rendered
as the Research-mode timeline and saved with cached scenarios. A proposal the tools rejected
(unknown param, out-of-range value, PUBLIC_MODE violation) is kept as a record with `error` set
to the tool's message and both metric sets `None`; it is paid-for history and the timeline
should show it.

## Design notes

- "Best" = highest **validation** Sharpe (not train — that asymmetry is the anti-overfitting
  point, worth one docstring paragraph).
- The agent's context contains *metrics summaries only* — never price data, never holdout dates
  (`load_data` rejects them, layer two of SF-8).
- Honest-reporting rule: the UI must always show validation and holdout metrics side by side;
  a big val/holdout gap is *itself* the demo's teaching moment, not a failure to hide.

## Decisions made in build (wk 8)

- **Two proposal tools, not the four MCP tools.** The model is given `propose_experiment`
  (`{strategy, params, rationale}`; `params` is a deep copy of the MCP `run_backtest` params
  schema) and `declare_done` (`{reason}`), and the runner executes `mcp_server.run_backtest`
  on the train dataset and then the validation dataset on its behalf. One model call per
  iteration is therefore exactly one experiment, which makes `MAX_AGENT_ITERS` a cap on
  `messages.create` calls (SF-7) rather than on a fuzzier notion of "turns". The pipeline is
  still driven only through the tool functions (`load_data` / `run_backtest`): the agent
  module imports no engine or portfolio code and never instantiates a strategy.
- **`strict` tool mode was not used.** The `params` schema has optional nested keys with
  per-strategy defaults; strict mode would force every key to be required. The gate is
  runner-side instead: `assert_no_codegen` then `validate_params` (the same order
  `run_backtest` uses), so a bad proposal is refused before any result handle is minted, and
  the validated dict (defaults merged) is what both backtests — and later `score_holdout` —
  receive.
- **Best selection:** highest validation Sharpe among records with `error is None`; a NaN
  Sharpe ranks as `-inf`; ties go to the earliest iteration. If nothing succeeded, `best` and
  `holdout_metrics` are `None` and neither `split_data` nor `score_holdout` is called.
- **Holdout scoring:** the handle is created only after the loop returns, from the same
  `mcp_server._price_source()` panel the tools used, filtered to the run's `tickers` when a
  subset was given (scoring on a different universe would not measure what the agent iterated
  on). It is a local, scored once, never stored in the result. An empty holdout raises.
- **`error` fields:** the result's `error` carries the SDK message only for `api_error`; each
  history record's `error` carries the tool's rejection message or `None`. The history gathered
  before an `api_error` stop is returned, never discarded — it is paid-for work.
- **Prompt caching breakpoints:** system prompt, last tool schema, and the last content block of
  the last message (so the growing transcript prefix is reused turn to turn). Cache-creation
  and cache-read tokens are billed at the full input rate, as in `nl_interface`.
- **No `public_mode` kwarg** (the stub had one): PUBLIC_MODE comes from the environment only,
  so neither a caller nor a prompt-injected goal can switch the parameter-only gate off.
- **Model:** `claude-sonnet-5` (AR-6; a `budget.PRICES_PER_MTOK` key). Every call is
  `estimate → allow → create → charge` inside the module's single `_call`.

## Done when

- `test_holdout_isolation.py` green (opaque handle + single-use + tool-level rejection, plus the
  agent-loop clause) — **done (wk 8)**; the mocked-client loop tests in `test_agent.py` prove
  cap/budget/api_error stops and that exactly one holdout scoring happens after the loop —
  **done (wk 8)**. A live run producing a coherent history on cached data is **pending the
  personal `ANTHROPIC_API_KEY`** (still a placeholder in `.env`, the same caveat as week 7);
  `tests/test_agent.py::test_live_smoke` runs it with `QUANTFORGE_LIVE_AI=1`.
