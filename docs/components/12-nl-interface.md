# Component 12 — `src/quantforge/ai/nl_interface.py` (natural-language interface)

**Week:** 7 · **Status:** stub · **Depends on:** mcp_server, budget, guardrails

## Function

The demo crowd-pleaser: "backtest momentum on tech stocks, 2015–2020, 10bps costs" → structured
plan (Claude Haiku, forced tool-use) → validated MCP tool calls → plain-English explanation of
the metrics. Every call metered; every gate (rate limit, budget, PUBLIC_MODE) respected.

## Requirements satisfied

- **FR-8** — NL → tools → NL. **SF-1** — estimate/allow/charge around both model calls.
- **AR-6** — Haiku for parsing *and* explaining (cheap, fast); prompt caching on (system prompt
  + tool schemas). **SF-3/SF-5** — parses into the same validated action shape as everything else.

## Interface (integral functions)

```python
def handle(query: str, *, session_key: str = "anon") -> dict
    # The one public entry point (UI calls this).
    # Flow: rate_limit(session_key) -> budget.allow(est) -> parse -> [assert_no_codegen]
    #       -> MCP tool calls -> explain -> budget.charge (actual usage) -> return.
    # Returns: {"plan": dict, "metrics": dict | None, "explanation": str,
    #           "spend_usd": float, "fallback": str | None}
    # Any closed gate -> no model call; "fallback" names the reason ("budget", "rate", ...)
    # so the UI routes to cached scenarios instead of erroring (SF-2).

def parse(query: str) -> dict
    # Haiku with a single forced tool ("plan_backtest") whose schema mirrors run_backtest's
    # inputs: {strategy, params, tickers, start, end, cost_bps}. Unparseable/off-topic ->
    # {"clarify": "<question for the user>"} instead of a guess.

def explain(plan: dict, metrics: dict) -> str
    # Haiku: metrics dict -> 3–5 plain-English sentences (what was run, headline numbers,
    # one caveat). No numbers invented — the prompt passes the exact metrics values.
```

## Design notes

- `parse` never fabricates defaults for missing essentials (e.g. no date range → asks). Missing
  *optional* fields use the whitelist defaults from `strategies.PARAM_WHITELIST`.
- The tool-use loop is deliberately single-shot (parse → execute → explain), not agentic —
  iteration is the research agent's job (component 13). Keeps NL queries ~$0.001-level cheap.
- The stub's `public_mode` parameter is dropped: PUBLIC_MODE comes from the environment via
  guardrails, so a caller can't accidentally (or maliciously) disable it.

## Done when

- Golden-path test with a mocked Anthropic client (canned tool-use response) proving:
  gates checked in order, charge() called with actual usage, fallback dict on a closed gate;
  live smoke test behind an env flag (not in CI).
