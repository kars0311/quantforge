# Component 12 — `src/quantforge/ai/nl_interface.py` (natural-language interface)

**Week:** 7 · **Status:** built / green (wk 7) — `tests/test_nl_interface.py` (37 offline + 1 env-flagged live smoke) · **Depends on:** mcp_server, budget, guardrails

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
MODEL = "claude-haiku-4-5"          # Haiku for parsing AND explaining (AR-6); bare id, no date suffix
PLAN_TOOL: dict                     # the one forced tool: {strategy, params, tickers, start, end, cost_bps, clarify}
SYSTEM_PARSE / SYSTEM_EXPLAIN: str  # frozen (no timestamps/ids) so the cached prefix is byte-stable

def handle(query: str, *, session_key: str = "anon", client=None) -> dict
    # The one public entry point (UI calls this).
    # Flow: rate_limit(session_key) -> budget.allow(2 * estimate) -> parse (charged) ->
    #       assert_no_codegen(raw plan) -> validate_params -> load_data -> run_backtest ->
    #       get_metrics -> explain (charged) -> return.
    # Returns EXACTLY: {"plan": dict | None, "metrics": dict | None, "explanation": str,
    #                   "spend_usd": float, "fallback": str | None}
    # Any closed gate -> no model call, spend_usd 0.0; "fallback" names the reason so the UI
    # routes to cached scenarios instead of erroring (SF-2).

def parse(query: str, *, client=None) -> dict
    # Haiku with a single forced tool ("plan_backtest") whose schema mirrors run_backtest's
    # inputs: {strategy, params, tickers, start, end, cost_bps}. Unparseable/off-topic ->
    # {"clarify": "<question for the user>"} instead of a guess. ValueError for an empty or
    # >2000-char query (before any spend).
def parse_with_spend(query, *, client=None) -> tuple[dict, float]   # plan + USD charged

def explain(plan: dict, metrics: dict, *, client=None) -> str
    # Haiku: metrics dict -> 3–5 plain-English sentences (what was run, headline numbers,
    # one caveat). No numbers invented — the prompt passes the exact metrics values as sorted
    # JSON. Never empty: a blank reply falls back to a fixed template.
def explain_with_spend(plan, metrics, *, client=None) -> tuple[str, float]

def _call(client, *, system, messages, tools=None, tool_choice=None, max_tokens) -> (response, usd)
    # The ONLY site that performs a Messages API request. Adds cache_control to the system block
    # and the last tool, then charges budget with the response's real usage before returning.
```

### Fallback vocabulary

`handle()["fallback"]` is `None` on success (including a clarification) or one of:

| code | meaning | model calls | spend |
|---|---|---|---|
| `rate` | `guardrails.rate_limit(session_key)` refused | 0 | 0 |
| `budget` | `budget.allow(2 × per-call estimate)` refused, or `AI_DISABLED=on` | 0 | 0 |
| `public_mode` | `PUBLIC_MODE=on` and the parsed plan was not parameter-only | 1 (parse) | parse |
| `invalid` | a tool rejected the plan (holdout dates, unknown ticker, bad cost…); the tool's message is the explanation | 1 (parse) | parse |
| `api_error` | the SDK raised (`anthropic.APIError`); its message is the explanation, never a traceback | 0–1 | what ran |

## Design notes

- `parse` never fabricates defaults for missing essentials (e.g. no date range → asks). Missing
  *optional* fields use the whitelist defaults from `strategies.PARAM_WHITELIST`.
- The tool-use loop is deliberately single-shot (parse → execute → explain), not agentic —
  iteration is the research agent's job (component 13). Keeps NL queries ~$0.001-level cheap.
- The stub's `public_mode` parameter is dropped: PUBLIC_MODE comes from the environment via
  guardrails, so a caller can't accidentally (or maliciously) disable it.

- The PUBLIC_MODE gate (`guardrails.assert_no_codegen`) inspects the model's **raw** plan before
  `validate_params` normalizes it. If validation ran first, a plan carrying a `code` key would
  surface as a polite clarification and the `public_mode` fallback (an abuse signal, not a typo)
  would never fire. `run_backtest` re-runs the same gate; a no-op outside public mode.
- Metering is conservative on purpose: `tokens_in` = uncached + cache-write + cache-read tokens,
  all priced at the full input rate. The ledger can only over-count. Charging happens inside
  `_call`, immediately, so a failure between the two calls never loses spend (SF-1).
- The budget gate asks for **two** calls up front: a query that can afford to parse but not to
  explain is refused whole rather than left half-done.
- Haiku's minimum cacheable prefix is larger than today's prompts, so the `cache_control`
  breakpoints may not engage yet; they cost nothing and engage as the prompts grow.

## Decisions made in build (wk 7)

- **PUBLIC_MODE gate on the raw plan, before `validate_params`** (details above): a `code`
  key in the model's output is reported as the `public_mode` fallback — an abuse signal — not
  softened into a clarification.
- **Conservative cache-token billing**: `_call` charges uncached + cache-write + cache-read
  tokens at the full input rate (missing usage fields count as 0). The ledger over-counts by
  design; prompt caching is a saving the budget does not depend on.
- **`_call` is the only `messages.create` site** and charges immediately after the response,
  so a failure between parse and explain never loses spend; a source-level test pins this.
- **`PLAN_TOOL.params` is a deep copy of `TOOL_SCHEMAS["run_backtest"].params`** so the NL
  plan shape and the tool contract cannot drift apart.
- **Holdout dates reach the model as a rule in the frozen system prompt** and are still
  rejected by `load_data` (fallback `invalid`, message names the holdout) if the model ignores
  it — the tool layer, not the prompt, is the guarantee.
- Prompt caching will not engage on Haiku until the prompts exceed the model's minimum
  cacheable prefix — documented, not a defect; the breakpoints cost nothing.

## Done when

- Golden-path test with a mocked Anthropic client (canned tool-use response) proving:
  gates checked in order, charge() called with actual usage, fallback dict on a closed gate;
  live smoke test behind an env flag (not in CI).

**Status:** done — `tests/test_nl_interface.py` (37 offline tests with a `FakeClient`, synthetic
prices, temp ledger; runs with `ANTHROPIC_API_KEY` unset) plus `test_live_smoke`, skipped unless
`QUANTFORGE_LIVE_AI=1`.
