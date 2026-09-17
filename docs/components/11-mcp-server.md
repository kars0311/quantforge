# Component 11 — `src/quantforge/ai/mcp_server.py` (MCP tool server)

**Week:** 7 · **Status:** built / green (wk 7) — `tests/test_mcp_tools.py`, `tests/test_mcp_server.py` · **Depends on:** loader, engines, strategies, portfolio, metrics, guardrails

## Function

Exposes the pipeline as four read-only, input-validated MCP tools. The NL interface and the
research agent drive the pipeline *only* through these tools — which is exactly why they are
engine-agnostic (a tool call names an engine; the server resolves it via the registry).
Results pass by opaque handle, so tool outputs stay small and the agent never holds raw data.

## Requirements satisfied

- **FR-7** — the four tools. **SF-7** — read-only, validated, no code accepted. **AR-1** — talks
  to `Strategy`/`Engine` interfaces via registries, never concrete classes. **AR-4** — a new
  engine appears here as one registry entry. **SF-8** — `load_data` rejects any window outside train+validation.

## Tools (name → input schema → validation → output)

| tool | inputs | validation | output |
|------|--------|-----------|--------|
| `load_data` | `tickers?: list[str], start: str, end: str` | tickers ⊆ UNIVERSE; dates ISO; **both dates must lie inside train+val bounds** (holdout dates rejected, never truncated — SF-8) | `{dataset_id, tickers, start, end, n_days}` |
| `run_backtest` | `dataset_id, strategy: str, params: dict, engine: str = "python", cost_bps: float = 10` | in order: known dataset_id → params is a dict (default `{}`) → strategy is a str → `assert_no_codegen` (PUBLIC_MODE only) → `strategies.validate_params` → engine ∈ ENGINES → cost_bps ∈ [0, 100]; a non-dict params / non-str strategy is a plain ValueError before the gate | `{result_id, metrics}` |
| `optimize_portfolio` | `result_ids: list[str] \| dataset_id, objective?: str` | exactly one selector; known ids; objective ∈ {max_sharpe, min_volatility} | `{weights, frontier: [{risk, ret}, …]}` |
| `get_metrics` | `result_id` | known id | `{metrics}` (the `_KEYS` dict) |

`run_backtest` passes `guardrails.assert_no_codegen` when PUBLIC_MODE is on (it is the only tool
that carries a `{strategy, params}` payload; the other three accept nothing but enums, ISO dates
and opaque handles, so their own validation is the equivalent gate). No tool accepts code, file
paths, or SQL — strings are enum-checked, numbers are range-checked.

## Interface (integral functions)

```python
ENGINES: dict[str, Engine]              # {"python": PythonEngine(), ...} — the registry (AR-4)
OBJECTIVES = {"max_sharpe", "min_volatility"}   # literal: the tool contract cannot widen silently
TOOL_SCHEMAS: list[dict]                # 4 Anthropic-style {name, description, input_schema},
                                        # TOOLS order, generated from UNIVERSE / STRATEGIES /
                                        # PARAM_WHITELIST / ENGINES / OBJECTIVES at import

def build_server() -> mcp.server.fastmcp.FastMCP
    # Fresh FastMCP("quantforge") per call; registers the four functions via add_tool and
    # advertises the strict TOOL_SCHEMAS input schemas. ValueErrors surface as MCP tool errors.

# In-process registry backing the handles (dataset_id/result_id -> object). Handles are opaque
# random tokens; they index a server-side dict, they are not paths or pickles.
def _get_dataset(dataset_id: str) -> pd.DataFrame      # raises ValueError on unknown id
def _get_result(result_id: str) -> BacktestResult      # raises ValueError on unknown id
def reset_registry() -> None                            # TESTS ONLY
_price_source: Callable[[], pd.DataFrame]               # monkeypatch seam; default = loader cache
```

## Design notes

- The agent and NL interface call tools **in-process** (same Python process as Streamlit) —
  MCP gives the schema/discovery layer, not a network hop. `python -m quantforge.ai.mcp_server`
  runs `build_server().run("stdio")` for external MCP clients (nice demo: point Claude Desktop
  at the pipeline). Not exercised in CI.
- **Reject, don't truncate (SF-8).** `load_data` refuses any window that leaves
  `[train.start, validation.end]` with an error naming the holdout, rather than clamping to the
  allowed range. A clamped request would let an agent *ask* for holdout dates and receive a
  plausible answer, hiding the attempt; a loud error is also what lets the model self-correct.
  The MCP-level test proves the refusal even when synthetic data for those dates exists.
- **`optimize_portfolio` takes exactly one selector.** `result_ids` (≥ 2, unique) allocates
  across strategy return streams (the FR-4 blend); `dataset_id` allocates across the tickers of
  a loaded window (buy-and-hold `pct_change`). Passing both or neither is a ValueError — the two
  panels mean different things and silently preferring one would misreport what was optimized.
- **Schemas are generated, and the `params` schema is a union.** One `params` object serves
  every strategy, so a knob shared by two strategies (`lookback`) carries the loosest bounds
  across them with per-strategy ranges in its description; `validate_params` remains the exact
  gate. On the wire the low-level MCP server validates arguments against the advertised schema
  with `jsonschema` — but only because `build_server` re-registers the call handler with
  `validate_input=True` (FastMCP's default is off; it would otherwise coerce `cost_bps: true`
  to `1.0` and silently drop unknown keys). With it on, `additionalProperties: false` rejects
  `code`/`source`/path keys and wrong types before the function runs; in-process `call_tool`
  relies on the functions' own validation. Both gates are pinned by wire-level tests.
- Tool *results* return metrics and small summaries, never full price frames — keeps tokens
  cheap (AR-6) and keeps raw holdout data structurally unreachable.

## Decisions made in build (wk 7)

- **Reject-not-truncate holdout dates** in `load_data` (SF-8; rationale in the design note
  above and in component 10) — the MCP-level test proves refusal even when synthetic data
  for those dates exists.
- **`ENGINES` is the registry the whole AI layer resolves engines through**, including
  `guardrails.score_holdout`, which imports it lazily inside the function to avoid the
  `mcp_server → guardrails` import cycle.
- **One exception type at the tool boundary.** Every validation failure — unknown handle,
  bad ticker, out-of-range param, holdout dates — is a `ValueError` (never `KeyError`), so an
  agent loop can catch one type and self-correct uniformly; over MCP it becomes a tool error.
- **`OBJECTIVES` is a literal**, not derived from the optimizer, so the tool contract cannot
  widen silently if `optimize.py` grows knobs (`weight_bounds` is deliberately not exposed).
- **Strict schemas overwrite FastMCP's signature-derived ones** in `build_server` (the
  module's one private-attribute access, pinned by a drift-guard test) so the advertised
  `input_schema` is the same object the NL interface caches as its prompt prefix (AR-6).
- **The parameter-only gate runs in `run_backtest` only** — the other three tools have no
  `{strategy, params}` payload to inspect (`assert_no_codegen` requires exactly that key set;
  `load_data`/`optimize_portfolio`/`get_metrics` take enums, ISO dates and opaque handles, and
  their own validation is the equivalent gate); `test_public_mode_no_codegen.py` proves the
  gate at the tool level. Inside `run_backtest` a non-dict `params` / non-str `strategy` is
  rejected as a plain ValueError *before* the gate, so those shapes never surface as
  `PublicModeViolation` (order pinned by `test_handoff_open_items.py`).
- **`optimize_portfolio`'s exactly-one-of selector rule lives in descriptions, not `oneOf`.**
  The strict schemas deliberately use no JSON-Schema combinators (plain
  `additionalProperties: false` objects with sorted enums, so the cached prompt prefix stays
  byte-stable and easy to validate), so the tool description and both the `result_ids` and
  `dataset_id` property descriptions say "pass exactly one of the two"; the function body
  enforces it with a ValueError.

## Done when

- Each tool has a happy-path + rejection test (bad ticker, out-of-range param, holdout dates,
  unknown handle); agent and NL interface run against it end-to-end on cached data.
