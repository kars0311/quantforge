# Component 11 — `src/quantforge/ai/mcp_server.py` (MCP tool server)

**Week:** 7 · **Status:** stub · **Depends on:** loader, engines, strategies, portfolio, metrics, guardrails

## Function

Exposes the pipeline as four read-only, input-validated MCP tools. The NL interface and the
research agent drive the pipeline *only* through these tools — which is exactly why they are
engine-agnostic (a tool call names an engine; the server resolves it via the registry).
Results pass by opaque handle, so tool outputs stay small and the agent never holds raw data.

## Requirements satisfied

- **FR-7** — the four tools. **SF-7** — read-only, validated, no code accepted. **AR-1** — talks
  to `Strategy`/`Engine` interfaces via registries, never concrete classes. **AR-4** — a new
  engine appears here as one registry entry. **SF-8** — `load_data` clamps to train+validation.

## Tools (name → input schema → validation → output)

| tool | inputs | validation | output |
|------|--------|-----------|--------|
| `load_data` | `tickers?: list[str], start: str, end: str` | tickers ⊆ UNIVERSE; dates ISO, **clamped to train+val bounds** (holdout dates rejected — SF-8) | `{dataset_id, tickers, start, end, n_days}` |
| `run_backtest` | `dataset_id, strategy: str, params: dict, engine: str = "python", cost_bps: float = 10` | known dataset_id; `strategies.validate_params`; engine ∈ ENGINES; cost_bps ∈ [0, 100] | `{result_id, metrics}` |
| `optimize_portfolio` | `result_ids: list[str] \| dataset_id, objective?: str` | known ids; objective ∈ {max_sharpe, min_volatility} | `{weights, frontier: [{risk, ret}, …]}` |
| `get_metrics` | `result_id` | known id | `{metrics}` (the `_KEYS` dict) |

Every tool call passes `guardrails.assert_no_codegen` first when PUBLIC_MODE is on. No tool
accepts code, file paths, or SQL — strings are enum-checked, numbers are range-checked.

## Interface (integral functions)

```python
ENGINES: dict[str, Engine]              # {"python": PythonEngine(), ...} — the registry (AR-4)

def build_server() -> mcp.server.Server
    # Constructs the MCP server, registers the four tools with their JSON schemas.

# In-process registry backing the handles (dataset_id/result_id -> object). Handles are opaque
# random tokens; they index a server-side dict, they are not paths or pickles.
def _get_dataset(dataset_id: str) -> pd.DataFrame      # raises on unknown id
def _get_result(result_id: str) -> BacktestResult      # raises on unknown id
```

## Design notes

- The agent and NL interface call tools **in-process** (same Python process as Streamlit) —
  MCP gives the schema/discovery layer, not a network hop. Running `build_server()` over stdio
  also works for external MCP clients (nice demo: point Claude Desktop at the pipeline).
- Tool *results* return metrics and small summaries, never full price frames — keeps tokens
  cheap (AR-6) and keeps raw holdout data structurally unreachable.

## Done when

- Each tool has a happy-path + rejection test (bad ticker, out-of-range param, holdout dates,
  unknown handle); agent and NL interface run against it end-to-end on cached data.
