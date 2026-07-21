# Component 09 — `src/quantforge/ai/budget.py` (spend accounting + kill-switch)

**Week:** 7 (wired from the FIRST AI call) · **Status:** stub · **Depends on:** nothing

## Function

Dollar accounting for every Claude API call in the app: estimate → `allow()` gate → call →
`charge()` ledger. One global, persistent, server-wide ledger shared by all visitors. When any
cap is hit, `allow()` returns False and the app degrades to cached scenarios — never an error
page. No API call anywhere in the codebase bypasses this module.

## Requirements satisfied

- **SF-1** — the mandatory estimate/allow/charge path with hard caps + kill-switch.
- **SF-2** — global cross-visitor ledger; graceful-fallback signal for the UI.

## Interface (integral functions)

```python
PRICES_PER_MTOK: dict[str, tuple[float, float]]   # model -> (usd/Mtok in, usd/Mtok out)
                                                  # from the claude-api skill's price table

def estimate(model: str, tokens_in: int, tokens_out: int) -> float
    # Pure arithmetic -> estimated USD. Used before every call to feed allow().

def allow(estimated_usd: float = 0.0) -> bool
    # False if: kill-switch env AI_DISABLED=on, OR spent_today + est > AI_BUDGET_USD_DAILY,
    # OR spent_total + est > AI_BUDGET_USD_TOTAL. Missing env caps -> treat as 0 (deny) in
    # PUBLIC_MODE, as a generous dev default otherwise. Never raises — a gate, not an error.

def charge(usd: float, *, model: str, tokens_in: int, tokens_out: int) -> None
    # Append actual spend (computed from the API response's usage fields) to the ledger.

def remaining() -> dict[str, float]
    # {"daily": usd_left_today, "total": usd_left} — rendered in the UI (SF-2 transparency).
```

## Design notes

- **Ledger:** a JSON file (`AI_LEDGER_PATH`, default `data_cache/ai_ledger.json`) mapping UTC
  date → spent USD + call count. File-based because the deploy is a single Fargate task;
  guarded by a file lock since Streamlit serves sessions from multiple threads. (If the app
  ever scales past one container, this becomes DynamoDB/Redis — documented seam, not built.)
- "Daily" is defined by **UTC** date — matches the ledger keys and the AWS Budgets granularity.
- Fail-closed: if the ledger file is corrupt/unreadable, `allow()` returns False in PUBLIC_MODE.

## Done when

- Unit tests: cap enforcement at the boundary, UTC-day rollover, kill-switch, persistence across
  reimport, fail-closed behavior. Grep-level check: no `anthropic` client call outside a
  module that goes through `allow`/`charge`.
