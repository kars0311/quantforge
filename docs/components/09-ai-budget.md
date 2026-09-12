# Component 09 — `src/quantforge/ai/budget.py` (spend accounting + kill-switch)

**Week:** 7 (wired from the FIRST AI call) · **Status:** built / green (wk 7) — `tests/test_budget.py` · **Depends on:** nothing

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

## Decisions made in build (wk 7)

- **A cap that resolves to ≤ 0 is "no budget" and denies outright** — even `allow(0.0)`.
  Without this rule the strict-`>` boundary would let a zero-cost call through when caps are
  unset in PUBLIC_MODE, contradicting fail-closed. Unset/blank/unparseable caps → `0` in
  public mode, `2` / `10` USD dev defaults otherwise; caps are re-read from the environment
  on every call so an operator can tighten them without a restart.
- **Corrupt-ledger policy is centralized** (`_load_ledger_with_policy`): PUBLIC_MODE →
  `allow()` False, `remaining()` zeros, and `charge()` raises `LedgerCorruptError` rather
  than overwriting (a silently reset ledger would erase real spend history); dev mode treats
  the file as empty with a logged warning.
- **Conservative cache-token billing.** The one caller that reads API usage
  (`nl_interface._call`) charges `input + cache_creation + cache_read` tokens all at the full
  input rate, so the ledger can only over-count relative to the invoice. Prompt-cache
  discounts are a saving the ledger deliberately does not take credit for.
- **Atomic, locked writes.** Temp file + `os.replace` under a module `threading.Lock` plus
  `fcntl.flock` on `<ledger>.lock`; 8 threads × 10 charges sum exactly in tests. The
  rate-limit state file (component 10) reuses this writer and lives beside the ledger, so
  `AI_LEDGER_PATH` is the single knob for where AI state goes.
- `_utc_today()` is a separate function so tests monkeypatch the day boundary instead of
  the clock.

## Done when

- Unit tests: cap enforcement at the boundary, UTC-day rollover, kill-switch, persistence across
  reimport, fail-closed behavior. Grep-level check: no `anthropic` client call outside a
  module that goes through `allow`/`charge`.
