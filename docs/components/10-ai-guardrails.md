# Component 10 — `src/quantforge/ai/guardrails.py` (overfitting + public-demo guardrails)

**Weeks:** 7–9 · **Status:** built / green (wk 7) — `tests/test_guardrails.py`, `test_holdout_isolation.py` (guard level; the agent-loop clause lands in week 8 with `ai/agent.py`), `test_public_mode_no_codegen.py` · **Depends on:** budget, data/loader (split bounds), strategies (whitelist)

## Function

Two guard families in one module. (a) **Statistical:** enforce the train/val/holdout discipline —
the agent iterates on train+val only; the holdout hides behind an opaque handle and is scored
exactly once. (b) **Public-demo safety:** PUBLIC_MODE parameter-only enforcement, rate limiting,
and routing to cached fallback when budget/rate gates close.

## Requirements satisfied

- **RG-4** — split enforcement + one-shot holdout. **SF-3** — `assert_no_codegen` (proven by
  `test_public_mode_no_codegen.py`). **SF-5** — rate limiting. **SF-7** — iteration cap constant
  lives here. **SF-8** — holdout opacity (proven by `test_holdout_isolation.py`). **SF-2/SF-4** —
  gate results routed to fallback, passcode checked in the UI against this module's helpers.

## Interface (integral functions)

```python
MAX_AGENT_ITERS = 10                      # product.md §5 decision — the hard cap agent.py obeys

class HoldoutHandle:
    # Opaque wrapper around the holdout price slice. The data lives in a closure — not an
    # attribute — so the agent (and repr/dir/str) cannot read it. Tracks .consumed.

class HoldoutAlreadyScored(RuntimeError): ...
class PublicModeViolation(ValueError): ...

def split_data(prices: pd.DataFrame | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame, HoldoutHandle]
    # Slices by loader.get_split_bounds() FIXED DATES (2010-19 / 2020-22 / 2023-26.5) — not
    # ratios; the stub's ratio args are dropped so there is exactly one split definition (RG-4).
    # Returns (train, validation, holdout_handle); prices defaults to loader.load_prices().

def score_holdout(handle: HoldoutHandle, strategy: str, params: dict,
                  engine: str = "python") -> dict[str, float]
    # Runs the vetted strategy on the hidden holdout slice ONCE; returns metrics.
    # Second call on the same handle -> HoldoutAlreadyScored. Called by the research runner
    # AFTER the agent loop ends — never by the agent itself.

def rate_limit(key: str) -> bool
    # Sliding per-hour call count per session/IP key vs AI_RATE_LIMIT_PER_HOUR (default 20).
    # Same persistence pattern as the budget ledger.

def assert_no_codegen(action: dict) -> None
    # action = {"strategy": str, "params": dict} — the ONLY action shape accepted in
    # PUBLIC_MODE. Delegates to strategies.validate_params (vetted name + whitelisted values);
    # any other key, any code/string-eval payload, any unvetted value -> PublicModeViolation.
```

## Design notes

- Holdout opacity is enforced in-process (closure + single-use flag) **and** structurally: the
  MCP `load_data` tool **rejects** any date range that leaves train+val (it never truncates —
  see below), so even a prompt-injected agent cannot request 2023+ data through the tools.
  Two independent layers.
- `PUBLIC_MODE` is read at call time via `public_mode()` (not import time, as the original
  stub did) so tests can flip it with `monkeypatch.setenv`.

## Decisions made in build (wk 7)

- **Reject, don't truncate, holdout dates.** `mcp_server.load_data` raises a ValueError naming
  the holdout and both allowed bounds for any window outside `[train.start, validation.end]`.
  Clamping would let an agent *ask* for holdout data and get a plausible answer back, hiding
  the attempt; a loud error is also what lets the model self-correct.
- **Rate-limit state file derived from `AI_LEDGER_PATH`.** `rate_limit` keeps
  `{key: [unix_ts]}` at `budget._ledger_path().with_name("ai_rate_limits.json")`, written
  with the budget module's locked atomic writer — one env var relocates all AI state. A
  corrupt state file is treated as empty (with a warning) in every mode, unlike the money
  ledger: the worst case is one extra hour of calls for one key, and `budget.allow` still
  stands behind it.
- **`score_holdout` lazy-imports `mcp_server.ENGINES`** (function-local import) — the engine
  registry lives with the tools (AR-4) and `mcp_server` imports `guardrails`, so a module-level
  import would be a cycle. Check order: handle type → `validate_params` → engine → already
  consumed → **set `consumed=True` before running**. Caller mistakes (bad strategy/param/
  engine) therefore do not spend the one shot; a crash mid-run does — a failed attempt on the
  holdout is still an attempt.
- **`HoldoutHandle` is metadata-only from the outside**: `__slots__` (no `__dict__`), a repr
  that shows only start/end/n_days/consumed, no container dunders, and pickling/deepcopy
  refused with `TypeError` so the slice cannot be smuggled out by serialization.
- **`assert_no_codegen` requires the exact key set `{strategy, params}`** in PUBLIC_MODE, with
  flat scalar param values; `code`/`source`/`python`/`eval`/`exec`/`file` keys fall out of
  that rule rather than a blocklist that could be evaded by renaming.
- The agent-loop assertion in `test_holdout_isolation.py` (that `agent.run_research` never
  touches the handle and the runner scores it once after the loop) is a **week-8** item,
  landing with `ai/agent.py`; the week-7 tests prove the guard itself.

## Done when

- `test_holdout_isolation.py` and `test_public_mode_no_codegen.py` un-skipped and green;
  rate-limit unit tests green. **Done (wk 7)** at guard level — the agent-loop clause is
  added in week 8.
