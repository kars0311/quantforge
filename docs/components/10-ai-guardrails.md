# Component 10 — `src/quantforge/ai/guardrails.py` (overfitting + public-demo guardrails)

**Weeks:** 7–9 · **Status:** stub · **Depends on:** budget, data/loader (split bounds), strategies (whitelist)

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
  MCP `load_data` tool clamps date ranges to train+val, so even a prompt-injected agent cannot
  request 2023+ data through the tools. Two independent layers.
- `PUBLIC_MODE` is read at call time (not import time, as the current stub does) so tests can
  flip it with `monkeypatch.setenv`.

## Done when

- `test_holdout_isolation.py` and `test_public_mode_no_codegen.py` un-skipped and green;
  rate-limit unit tests green.
