# Component 16 — `tests/` (correctness, rigor, and safety proof)

**Weeks:** continuous (milestones 3, 6, 8, 9) · **Status:** wk 1–5 suites green; 2 later-week
suites skipped (wk 8–9)

## Function

The proof layer: every rigor and safety claim in the docs has a test that demonstrates it.
`pytest` stays green at every step (the `.claude/` Stop hook enforces it locally; CI on GitHub).
All tests run offline — network-dependent paths use committed fixtures or mocks.

## Requirements satisfied

**QG-1** (always green) · **RG-1/RG-2/RG-5/RG-6** · **SF-1/SF-3/SF-8** — each mapped below.

## Test inventory (file → what it proves → requirement)

| file | proves | req | status |
|------|--------|-----|--------|
| `test_smoke_vertical_slice.py` | momentum → engine → metrics end-to-end on synthetic data | QG-4 | ✅ green |
| `test_engine_vs_backtestingpy.py` | single-asset momentum: equity/Sharpe match `backtesting.py` within 1e-6 rel | RG-5 | skipped → wk 3 |
| `test_interchange_roundtrip.py` (new) | write→read identity per kind; SchemaError on bad frames; wide↔long inverse | AR-2 | wk 1 |
| `test_loader.py` (new) | cache-hit = no network (mocked yfinance); output validates; split bounds sane | FR-1, RG-4 | wk 1 |
| `test_no_lookahead.py` (new) | a deliberately prescient signal (weights = sign of the *same day's* return) earns ~0 through the engine — the shift delays every position one day, killing same-day peeking | RG-1 | wk 3 |
| `test_cost_accounting.py` (new) | hand-computed 2-asset, 5-day case: turnover and cost deductions exact to 1e-12 | RG-2 | wk 3 |
| `test_metrics_reference.py` (new) | each `_KEYS` metric pinned to a hand-computed fixture value | AR-3 | wk 3 |
| `test_strategies.py` | mean-reversion entries/exits on a hand-built oscillating series (both modes); gross ≤ 1; warmup flat; `validate_params` rejections | FR-3, SF-3 | ✅ green (wk 4) |
| `test_interface_freeze.py` | pins the frozen `Strategy`/`Engine`/`BacktestResult` seam by introspection — signatures, dataclass fields/order, abstract-method sets, registry membership — so any interface edit fails loudly | AR-1 | ✅ green (wk 4) |
| `test_portfolio_optimize.py` | Σw = 1 within 1e-8 + long-only bounds (both objectives); min-vol matches a closed-form 2-asset solution; every params/panel/bounds rejection message-matched; `combine_returns` vs hand-computed blend at 1e-12; frontier anchored at min-vol, risk/ret monotone, interchange `weights`/`frontier` kinds validate + round-trip; no input mutation; determinism | FR-4, AR-2 | ✅ green (wk 5) |
| `test_portfolio_combination.py` | FR-4 end-to-end: both vetted strategies → engine → 2-col return panel → `optimize_weights` → `combine_returns` → `compute_metrics`, all via frozen interfaces; blend equals a plain-numpy dot product at 1e-12; convex-blend vol ≤ max individual vol; split discipline (no holdout dates) | FR-4, RG-4 | ✅ green (wk 5) |
| `test_app_shell.py` | UI shell headless: import pulls no `quantforge.ai.*`; chart builders pinned to hand computations (benchmark equity no-look-ahead, drawdown ≤ 0); date pickers bounded to train+val with holdout unselectable; param controls introspect `PARAM_WHITELIST` live; full-script `AppTest` run with zero exceptions | FR-10, RG-4, AR-1 | ✅ green (wk 5) |
| `test_verify_week5_optimize.py` / `test_verify_week5_frontier.py` / `test_portfolio_combination_verifier.py` / `test_app_shell_verifier.py` / `test_week5_closeout_verifier.py` | independent verifier suites (build-verified workflow): re-prove the week-5 milestones (and the docs-closeout state) with adversarial cases written by a separate agent | FR-4, FR-10 | ✅ green (wk 5) |
| `test_r_cross_check.py` (new) | subprocess `Rscript tearsheet.R` on a fixture; metrics + weights within ~1% (`skipif` no Rscript/packages) | RG-6 | wk 6 |
| `test_budget.py` (new) | caps at boundary, UTC rollover, kill-switch, fail-closed, persistence | SF-1/SF-2 | wk 7 |
| `test_mcp_tools.py` (new) | per-tool happy path + rejections (bad ticker, range, params, unknown handle, **holdout dates**) | FR-7, SF-7 | wk 7 |
| `test_holdout_isolation.py` | handle opacity (no attribute/repr leak), single-use scoring, tool-level date clamp | SF-8, RG-4 | skipped → wk 8 |
| `test_public_mode_no_codegen.py` | PUBLIC_MODE rejects codegen/unvetted actions; budget blocks past cap | SF-3 | skipped → wk 9 |
| `test_nl_interface.py` / `test_agent.py` (new) | mocked-client flows: gate order, charge on usage, stop conditions, exactly one holdout scoring | FR-8/9, SF-1 | wk 7–8 |

## Design notes

- Mocked-Anthropic pattern: a fake client returning canned tool-use responses — AI logic gets
  tested without spend or network; live smoke tests hide behind an env flag, excluded from CI.
- Synthetic/fixture data everywhere (extend the smoke test's `_synthetic_prices` helper into a
  shared `conftest.py` fixture).

## Done when

Every row above green (or justified `skipif`) — at that point each RG/SF claim in the writeup
links to the test that proves it.
