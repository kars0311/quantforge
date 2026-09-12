# Component 16 — `tests/` (correctness, rigor, and safety proof)

**Weeks:** continuous (milestones 3, 6, 8, 9) · **Status:** wk 1–7 suites green; only the
live-AI smoke test skips (`test_nl_interface.py::test_live_smoke`, opt-in via
`QUANTFORGE_LIVE_AI=1`); R-dependent wk-6 suites `skipif` where Rscript/R packages are absent

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
| `test_r_cross_check.py` | subprocess `Rscript tearsheet.R` on the real pipeline's hand-off; metrics asserted at both ~1% (RG-6 contract) and ~1e-9 (transcription), min-variance weights + achieved vol within 1% of PyPortfolioOpt (`skipif` no Rscript/packages — machine-local) | RG-6 | ✅ green (wk 6) |
| `test_r_interchange.py` | Python → R → Python Parquet round trip: R reads and re-writes `write_frame` output, values survive exactly (dates ns-UTC, floats to 1e-12); R-side `stopifnot` guards proven able to fail (`skipif` no Rscript/arrow) | AR-2 | ✅ green (wk 6) |
| `test_r_interchange_verifier.py` / `test_r_tearsheet_verifier.py` / `test_r_tearsheet_png_verifier.py` / `test_r_tearsheet_weights_verifier.py` / `test_r_cross_check_verifier.py` | independent verifier suites (build-verified workflow): re-prove the week-6 R milestones adversarially — hand-computed metric values, failure-path diagnostics with no partial artifacts, PNG/table output, closed-form 2-asset optimum (`skipif` without R) | RG-6, AR-2/AR-3 | ✅ green (wk 6) |
| `test_budget.py` | hand-computed `estimate`; strict daily/total cap boundary; UTC rollover via monkeypatched `_utc_today`; kill-switch (case-insensitive); `allow` never raises; PUBLIC_MODE unset caps deny, dev defaults 2/10; corrupt ledger fail-closed in public mode / warn in dev; persistence across `importlib.reload`; atomic writes leave no temp files; 8-thread charge sum exact; no `anthropic` import | SF-1/SF-2 | ✅ green (wk 7) |
| `test_guardrails.py` | `public_mode()` read at call time; fixed-date `split_data` (exact train/validation ranges, disjoint, out-of-range rows dropped, no ratio args); empty holdout; bad-input scoring does not consume the shot; `rate_limit` boundary (N allowed, N+1 denied), per-key independence, window expiry at exactly 3600 s, env read at call time, state file beside the ledger, corrupt state, persistence across a subprocess; `assert_no_codegen` no-op in dev | RG-4, SF-5, SF-3 | ✅ green (wk 7) |
| `test_mcp_tools.py` | per-tool happy path + rejections against a synthetic `_price_source` that extends into the holdout: bad ticker, malformed/impossible dates, **holdout dates rejected (never truncated) and nothing registered**, unknown handles, unvetted strategy, `validate_params` messages, `cost_bps` bounds, PUBLIC_MODE nested-params violation, `optimize_portfolio` exactly-one-selector rule; tool pipeline equals a direct strategy+engine call; outputs JSON-serialisable; module source has no file/code-execution calls | FR-7, SF-7, SF-8 | ✅ green (wk 7) |
| `test_mcp_server.py` | `build_server()` lists exactly `TOOLS` in order, fresh instance per call; every advertised `inputSchema` is the strict `TOOL_SCHEMAS` entry (`additionalProperties: false`, sorted enums from `UNIVERSE`/`STRATEGIES`/`ENGINES`/`OBJECTIVES`, union `params` bounds equal min/max over `PARAM_WHITELIST`); wire-level jsonschema rejects `code`/`source`/path keys, bad ticker, malformed date, `cost_bps` 101; MCP end-to-end `load_data → run_backtest → get_metrics` equals the direct call; holdout request surfaces an error naming the holdout | FR-7, SF-7, AR-4 | ✅ green (wk 7) |
| `test_holdout_isolation.py` | handle opacity (`__slots__`, no `__dict__`/`vars()`, slots hold no frame, no container dunders, repr is metadata-only and leaks no holdout value, pickle/deepcopy refused), train/val strictly before the holdout, single-use scoring (`HoldoutAlreadyScored` regardless of strategy; a failing first run still consumes), closure returns only floats with `cost_bps=10` | SF-8, RG-4 | ✅ green (wk 7 guard-level; agent-loop clause wk 8) |
| `test_public_mode_no_codegen.py` | PUBLIC_MODE rejects every non-parameter-only action (code-only, `source`/`python`/`eval`/`exec`/`file` keys, missing keys, unvetted strategy, out-of-whitelist and unknown params, nested/callable/bytes values, non-dict shapes) with a `PublicModeViolation` naming the reason; parameter-only actions accepted; same inputs pass with PUBLIC_MODE unset; `budget.allow` blocks past the cap and denies everything when caps are unset in public mode | SF-3, SF-1 | ✅ green (wk 7) |
| `test_nl_interface.py` | mocked-client (`FakeClient`) flows with `ANTHROPIC_API_KEY` unset: golden path with hand-computed spend and ledger bucket; gate order rate → budget → parse → codegen → tools → explain; each fallback code (`rate`, `budget`, `public_mode`, `invalid`, `api_error`) with its exact call/charge count; clarify paths never guess; `charge()` receives actual usage incl. cache tokens; `cache_control` on system + last tool; `messages.create` appears once, inside `_call`; `test_live_smoke` skipped unless `QUANTFORGE_LIVE_AI=1` | FR-8, SF-1, AR-6 | ✅ green (wk 7; 1 opt-in skip) |
| `test_budget_verify.py` / `test_guardrails_verifier.py` / `test_mcp_tools_verify.py` / `test_mcp_server_verify.py` / `test_nl_interface_verify.py` / `test_week7_closeout_verifier.py` | independent verifier suites (build-verified workflow): re-prove the week-7 AI-layer milestones adversarially — cap boundaries, corrupt-ledger policy, holdout rejection through the tools and over MCP, schema strictness, fallback vocabulary and metering | SF-1/SF-3/SF-7/SF-8, FR-7/FR-8 | ✅ green (wk 7) |

## Design notes

- Mocked-Anthropic pattern: a fake client returning canned tool-use responses — AI logic gets
  tested without spend or network; live smoke tests hide behind an env flag, excluded from CI.
- Synthetic/fixture data everywhere (extend the smoke test's `_synthetic_prices` helper into a
  shared `conftest.py` fixture).

## Done when

Every row above green (or justified `skipif`) — at that point each RG/SF claim in the writeup
links to the test that proves it.
