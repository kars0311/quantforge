# handoff.md — running project status

Living status doc for QuantForge. Updated after every workflow run completes (see AGENTS.md).
Newest entry first.

---

## 2026-09-10 — Week 7 complete: MCP + NL interface + budget/guardrails (`build-verified` workflow)

**Status: Week 7 complete and green.** `ruff check .` clean; fresh full `pytest` at closeout:
**879 passed / 1 skipped** (baseline at run start was 488 passed / 2 skipped; the
two old skips were the wk-8/9 placeholder stubs in `test_holdout_isolation.py` and
`test_public_mode_no_codegen.py`, both now un-skipped and green — the single remaining skip is
`tests/test_nl_interface.py::test_live_smoke`, the live-API smoke that runs only with
`QUANTFORGE_LIVE_AI=1`). Growth is the five week-7 suites plus the workflow verifiers' proving
suites; `tests/test_week7_closeout_verifier.py` lands after this count, as the week-6
verifier did; after that verifier and the cleaner's polish pass the final fresh run is
**919 passed / 1 skipped** (re-run by hand on 2026-09-11, ruff clean). All three Week-7 boxes
in `docs/TEN_WEEK_PLAN.md` are ticked (21 total). Week 6 was committed as `f2e8cd0`;
**Week 7 is uncommitted** — commit pending Kent's approval (see open items). Note: an earlier,
unrecorded Week-7 attempt was hard-reverted at Kent's request on 2026-09-10 before this run;
this entry describes the rebuilt version only.

### What was built this run

1. **`src/quantforge/ai/budget.py`** (stub → complete) — `PRICES_PER_MTOK`, `estimate`,
   `allow`, `charge`, `remaining`; JSON ledger keyed by UTC day at `AI_LEDGER_PATH`, atomic
   temp-file + `os.replace` writes under a thread lock + `fcntl.flock`; caps re-read from the
   environment every call; kill-switch `AI_DISABLED=on`; fail-closed policy (a cap that
   resolves to ≤ 0 denies outright, so unset caps in PUBLIC_MODE deny every call; corrupt
   ledger → deny and refuse to overwrite in public mode, warn-and-continue in dev).
   Proven by `tests/test_budget.py` (35 tests incl. an 8-thread exact-sum charge test).
2. **`src/quantforge/ai/guardrails.py`** (stub → complete) — `MAX_AGENT_ITERS`,
   `VETTED_STRATEGIES`, `public_mode()` read at call time, `HoldoutHandle` (slots-only,
   metadata repr, unpicklable; the slice lives in a closure), `split_data` by the loader's
   fixed dates, `score_holdout` (one shot; `consumed` set before the run; lazy-imports
   `mcp_server.ENGINES`), `rate_limit` (sliding hour, state file beside the ledger),
   `assert_no_codegen` (exact `{strategy, params}` key set, flat scalar values, whitelist).
   Proven by `tests/test_guardrails.py` (20), `tests/test_holdout_isolation.py` (9,
   un-skipped) and `tests/test_public_mode_no_codegen.py` (26, un-skipped).
3. **`src/quantforge/ai/mcp_server.py`** (stub → complete) — `ENGINES` registry, `OBJECTIVES`,
   opaque `ds_…`/`res_…` handle registries, the four validated tools (`load_data` rejects —
   never truncates — any window touching the holdout; `run_backtest` runs
   `assert_no_codegen` + `validate_params`; `optimize_portfolio` takes exactly one selector;
   `get_metrics`), generated strict `TOOL_SCHEMAS`, `build_server()` (FastMCP, strict schemas
   advertised and validated on the wire), `python -m quantforge.ai.mcp_server` stdio entry.
   Proven by `tests/test_mcp_tools.py` (58) and `tests/test_mcp_server.py` (15).
4. **`src/quantforge/ai/nl_interface.py`** (stub → complete) — Haiku (`claude-haiku-4-5`)
   forced-tool `parse`, `explain`, and `handle` over the tools; `_call` is the only
   `messages.create` site and charges the budget immediately with conservative
   (uncached + cache-write + cache-read at full input rate) metering; prompt caching
   breakpoints on the system prompt + last tool; gate order rate → budget (2× per-call
   estimate) → parse → `assert_no_codegen` on the raw plan → tools → explain; fallback
   vocabulary `{rate, budget, public_mode, invalid, api_error}`. Proven by
   `tests/test_nl_interface.py` (37 offline with a `FakeClient` and `ANTHROPIC_API_KEY`
   unset, + 1 env-flagged live smoke).
5. **Verifier suites** (independent agents, adversarial): `test_budget_verify.py`,
   `test_guardrails_verifier.py`, `test_mcp_tools_verify.py`, `test_mcp_server_verify.py`,
   `test_nl_interface_verify.py`.
6. **Docs/config closeout (this entry)** — `.env.example` now carries every variable in the
   `docs/components/18-runtime-config.md` table with a comment (dev defaults 2/10 when caps are
   unset; PUBLIC_MODE with unset caps denies everything); `QUANTFORGE_LIVE_AI` documented as
   test-only in `tests/test_nl_interface.py`, not as a runtime setting; component docs 09/10/11/12
   status stub → built/green (wk 7) with "Decisions made in build" lists (reject-not-truncate
   holdout dates; conservative cache-token billing; rate-limit file derived from
   `AI_LEDGER_PATH`; `score_holdout` lazy-imports `ENGINES`); 18 status → `.env.example`
   complete for week 7; `16-tests.md` rows for the five suites + the two un-skipped suites
   green (wk 7), week-7 verifier row added, status "wk 1–7 suites green; only the live-AI
   smoke test skips"; Week-7 plan boxes ticked; README "Methodology and known limitations"
   gained an "AI layer safety" paragraph pointing at `ai/budget.py`, `ai/guardrails.py`, the
   parameter-only PUBLIC_MODE rule and the proving tests; the week-3/4/5/6 closeout verifiers'
   plan-count / handoff-position / 16-tests-status pins advanced to the week-7 state, per
   precedent. A stale "load_data clamps" wording in `11-mcp-server.md` / `13-ai-agent.md` was
   corrected to "rejects" to match the built behaviour.

### Agent failures and resolutions

None — all six build milestones and their independent verifier suites completed and returned
green on the first pass; no stalls, retries, or manual verifications were needed this run.

### Open items (carried forward)

1. Commit Week 7 (the four `ai/` modules, the seven week-7 suites + five verifier suites +
   `test_week7_closeout_verifier.py`, `.env.example`/`.gitignore`, and the doc sync) once Kent
   approves — per repo practice, commits happen only with his explicit approval. (Week 6 is
   already committed as `f2e8cd0`.)
2. `ANTHROPIC_API_KEY` is still a placeholder in `.env`, so the live smoke test
   (`QUANTFORGE_LIVE_AI=1 pytest tests/test_nl_interface.py -k live_smoke`) has **not** been
   run; every AI-layer proof so far is offline with a fake client. Run it once the personal key
   is in place (costs well under a cent).
3. Prompt caching will not engage on Haiku until the prompts exceed the model's minimum
   cacheable prefix — the `cache_control` breakpoints are in place and cost nothing; documented
   in `docs/components/12-nl-interface.md`, not a defect.
4. UI wiring of `nl_interface.handle` (chat panel, fallback routing to cached scenarios,
   `DEMO_PASSCODE` gate) is a **week 9** item; the app shell still imports no `quantforge.ai.*`.
5. The agent-loop clause of `tests/test_holdout_isolation.py` (that `agent.run_research` never
   touches the handle and the runner scores it exactly once after the loop) lands in week 8
   with `ai/agent.py`; week 7 proves the guard itself.

**Next up (per docs/TEN_WEEK_PLAN.md): Week 8 — AI research agent** (`ai/agent.py`,
`run_research`: propose → backtest → read metrics → refine over the MCP tools on train +
validation only, `MAX_AGENT_ITERS` cap, budget cap; one-shot holdout via `score_holdout`
after the loop; `test_agent.py` + the agent clause of `test_holdout_isolation.py`).

---

## 2026-09-01 — Week 6 complete: R analytics layer + polyglot cross-check (`build-verified` workflow)

**Status: Week 6 complete and green.** `ruff check .` clean; fresh full `pytest` at closeout:
**488 passed / 2 skipped** (baseline at run start was 437 passed / 2 skipped; growth is the
week-6 R suites plus the workflow verifiers' proving suites — the final +4 over the last
builder milestone's 484 is `test_r_cross_check_verifier.py` landing after that count; the 2
skips are the unchanged wk 8–9 stubs). All three Week-6 boxes in `docs/TEN_WEEK_PLAN.md` are
ticked — the third ("pressure-test the stats methodology") is satisfied by the new
"Cross-language conventions and caveats" section in `docs/components/08-r-tearsheet.md`.
**Uncommitted** — commit pending Kent's approval (see open items).

**CI caveat:** every R-dependent suite `skipif`s when `Rscript` or the required R packages
(arrow, xts, tidyquant, PerformanceAnalytics, PortfolioAnalytics, ROI, ROI.plugin.quadprog)
are absent — so the RG-6 cross-check is **machine-local**: it executes fully on this machine
(all 488 pass with zero R skips locally) but skips in bare CI. The counts above are from this
machine with the full R stack installed.

### What was built this run

1. **`tests/test_r_interchange.py`** — Python → R → Python Parquet round trip certifying the
   week-1 `SCHEMAS` dict as the frozen cross-language contract: R (`arrow`) reads a
   `write_frame` file, asserts schema/tz via named `stopifnot`s, writes it back; Python
   re-validates and matches dates exactly (ns-UTC) and floats to 1e-12. Includes a
   guard-the-guard test proving the R-side assertions can actually fail.
2. **`analytics_r/tearsheet.R`** (stub → complete) — one linear, heavily commented headless
   Rscript (each step annotated with its Python equivalent): CLI + loud boundary validation
   mirroring `interchange.validate_frame`; `compute_metrics_r` transcribing the six `_KEYS`
   with Python's exact conventions (ANN=252, risk-free 0, population std via
   `sd(r)*sqrt((n-1)/n)`, n=1 edge matched); PNG tearsheet via
   `charts.PerformanceSummary` (file device, headless) + display-only
   `table.AnnualizedReturns`/`table.Drawdowns`; PortfolioAnalytics/ROI/quadprog long-only
   full-investment **min-variance** second optimizer; writes `metrics_r.parquet` and
   `weights_r.parquet` in interchange kinds. Every failure path exits non-zero with a named
   diagnostic and no partial artifacts.
3. **`tests/test_r_cross_check.py`** — the RG-6 proof: real pipeline (both vetted strategies →
   `PythonEngine` → panel → `optimize_weights`) handed to `Rscript` via the interchange files;
   metrics asserted at both ~1% (the RG-6 contract) **and** ~1e-9 (transcription agreement);
   min-variance weights within 1% per asset (observed ~5e-6) and achieved annualized vol
   within 1% relative of PyPortfolioOpt's.
4. **Verifier suites** (independent agents, adversarial): `test_r_interchange_verifier.py`,
   `test_r_tearsheet_verifier.py`, `test_r_tearsheet_png_verifier.py`,
   `test_r_tearsheet_weights_verifier.py`, `test_r_cross_check_verifier.py` — hand-computed
   metric values, closed-form 2-asset optimum, failure-path diagnostics, PNG/table output.
5. **Docs closeout (this entry)** — Week-6 plan boxes ticked; methodology pressure-test notes
   ("Cross-language conventions and caveats") added to `docs/components/08-r-tearsheet.md`
   with a pointer from README's "Methodology and known limitations"; `08-r-tearsheet.md`
   status stub → built/green; `07-portfolio.md`'s "RG-6 NOT met" caveat flipped to done;
   `16-tests.md` inventory updated (R rows green wk 6); the week-3/4/5 closeout verifiers'
   plan/handoff pins advanced to the week-6 state, per precedent. Environment prep along the
   way: `ROI.plugin.quadprog` installed from CRAN.

### Agent failures and resolutions

None — all five build milestones and their independent verifier suites completed and returned
green on the first pass; no stalls, retries, or manual verifications were needed this run.

### Open items (carried forward)

1. **Resolved this run (long-carried since week 1): R-side Parquet read check.** The item
   asked for `Rscript -e 'arrow::read_parquet(...)'` on a written file; delivered as the
   stronger `tests/test_r_interchange.py` full round trip (R reads, asserts, and re-writes;
   Python re-validates) — the interchange schema is now certified from both languages and can
   be treated as frozen for the polyglot boundary.
2. Commit Week 6 (tearsheet.R + the six R suites + doc sync) once Kent approves — per repo
   practice, commits happen only with his explicit approval.
3. CI note (informational, not a defect): the R cross-check runs only where the R stack is
   installed (see the caveat above); bare CI exercises the Python-side suites and skips the R
   ones with named `skipif` reasons.

**Next up (per docs/TEN_WEEK_PLAN.md): Week 7 — MCP + natural-language interface**
(`ai/mcp_server.py` exposing `load_data`/`run_backtest`/`optimize_portfolio`/`get_metrics`;
`ai/nl_interface.py` Claude tool-use; `ai/budget.py` caps + `ai/guardrails.py` wired from
day one).

---

## 2026-08-31 — Week 5 complete: portfolio layer + Streamlit UI shell (`build-verified` workflow)

**Status: Week 5 complete and green.** `ruff check .` clean; `pytest` **428 passed / 2 skipped**
(437 passed / 2 skipped after the closeout verifier's own suite
`test_week5_closeout_verifier.py` landed following this entry, matching the week-4 precedent;
baseline at run start was 304 passed / 2 skipped; growth is the three new week-5 suites plus
the workflow verifiers' proving suites `test_verify_week5_optimize.py` /
`test_verify_week5_frontier.py` / `test_portfolio_combination_verifier.py` /
`test_app_shell_verifier.py`; the 2 skips are the unchanged wk 8–9 stubs). All three Week-5
boxes in `docs/TEN_WEEK_PLAN.md` are ticked. **Uncommitted** — commit pending Kent's approval
(see open items).

### What was built this run

1. **`src/quantforge/portfolio/optimize.py`** (stub → working) — `optimize_weights` (pypfopt
   mean-variance with textbook estimators — `mean_historical_return` + `sample_cov`, 252-day
   annualization — max_sharpe/min_volatility, long-only bounds, Σw = 1 within 1e-8, whitelist
   param validation with named `ValueError`s, inputs never mutated), `frontier` (min-vol-anchored
   target-return sweep, fresh `EfficientFrontier` per point, (risk, ret) frame self-validated
   against the interchange `frontier` kind), and `combine_returns` (label-aligned fixed-weight
   blend named "portfolio"; rejects |Σw − 1| > 1e-8 rather than silently renormalizing). Both
   optimizer entry points share one `_estimate_moments` helper so they can never drift onto
   different estimators — deliberately kept to textbook defaults for the week-6 R cross-check.
2. **`tests/test_portfolio_optimize.py`** (48 tests) — Σ/bounds invariants for both objectives;
   min-vol pinned to an in-test closed-form 2-asset solution; `combine_returns` vs hand-computed
   blends at 1e-12; every rejection class message-matched; interchange `weights`/`frontier`
   validation + Parquet round-trips; determinism and no-mutation proofs.
3. **`tests/test_portfolio_combination.py`** (13 tests) — FR-4 end-to-end through frozen
   interfaces only (zero production-code change): momentum + mean-reversion via the `STRATEGIES`
   registry → `PythonEngine` (cost_bps=10) → inner-joined panel → `optimize_weights` →
   `combine_returns` → `compute_metrics`; blend equals a plain-numpy dot product at 1e-12;
   convex-blend vol ≤ max individual vol; split discipline (all dates ≤ validation end,
   holdout untouched).
4. **`app/streamlit_app.py`** (stub → week-5 shell) + **`tests/test_app_shell.py`** (16 tests) —
   sidebar config (UNIVERSE tickers, date pickers hard-bounded to train+validation so the
   holdout is unselectable, `PARAM_WHITELIST`-driven param controls, cost_bps, Python-only
   engine selector); tabs Backtest · Portfolio · Methodology live against cached Parquet
   (`run_pipeline` under `st.cache_data`; refuses to download — cache must be primed outside the
   UI), AI Chat · Research mode as labeled placeholders for weeks 7/8 with zero `quantforge.ai`
   imports (subprocess-proven); pure Plotly builders (equity vs shift-consistent equal-weight
   benchmark, drawdown, frontier with max-Sharpe star); `render_metrics` iterates exactly
   `metrics/performance._KEYS`. Full-script `streamlit.testing.v1.AppTest` runs with zero
   exceptions; manual headless boot served HTTP 200.

### Agent failures and resolutions

None — all four build milestones and their independent verifier suites completed and returned
green on the first pass; no stalls, retries, or manual verifications were needed this run. The
docs-closeout milestone (this entry) also advanced the week-3/week-4 closeout verifiers' plan
and handoff pins to the week-5 state, per the precedent recorded in the 2026-08-30 entry.

### Open items (carried forward)

1. Deferred to week 6 (before schema freeze): R-side Parquet read check
   (`Rscript -e 'arrow::read_parquet(...)'` on a written prices file).
2. **Next rigor item (week 6):** R PortfolioAnalytics cross-check of the Python optimizer
   results — weights and achieved vol within ~1% (RG-6). Until it lands, the RG-6 portion of
   `docs/components/07-portfolio.md`'s done-when is explicitly NOT met (stated in that doc);
   the optimizer is single-implementation verified only.
3. Commit Week 5 (portfolio layer + UI shell + suites + doc sync) once Kent approves — per repo
   practice, commits happen only with his explicit approval.

**Next up: Week 6 — R analytics layer** (`analytics_r/tearsheet.R` reading the Parquet
hand-off; tidyquant + PerformanceAnalytics tearsheet; PortfolioAnalytics as the second
optimizer for the cross-check).

---

## 2026-08-30 — Week 4 complete: mean-reversion + frozen seam (`build-verified` workflow)

**Status: Week 4 complete and green.** `ruff check .` clean; `pytest` **304 passed / 2 skipped**
(291 at closeout time; +13 from the closeout verifier's own suite landing after this entry)
(baseline at run start was 184 passed / 2 skipped; growth is the two new week-4 suites, the new
interchange-guard test, and the workflow verifiers' proving suites
`test_mean_reversion_verifier.py` / `test_strategy_registry_verifier.py` /
`test_strategies_week4_verifier.py` / `test_interface_freeze_verifier.py`; the 2 skips are the
unchanged wk 8–9 stubs). Both Week 4
boxes in `docs/TEN_WEEK_PLAN.md` are ticked. Committed 2026-08-30 (`35453b6`, "week 4
complete") with Kent's approval and pushed.

### What was built this run

1. **`src/quantforge/strategies/mean_reversion.py`** — `MeanReversionStrategy`: rolling z-score
   `(P − SMA)/SD` (window = `min_periods` = lookback, strictly backward-looking), hysteresis via
   a vectorized entry/exit event state machine (entry wins same-bar overlap; NaN z forces flat
   and severs the ffill chain), `long_flat` and `long_short` modes with equal weight across
   active names (Σw = 1 long-only, Σ|w| = 1 gross long-short), warmup rows flat. Short-side
   friction caveat (borrow, locate, squeeze) documented in the module docstring.
2. **`src/quantforge/strategies/__init__.py`** — the SF-3 vetted set: `STRATEGIES` registry
   (momentum + mean_reversion, matches `ai/guardrails.VETTED_STRATEGIES`), `PARAM_WHITELIST`
   with the exact doc bounds/defaults, and `validate_params` (merge-with-defaults, never
   mutates input, `ValueError` naming the offending param; bools rejected for numeric params,
   integral floats accepted for int params).
3. **`tests/test_strategies.py`** — 21-test proof suite: hand-derived z-scores pin the exact
   entry/hold/exit rows in both modes (cross-checked by an independent numpy recomputation),
   gross/warmup/shape invariants, tail-truncation (no look-ahead), hysteresis path dependence,
   every `validate_params` rejection class, and end-to-end runs through `PythonEngine`.
4. **Frozen seam** — `src/quantforge/engine/base.py` docstrings now declare the interface
   FROZEN as of week 4 (verified docstring-only by AST comparison; zero behavior change), and
   `tests/test_interface_freeze.py` (12 introspection tests) pins signatures, dataclass
   fields/order, abstract sets, and registry membership so any interface edit fails loudly.
5. **`to_long` duplicate-column guard** (`src/quantforge/interchange.py`) — clears open item #1:
   a wide frame with duplicated column labels now raises `SchemaError` naming the duplicated
   label(s) instead of a bare pandas `AttributeError`; proven by a new test in
   `tests/test_interchange_wide_long.py`. No other interchange behavior changed (round-trip
   suites untouched and green). Docs synced: plan boxes ticked; status headers in
   `docs/components/03-engine-base.md` / `05-strategies.md` / `16-tests.md` updated; the
   week-3 closeout verifier's plan/handoff pins advanced to the week-4 state (history entry
   now looked up by content, not position).

### Open items (carried forward)

1. Deferred to week 6 (before schema freeze): R-side Parquet read check
   (`Rscript -e 'arrow::read_parquet(...)'` on a written prices file).
2. Resolved 2026-08-30: Week 4 committed as `35453b6` and pushed.

**Next up (per docs/TEN_WEEK_PLAN.md):** Week 5 — portfolio layer (`portfolio/optimize.py`,
PyPortfolioOpt mean-variance + efficient frontier, strategy combination) + Streamlit UI shell.

---

## 2026-08-29 — Week 3 rigor pass complete (`build-verified` workflow)

**Status: Week 3 complete and green.** `ruff check .` clean; `pytest` **184 passed / 2 skipped**
(baseline at run start was 130 passed / 2 skipped; growth is the new rigor suites plus the
workflow verifiers' proving tests). All Week 1–3 boxes in `docs/TEN_WEEK_PLAN.md` are now
ticked. Committed 2026-08-30 ("week 3 complete") with Kent's approval — everything except
`tests/test_tmp_validation_has_teeth.py`, which stays uncommitted pending his review.

### What was built this run

1. **Engine hardening** (`src/quantforge/engine/python_engine.py`): input validation raising
   `ValueError` on duplicated/non-monotonic price indexes, non-numeric columns, and gross
   exposure > 1.0; NaN policy documented in the docstring (NaN price ⇒ 0 return, NaN position ⇒
   flat); `meta` enriched with `n_days` and `total_turnover`. Zero numerical change for valid
   inputs — equity curves verified byte-identical to the pre-change engine.
2. **`tests/test_no_lookahead.py`** — proves a prescient signal earns statistically nothing
   through the engine, a day-*t* position earns exactly day *t+1*'s return, day 0 is flat, and
   tail truncation never changes earlier returns. Mutating away `positions.shift(1)` makes it
   fail in the profitable direction (teeth confirmed).
3. **`tests/test_cost_accounting.py`** — hand-computed 2-asset/5-day case pinning net returns,
   equity, and `total_turnover` to 1e-12; zero-cost default, cost linearity in bps, buy/sell/
   short symmetry, and day-1 entry-cost timing. A sed-mutated half-cost engine fails 4/6 tests.
4. **`tests/test_metrics_reference.py`** — pins every `_KEYS` metric in `metrics/performance.py`
   to in-test numpy recomputations on literal returns (population std ddof=0, strict `r > 0`
   hit-rate, exact ordered key list — the week-6 R cross-language contract). Mutation-checked.
5. **Docs**: README gained a "Methodology and known limitations" section (survivorship caveat
   with pointer to the authoritative note in `data/loader.py`, no-look-ahead and cost-model
   summaries citing their proving tests); `docs/components/02-data-loader.md` "~28 names" → 30
   (closes the stale open item from 2026-07-21).

### Notes

- `tests/test_tmp_validation_has_teeth.py` (untracked) was **deliberately left untouched**
  throughout this run — it is a temporary meta-test awaiting Kent's review.
- The engine-vs-`backtesting.py` validation item was already done before this run (see the
  manual-session entry below).
- Post-run fix by the main session: `docs/components/16-tests.md`'s row for
  `test_no_lookahead.py` described an inverted construction (sign of *tomorrow's* return earns
  ~0 — actually earns positive under the engine convention); reworded to match the implemented
  same-day-peek test, per the milestone-2 verifier's finding.

### Open items (carried forward)

1. Polish: `to_long` still lacks the `df.columns.is_unique` guard (bare pandas error instead of
   `SchemaError` on duplicate column labels).
2. Deferred to week 6 (before schema freeze): R-side Parquet read check
   (`Rscript -e 'arrow::read_parquet(...)'` on a written prices file).
3. Resolved 2026-08-30: the temporary `tests/test_tmp_validation_has_teeth.py` passed and Kent
   chose deletion. Also removed the closeout verifier's `test_temporary_meta_test_left_in_place`
   guard — it enforced a run-scoped instruction (and referenced an untracked file, so it would
   have failed on a fresh clone).

**Next up (per docs/TEN_WEEK_PLAN.md):** Week 4 — mean-reversion strategy + freeze the
`Strategy`/`Engine` interfaces in `engine/base.py`.

---

## 2026-08-29 — Engine validated against backtesting.py (manual session, no workflow)

**Status: green.** `ruff check .` clean; `pytest` 129 passed / 2 skipped.

- Installed `backtesting` (0.6.5) into the venv — it was declared in `requirements.txt` but missing.
- Implemented `tests/test_engine_vs_backtestingpy.py` (was a `TODO(week3)` skip): single-asset
  long/flat momentum run through both `PythonEngine` and `backtesting.py` on identical seeded data.
  Flat OHLC bars + `trade_on_close=True` make the fill convention match the engine's
  `positions.shift(1)`; equity curves agree to rtol 5e-4, Sharpe/total-return/max-drawdown match.
  This checks off the "validate the engine vs backtesting.py" item of Week 3.
- Week 1 work is now committed (`3f95ad7`); the older "commit pending approval" note below is stale.
- Still open from the 2026-07-21 entry: the `~28 names` → 30 doc fix in
  `docs/components/02-data-loader.md`, and the missing `df.columns.is_unique` guard in `to_long`.

**Next up:** rest of Week 3 rigor — explicit no-look-ahead and cost-accounting pytest coverage,
survivorship caveat documentation. This session's work is uncommitted.

---

## 2026-07-21 — Week 1: interchange + loader (`build-verified` workflow, run `wf_54587a13-997`)

**Status: Week 1 complete and green.** `ruff check .` clean; `pytest` 128 passed / 3 skipped
(skips are the still-stubbed later-week modules). All work is **uncommitted** pending Kent's
approval.

### What was built

- `src/quantforge/interchange.py` — 7-kind schema contract (`SCHEMAS`, `SchemaError`,
  `validate_frame`), Parquet `write_frame`/`read_frame` (validates before touching disk; dates
  pinned to ns-UTC on read), `to_wide`/`to_long` for the three panel kinds.
- `src/quantforge/data/loader.py` — `UNIVERSE` (30 tickers), `START`/`END`/`SPLITS`,
  `get_split_bounds()`, `load_prices` with a whole-universe Parquet cache (cache hit never
  imports yfinance; cache miss downloads once).
- Six new test files (~70 tests): `test_interchange_schema.py`, `test_interchange_roundtrip.py`,
  `test_interchange_wide_long.py`, `test_interchange_verifier.py`, `test_loader_constants.py`,
  `test_loader.py`.

### How the 7th (final) agent failed

The workflow planned 7 milestones; 6 were built **and** independently verified. Milestone 7
("tests/test_loader.py — mocked yfinance, no network") was **built successfully**, but its
verifier agent never returned: it stalled with no progress after ~23 minutes, the workflow
retried it, and the retry died on an API error ("Connection closed mid-response") ~40 minutes
in. This was an infrastructure failure, not a code defect — the workflow correctly stopped and
reported `failed-verification` for that milestone.

**Resolution:** the main session performed the verification manually — read
`tests/test_loader.py` (confirmed genuinely offline: yfinance either poisoned so import raises,
or replaced by a canned fake that records calls) and ran the full suite green. Milestone 7 is
therefore considered verified, by hand rather than by agent.

### Open items

1. Doc fix: `docs/components/02-data-loader.md` says "~28 names" but lists (and the code
   implements) 30 — update the wording to 30.
2. Polish: `to_long` on duplicate column labels raises a bare pandas `AttributeError` instead
   of `SchemaError`; a `df.columns.is_unique` check would fix it (not reachable from `to_wide`
   output).
3. Deferred to week 6 (before schema freeze): run the R-side check —
   `Rscript -e 'arrow::read_parquet(...)'` on a written prices file. A pytest already pins the
   physical Parquet types to the contract.
4. Commit Week 1 (code + tests + `.claude/workflows/build-verified.js`) once Kent approves.
5. The workflow gained a post-run **Clean** phase on 2026-07-22 (after this run); Week 1's
   files have not had that polish pass. Option: resume `wf_54587a13-997` to run just the
   cleaner from cache.

### Run stats

15 agents, ~688k subagent tokens, ~1h47m wall clock. One builder stall-retry (milestone 6)
recovered on its own; only the milestone-7 verifier was lost.

**Next up (per docs/TEN_WEEK_PLAN.md):** Week 2 — engine + momentum are already the working
vertical slice, so the next build target is rigor tests + engine-vs-backtesting.py validation.
