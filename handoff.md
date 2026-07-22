# handoff.md — running project status

Living status doc for QuantForge. Updated after every workflow run completes (see AGENTS.md).
Newest entry first.

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
