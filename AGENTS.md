# AGENTS.md — build guide for this repo

You (an AI coding agent) are building out this project from a scaffold. This file is the operating
manual. Read `docs/PROJECT_BRIEF.md`, `docs/architecture.md`, and `docs/TEN_WEEK_PLAN.md` before
starting.

## What this is

An open-source, AI-driven, **polyglot** quant research pipeline: data → strategies → backtest →
portfolio optimization → risk/perf analytics → interactive UI → cloud, with an AI research agent and
a natural-language interface driving the pipeline over **MCP tools**.

## Non-negotiable rules

1. **Statistical rigor (this is the whole point):**
   - **No look-ahead bias.** A position decided using data through day *t* earns day *t+1*'s return.
     The engine enforces this with `positions.shift(1)`. Never let a signal peek at same-day data it
     couldn't have known at decision time.
   - **Transaction costs** modeled on turnover.
   - **Survivorship bias** acknowledged (fixed historical universe; document the caveat).
   - **Out-of-sample:** the AI agent gets train/validation only; the **holdout is scored once** and
     never optimized against. An LLM iterating on the test set is p-hacking — prevent it.
   - **Prove correctness:** validate the custom engine against `backtesting.py`
     (`tests/test_engine_vs_backtestingpy.py`).
2. **Safety / cost (a public URL with paid AI hooks is a liability):**
   - All Claude API calls go through `ai/budget.py` (hard `AI_BUDGET_USD_*` caps + kill-switch).
   - `PUBLIC_MODE=on` ⇒ **parameter-only**: never execute LLM-generated code on the server; only
     vetted strategies + whitelisted params. Gate live AI behind `DEMO_PASSCODE`; rate-limit.
3. **Architecture discipline:**
   - Everything depends on the `Strategy`/`Engine` interfaces in `engine/base.py`. **Freeze them.**
   - Cross-stage / cross-language hand-offs use the **Arrow/Parquet contract** in `interchange.py`.
   - `metrics/performance.py` is the single source of truth for metrics; the R layer must match it.
4. **Defensibility:** add docstrings that explain *why*, not just *what*. The human author must be
   able to explain every component in an interview. Prefer clear, conventional code over cleverness.

## Quality gates (ENFORCED — do not skip)

This repo ships with hooks in `.claude/` that run automatically:
- **After any code edit** (`Write`/`Edit`): **ruff lint** must pass.
- **At the end of every turn** (`Stop`): the **pytest suite** must be green.

If a gate fails, the failure is fed back to you — **fix it before continuing or finishing.** Even when
hooks aren't active (e.g., a different tool), treat these as mandatory: whenever you add or change
code, run `ruff check .` and `pytest` yourself and keep both green. Never leave the suite red.

## Don't swap these choices

Python 3.12 + numpy/pandas; `backtesting.py` for validation; `PyPortfolioOpt`; R via
tidyquant/PerformanceAnalytics/PortfolioAnalytics; Streamlit + Plotly; Anthropic `anthropic` + `mcp`;
Docker + Terraform/AWS Fargate. Do **not** introduce MATLAB this summer (it's the next-phase add).

## Build order

Follow `docs/TEN_WEEK_PLAN.md`. Dependency order: interchange + loader → engine + momentum → rigor +
validation test → mean-reversion + freeze interfaces → portfolio → R layer → MCP + NL → agent +
guardrails → UI → deploy + harden. Keep `pytest` green at every step.

## Reference implementation already present

A **working vertical slice** exists as the pattern to extend (run `pytest` — it passes):
`metrics/performance.py`, `engine/python_engine.py`, `strategies/momentum.py`, verified by
`tests/test_smoke_vertical_slice.py` on synthetic data. Match its style and rigor for everything else.
Modules still stubbed raise `NotImplementedError` with a `TODO(week…)` marker.

## When building the AI layer

Use the **`claude-api` skill**. Turn on **prompt caching** (cache the system prompt, tool schemas, and
price data). Use Haiku for NL parsing and Sonnet for agent reasoning; Opus only if clearly needed.
Every model call: estimate → `budget.allow()` → call → `budget.charge()`.

## Status handoff (keep current)

`handoff.md` at the repo root is the living project-status doc. **After every workflow run the
user executes completes, update it** — prepend a dated entry with: what was built/changed, test
suite status (ruff + pytest counts), any agent failures and how they were resolved, open items,
and the next build target. Newest entry first. This is the first thing a fresh session should
read to pick up where the last one left off.

## Run / test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                  # keep green
streamlit run app/streamlit_app.py
```

`conftest.py` puts `src/` on the path, so imports like `from quantforge...` work without installing.
