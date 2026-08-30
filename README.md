# QuantForge — an AI-driven, polyglot quant research environment

> A self-directed summer project: an open-source quant research pipeline (data → strategies →
> backtest → portfolio optimization → risk/perf analytics), driven by an **AI research agent** over
> **MCP tools**, with a snazzy interactive UI and a cloud-hosted, budget-capped demo.

**This is an independent, open-source project.** It uses only public data and open-source / personal
tooling. It is not affiliated with or built on any employer's internal systems.

---

## Why this exists

- **Build something real and portfolio-worthy** — recognizable quant workflows, implemented with
  rigor (no look-ahead/survivorship bias, transaction costs, out-of-sample evaluation), in code the
  author can defend line-by-line.
- **Show the architecture, not just a toy** — one `Engine`/`Strategy` interface, multiple compute
  backends, an AI agent that drives the whole thing through tools.

## Architecture at a glance

```
data ingestion → strategies → backtest engine → portfolio optimization → risk/perf analytics → UI → cloud
                                       ▲                                          │
                                       └── AI research agent (propose→backtest→read→refine) ◀── MCP tools
```

**Polyglot by design.** Every stage exchanges data via an **Arrow/Parquet interchange contract**, so
each piece can use the best tool for the job:

| Stage | Tool | Status |
|-------|------|--------|
| Backtest engine | Python (custom vectorized) | core |
| Analytics / risk | **R** (tidyquant, PerformanceAnalytics, PortfolioAnalytics) | core |
| Visual workflow engine | KNIME | stretch |
| (future) comparison engine | — | future |

The compute engine sits behind a single interface (`src/quantforge/engine/base.py`), so a new engine
can drop in later without touching the rest of the app. See [`docs/architecture.md`](docs/architecture.md).

## The AI layer

- **MCP server** exposes the pipeline as tools: `load_data`, `run_backtest`, `optimize_portfolio`,
  `get_metrics`.
- **Research agent** proposes a strategy, backtests it, reads its own metrics, and iterates — with
  **mandatory overfitting guardrails** (train / validation / *untouched* holdout, iteration cap,
  budget cap). See `src/quantforge/ai/`.
- **Natural-language interface** turns "backtest momentum on tech, 2015–2020, 10bps costs" into tool
  calls and explains the result in plain English.

> **Two distinct AI uses:** the **Claude API** is a metered, budget-capped *runtime feature* of this
> app. Claude Code (a coding assistant) is a separate *development* tool used to build the repo.

## Safety & cost control (read before deploying publicly)

A public URL that can trigger paid API calls **will** get hit by bots. This repo treats that as a hard
requirement — see `src/quantforge/ai/guardrails.py` and [`docs/architecture.md`](docs/architecture.md):

- Global server-side **budget ledger** (daily + total caps); graceful fallback to cached runs.
- Independent **AWS Budgets alarm** at the infra layer.
- Expensive AI paths **gated** (passcode); open traffic gets cached scenarios only.
- **No LLM-generated code is executed on the public server** — public mode is parameter-only.

## Methodology and known limitations

Honest accounting of what the backtests do and do not claim. Each convention below is enforced by a
dedicated test where one exists; the custom engine is additionally validated against `backtesting.py`
(`tests/test_engine_vs_backtestingpy.py`).

- **Survivorship bias (known limitation).** The universe is a **fixed, hand-picked list of 30 of
  *today's* names** — not a point-in-time constituent history. Companies that were delisted, went
  bankrupt, or shrank out of relevance along the way are absent, so historical results are
  **optimistic**: every name in the panel is, by construction, a survivor. The proper fix — a
  point-in-time universe — is an acknowledged non-goal for v1.0
  ([`docs/product.md` §1](docs/product.md)). The authoritative caveat lives in the module docstring
  of [`src/quantforge/data/loader.py`](src/quantforge/data/loader.py), next to the frozen
  `UNIVERSE` list itself.
- **No look-ahead.** Positions decided using data through the close of day *t* earn day *t+1*'s
  return. The engine enforces this mechanically via `positions.shift(1)` — a signal can never be
  paid for same-day information it could not have known at decision time. Proven by
  [`tests/test_no_lookahead.py`](tests/test_no_lookahead.py), which shows a deliberately prescient
  signal earns nothing once run through the engine.
- **Transaction costs.** Modeled as basis points charged on turnover (`Σ|Δweight|` per day). The
  project's default assumption is **10 bps per unit turnover** ([`docs/product.md` §5.3](docs/product.md)),
  user-overridable via the engine's `cost_bps` parameter. The accounting — hand-computed cost
  series, linearity in the rate, buy/sell/short symmetry, day-of-charge — is proven by
  [`tests/test_cost_accounting.py`](tests/test_cost_accounting.py).
- **Short borrow fees are not modeled (known limitation).** Short positions are charged turnover
  costs like any trade, but the ongoing cost of borrowing shares is ignored, so long–short results
  are slightly optimistic ([`docs/components/04-python-engine.md`](docs/components/04-python-engine.md)).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY, set AI_BUDGET_USD
pytest                          # correctness + rigor + safety tests
streamlit run app/streamlit_app.py
```

R analytics layer:

```bash
Rscript analytics_r/tearsheet.R   # reads the Parquet hand-off, emits a tearsheet
```

## Repo layout

See [`docs/architecture.md`](docs/architecture.md) for the full map. Start here:
[`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) (what to build) and
[`docs/TEN_WEEK_PLAN.md`](docs/TEN_WEEK_PLAN.md) (the week-by-week checklist).

## Status

Scaffold. Modules are stubs with docstrings + `TODO`s — the implementation is the project.
