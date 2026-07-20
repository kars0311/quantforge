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
