# Architecture

## Layers

```
data ingestion → strategies → backtest engine → portfolio optimization → risk/perf analytics → UI → cloud
                                       ▲                                          │
                                       └── AI research agent (propose→backtest→read→refine) ◀── MCP tools
```

## The swappable-engine seam

Two thin interfaces (`src/quantforge/engine/base.py`) are the keystone:

- **`Strategy`** — `generate_signals(prices, params) -> positions`
- **`Engine`** — `run_backtest(prices, positions, params) -> BacktestResult`

Everything downstream (portfolio, metrics, UI, AI agent) depends only on these interfaces, never on a
concrete engine. That is what lets a new backend drop in without touching the rest of the app.

**Engine roster:** `PythonEngine` (core) · R analytics/risk layer (core) · `KnimeEngine` (stretch) ·
a future comparison engine (next phase, official follow-on).

## Polyglot interchange contract

Every cross-stage, cross-language hand-off is **Arrow/Parquet** (`src/quantforge/interchange.py`).
Defined once: column names, dtypes, index/timezone conventions for prices, positions, returns, and
metrics. Python writes it; R reads it with `arrow`; KNIME and other tools read it natively. No brittle
in-process language bridge — the file *is* the contract. (This also makes a future MATLAB engine
trivial: MATLAB reads/writes Parquet natively.)

## AI layer

- **`ai/mcp_server.py`** — exposes the pipeline as MCP tools: `load_data`, `run_backtest`,
  `optimize_portfolio`, `get_metrics`. Read-only, input-validated.
- **`ai/nl_interface.py`** — natural language → tool calls → plain-English explanation.
- **`ai/agent.py`** — research loop: propose → backtest → read metrics → refine. Engine-agnostic
  (drives whatever engine is registered, via the tools).
- **`ai/guardrails.py`** — overfitting guardrails (train/val/holdout split + iteration cap) **and**
  public-demo safety (global budget ledger, rate limit, `PUBLIC_MODE`).
- **`ai/budget.py`** — token-spend accounting + hard `AI_BUDGET_USD` kill-switch.

### Overfitting guardrails (the rigor signal)
The data is split train / validation / **untouched holdout**. The agent may iterate on train+val only,
under a hard iteration cap. The holdout is scored **once**, at the end, and never optimized against.
An LLM looping against the test set is p-hacking; these guardrails are how we prevent it.

## Public-demo safety (defense in depth)

1. **Global budget ledger** (daily + total) across all visitors → graceful fallback to cached runs.
2. **AWS Budgets alarm** at the infra layer, independent of app logic.
3. **Gate** live AI behind a passcode; ungated traffic sees cached scenarios only.
4. **Rate-limit** per IP/session.
5. **`PUBLIC_MODE=on` ⇒ parameter-only:** the LLM selects vetted strategies and tunes whitelisted
   params; **no LLM-generated code is executed on the server.** (Codegen research mode = local/dev only.)
6. API key server-side only; MCP tools read-only and validated.

## Deploy

Containerized (Docker) → AWS ECS Fargate via Terraform (`infra/`). Streamlit Community Cloud is the
zero-infra fallback so there's always a live URL.

**Future (next-phase) hosting target:** a MATLAB engine would be hosted MATLAB-natively on **MATLAB
Web App Server** (interactive app, the analog of the Streamlit UI here) and/or **MATLAB Production
Server** (MATLAB compute as a scalable API). This summer's Streamlit + Fargate deploy is the
conceptual rehearsal for that — same shape, open-source tools.

## Directory map

```
src/quantforge/
  interchange.py            # Arrow/Parquet contract (the polyglot backbone)
  data/loader.py            # yfinance → parquet cache; bias handling
  engine/base.py            # Strategy + Engine interfaces (the seam)
  engine/python_engine.py   # custom vectorized backtester
  engine/knime_engine.py    # stretch: headless KNIME workflow as an engine
  strategies/{momentum,mean_reversion,pairs}.py
  portfolio/optimize.py     # PyPortfolioOpt
  metrics/performance.py    # Sharpe, drawdown, CAGR, ...
  ai/{mcp_server,agent,nl_interface,budget,guardrails,sentiment}.py
analytics_r/tearsheet.R     # R analytics/risk (core)
knime/                      # stretch
app/streamlit_app.py        # UI
tests/                      # correctness + rigor + safety
infra/                      # Terraform (Fargate + AWS Budgets alarm)
```
