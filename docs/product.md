# QuantForge — Product Requirements & Component Plan

*Consolidated from `README.md`, `docs/PROJECT_BRIEF.md`, `docs/architecture.md`,
`docs/TEN_WEEK_PLAN.md`, `AGENTS.md`, and the `infra/` / `knime/` READMEs. This is the single
product-planning document; the source docs stay authoritative for narrative and schedule.*

---

## 1. Product vision

An open-source, AI-driven, **polyglot** quant research pipeline with a hosted, budget-capped demo.
The one-sentence end state:

> Type "backtest a momentum strategy on tech stocks, 2015–2020, 10bps costs" — an AI agent parses
> it, runs the pipeline through MCP tools, shows the equity curve, drawdown, Sharpe, and an
> optimized portfolio on the efficient frontier, and explains the result in plain English. A
> "Research mode" lets the agent propose a strategy, backtest it, read its own metrics, and iterate
> — with overfitting guardrails — live, in the cloud.

**Who it's for:**

- **Kent (author):** a portfolio-worthy artifact he can defend line-by-line in interviews.
- **Demo visitors** (recruiters, quants, AEs): a live URL with one-click scenarios and a gated
  live-AI mode.
- **Reviewers of the code:** a repo demonstrating statistical rigor, clean architecture, and
  responsible AI usage.
- **The AI agent itself** (internal user): consumes the pipeline exclusively through MCP tools.

**Non-goals (explicitly out of scope for v1.0):**

- No live/paper trading, broker connectivity, or real money.
- No MATLAB this summer (next-phase add; the architecture just leaves the seam for it).
- No execution of LLM-generated code on the public server, ever.
- No point-in-time universe reconstruction (survivorship bias is documented, not eliminated).

---

## 2. Product requirements

Requirements are numbered so components (§3) can trace to them. **[core]** must ship in v1.0;
**[stretch]** only if ahead of schedule.

### FR — Functional requirements

| ID | Requirement | Scope |
|----|-------------|-------|
| FR-1 | Download and cache daily price data for a **fixed historical universe** of tickers to Parquet (yfinance source). | core |
| FR-2 | Provide a custom **vectorized backtest engine** that turns (prices, positions) into returns, equity curve, and per-period costs. | core |
| FR-3 | Ship two vetted strategies: **momentum** and **mean-reversion**, each emitting positions from prices + params. | core |
| FR-4 | **Portfolio optimization**: mean-variance weights and an efficient frontier via PyPortfolioOpt; combine strategies into a portfolio. | core |
| FR-5 | Compute performance/risk metrics: total return, CAGR, Sharpe, max drawdown (single source of truth in Python). | core |
| FR-6 | **R analytics/risk layer**: a tearsheet built with tidyquant + PerformanceAnalytics from the Parquet hand-off, plus PortfolioAnalytics as a second optimizer cross-checked against Python. | core |
| FR-7 | Expose the pipeline as **MCP tools**: `load_data`, `run_backtest`, `optimize_portfolio`, `get_metrics` — read-only, input-validated. | core |
| FR-8 | **Natural-language interface**: parse a plain-English request into tool calls, run them, and explain the result in plain English. | core |
| FR-9 | **AI research agent**: propose → backtest → read metrics → refine loop, driving the pipeline only through the tools (engine-agnostic). | core |
| FR-10 | **Streamlit UI**: config panel, equity/drawdown/frontier charts (Plotly), AI chat panel, Research mode, engine selector, one-click prebuilt scenarios. | core |
| FR-11 | **Cloud deploy**: Docker container → AWS ECS Fargate via Terraform, with Streamlit Community Cloud as the zero-infra fallback. Live public URL. | core |
| FR-12 | KNIME alternative engine behind the same `Engine` interface, selectable in the UI. | stretch |
| FR-13 | Pairs-trading strategy. | stretch |
| FR-14 | LLM news-sentiment alt-data signal (`ai/sentiment.py`). | stretch |
| FR-15 | `vectorbt` parameter sweeps. | stretch |

### RG — Statistical-rigor requirements (the whole point — never trade these away)

| ID | Requirement |
|----|-------------|
| RG-1 | **No look-ahead bias.** A position decided with data through day *t* earns day *t+1*'s return; the engine enforces this via `positions.shift(1)`. No signal may use same-day data it couldn't have known at decision time. |
| RG-2 | **Transaction costs** modeled on turnover (configurable, e.g. bps per unit traded). |
| RG-3 | **Survivorship bias acknowledged**: fixed historical universe with the caveat documented honestly — not silently ignored. |
| RG-4 | **Out-of-sample discipline**: data split into train / validation / **untouched holdout**. The AI agent sees train+val only; the holdout is scored **once** and never optimized against. |
| RG-5 | **Engine correctness proven**: the custom engine is validated against `backtesting.py` within tolerance (`tests/test_engine_vs_backtestingpy.py`). |
| RG-6 | **Cross-language consistency**: R-computed metrics/optimizer results match the Python source of truth within tolerance. |

### SF — Safety & cost requirements (a public URL with paid AI hooks is a liability)

| ID | Requirement |
|----|-------------|
| SF-1 | Every Claude API call goes through `ai/budget.py`: estimate → `budget.allow()` → call → `budget.charge()`, with hard `AI_BUDGET_USD_DAILY` / `AI_BUDGET_USD_TOTAL` caps and a kill-switch. |
| SF-2 | Global server-side budget ledger shared across all visitors; on exhaustion, graceful fallback to cached scenario runs (never an error page). |
| SF-3 | `PUBLIC_MODE=on` ⇒ **parameter-only**: the LLM may only select vetted strategies and tune whitelisted params. No LLM-generated code executes on the server (codegen research mode is local/dev only). Proven by `tests/test_public_mode_no_codegen.py`. |
| SF-4 | Live AI gated behind `DEMO_PASSCODE`; ungated traffic gets cached scenarios only. |
| SF-5 | Per-IP/session rate limiting. |
| SF-6 | Independent **AWS Budgets alarm** at the infra layer (works even if app logic fails). |
| SF-7 | API key server-side only; MCP tools read-only and input-validated; agent iteration count hard-capped. |
| SF-8 | Holdout isolation is provable: `tests/test_holdout_isolation.py` demonstrates the agent cannot read the holdout. |

### AR — Architecture requirements

| ID | Requirement |
|----|-------------|
| AR-1 | All strategies and engines implement the `Strategy` / `Engine` interfaces in `engine/base.py`. Once mean-reversion lands (week 4), the interfaces are **frozen**. Nothing downstream may depend on a concrete engine. |
| AR-2 | Every cross-stage / cross-language hand-off uses the **Arrow/Parquet interchange contract** (`interchange.py`): column names, dtypes, index/timezone conventions defined once for prices, positions, returns, and metrics. The file *is* the contract — no in-process language bridges. |
| AR-3 | `metrics/performance.py` is the single source of truth for metric definitions; R (and any future engine) must match it, not redefine it. |
| AR-4 | The "next engine drops in here" seam is documented, so a future engine (KNIME now, MATLAB next phase) slots in without touching the rest of the app. |
| AR-5 | Fixed tech stack: Python 3.12 + numpy/pandas, `backtesting.py` (validation only), PyPortfolioOpt, R (tidyquant/PerformanceAnalytics/PortfolioAnalytics), Streamlit + Plotly, `anthropic` + `mcp`, Docker + Terraform/AWS Fargate. Don't swap these. |
| AR-6 | AI model economy: prompt caching on (system prompt, tool schemas, price data); Haiku for NL parsing, Sonnet for agent reasoning, Opus only if clearly needed. |

### QG — Quality & process requirements

| ID | Requirement |
|----|-------------|
| QG-1 | `ruff check .` passes after every code change; `pytest` stays green at every step (enforced by `.claude/` hooks; CI on GitHub). |
| QG-2 | **Defensibility**: docstrings explain *why*, not just *what*; clear conventional code over cleverness; nothing ships that the author can't explain in an interview. |
| QG-3 | Build order follows `docs/TEN_WEEK_PLAN.md`: interchange + loader → engine + momentum → rigor + validation → mean-reversion + freeze → portfolio → R layer → MCP + NL → agent + guardrails → UI → deploy. |
| QG-4 | Match the style and rigor of the existing vertical slice (`metrics/performance.py`, `engine/python_engine.py`, `strategies/momentum.py`, `tests/test_smoke_vertical_slice.py`). |

### DL — Deliverables (definition of done, v1.0)

| ID | Deliverable |
|----|-------------|
| DL-1 | Public GitHub repo, clean README, green CI. |
| DL-2 | Live hosted demo URL with one-click prebuilt scenarios, budget-capped. |
| DL-3 | AI research agent + NL interface working over MCP tools, with documented guardrails. |
| DL-4 | R analytics tearsheet matching Python metrics via the Parquet hand-off. |
| DL-5 | Methodology writeup (assumptions, results, limitations, guardrails, interop) + 60–90s demo video/GIF. |
| DL-6 | Documented "next engine drops in here" seam + architecture diagram; tag `v1.0`. |

---

## 3. Components

Ordered roughly by build dependency (per QG-3). "Status" reflects the scaffold today.
**Each component has a function-level design doc in [`docs/components/`](components/)** (numbered
to match the sections below): planned functions/methods with inputs and outputs, requirement
traceability, and done-when criteria.

### 3.1 `src/quantforge/interchange.py` — Arrow/Parquet interchange contract

- **Function:** Defines and enforces the cross-language data contract: schemas (columns, dtypes,
  index/timezone conventions) for prices, positions, returns, and metrics; read/write helpers so
  Python, R, and KNIME all exchange the same files.
- **Requirements:** AR-2 (the polyglot backbone), AR-4. Schema changes after week 6 are breaking
  changes — treat like a frozen API.
- **Depends on:** nothing (foundation). **Status:** stub. **Week:** 1.

### 3.2 `src/quantforge/data/loader.py` — data ingestion

- **Function:** Downloads daily prices from yfinance for the fixed universe, caches to Parquet via
  the interchange contract, and provides the train/validation/holdout date-split boundaries.
- **Requirements:** FR-1, RG-3 (document the survivorship caveat here), RG-4 (split definitions
  live in one place), AR-2 (output conforms to the contract). Must be deterministic/reproducible
  from cache — the demo cannot depend on yfinance being up.
- **Depends on:** interchange. **Status:** stub. **Week:** 1.

### 3.3 `src/quantforge/engine/base.py` — the seam (interfaces)

- **Function:** Declares `Strategy` (`generate_signals(prices, params) -> positions`), `Engine`
  (`run_backtest(prices, positions, params) -> BacktestResult`), and the `BacktestResult` container.
- **Requirements:** AR-1 (everything downstream depends only on these; **freeze at week 4**),
  AR-4. Interfaces must be engine-agnostic — nothing Python-engine-specific may leak in.
- **Depends on:** nothing. **Status:** working (vertical slice). **Week:** 2–4 (frozen at 4).

### 3.4 `src/quantforge/engine/python_engine.py` — custom vectorized backtester

- **Function:** The core `Engine`: applies `positions.shift(1)` (RG-1), computes per-asset and
  portfolio returns, deducts turnover-based transaction costs (RG-2), and returns equity curve +
  returns in a `BacktestResult`.
- **Requirements:** FR-2, RG-1, RG-2, RG-5 (must match `backtesting.py` within tolerance), AR-1.
- **Depends on:** base, interchange. **Status:** working (vertical slice); costs/validation to
  harden in week 3. **Weeks:** 2–3.

### 3.5 `src/quantforge/strategies/` — momentum, mean_reversion, (pairs)

- **Function:** Each strategy maps prices + whitelisted params to positions, using only information
  available at decision time.
- **Requirements:** FR-3 (momentum core/working, mean-reversion core), FR-13 (pairs, stretch),
  RG-1 (no same-day peeking — the engine's shift is the backstop, not an excuse), AR-1, SF-3
  (strategies are the "vetted" set public mode is limited to; each declares its whitelisted
  param space).
- **Depends on:** base. **Status:** momentum working; mean_reversion/pairs stubs. **Weeks:** 2, 4.

### 3.6 `src/quantforge/metrics/performance.py` — performance metrics

- **Function:** Single source of truth for total return, CAGR, Sharpe, and max drawdown (extensible
  to more), computed from a returns/equity series.
- **Requirements:** FR-5, AR-3, RG-6 (R must reproduce these within tolerance).
- **Depends on:** nothing beyond pandas/numpy. **Status:** working (vertical slice). **Week:** 2.

### 3.7 `src/quantforge/portfolio/optimize.py` — portfolio optimization

- **Function:** PyPortfolioOpt mean-variance optimization: expected returns + covariance → optimal
  weights, plus efficient-frontier points for the UI; combines strategy return streams into a
  portfolio.
- **Requirements:** FR-4, RG-6 (cross-checked against R PortfolioAnalytics), AR-2 (weights/frontier
  emitted via the contract for R and the UI).
- **Depends on:** metrics, interchange. **Status:** stub. **Week:** 5.

### 3.8 `analytics_r/tearsheet.R` — R analytics/risk layer

- **Function:** Reads the Parquet hand-off with `arrow`; produces a tearsheet (returns table,
  drawdowns, risk stats, plots) with tidyquant + PerformanceAnalytics; runs PortfolioAnalytics as a
  second optimizer and reports agreement with the Python results.
- **Requirements:** FR-6, RG-6, AR-2, AR-3 (match Python's metric definitions — don't invent
  variants), DL-4. Runs headless via `Rscript` (no interactive R).
- **Depends on:** interchange, metrics (as the reference), portfolio. **Status:** stub. **Week:** 6.

### 3.9 `src/quantforge/ai/budget.py` — spend accounting

- **Function:** Token/dollar accounting for every Claude call: cost estimation, `allow()` gate
  against daily/total caps, `charge()` ledger, kill-switch. Global (server-wide), persistent across
  requests.
- **Requirements:** SF-1, SF-2. No API call anywhere in the codebase bypasses it.
- **Depends on:** nothing. **Status:** stub. **Week:** 7 (wired from the first AI call, not bolted
  on later).

### 3.10 `src/quantforge/ai/guardrails.py` — overfitting + public-demo guardrails

- **Function:** Two guard families: (a) research rigor — enforces the train/val/holdout split,
  denies any holdout access to the agent, hard iteration cap, one-shot holdout scoring;
  (b) public-demo safety — `PUBLIC_MODE` parameter-only enforcement, passcode gate, rate limiting,
  budget-exhaustion fallback routing.
- **Requirements:** RG-4, SF-2–SF-5, SF-7, SF-8. Provable by `test_holdout_isolation.py` and
  `test_public_mode_no_codegen.py`.
- **Depends on:** budget, loader (split boundaries). **Status:** stub. **Weeks:** 7–9.

### 3.11 `src/quantforge/ai/mcp_server.py` — MCP tool server

- **Function:** Exposes `load_data`, `run_backtest`, `optimize_portfolio`, `get_metrics` as MCP
  tools. Validates all inputs (tickers, date ranges, params) against whitelists; read-only against
  the pipeline; engine-agnostic (drives whatever `Engine` is registered).
- **Requirements:** FR-7, SF-7, AR-1 (talks to interfaces, not concrete engines), AR-4.
- **Depends on:** loader, engines, portfolio, metrics, guardrails. **Status:** stub. **Week:** 7.

### 3.12 `src/quantforge/ai/nl_interface.py` — natural-language interface

- **Function:** Claude tool-use loop: plain-English request → validated tool calls via the MCP
  tools → plain-English explanation of the results.
- **Requirements:** FR-8, SF-1 (every call metered), AR-6 (Haiku for parsing; prompt caching on).
- **Depends on:** mcp_server, budget. **Status:** stub. **Week:** 7.

### 3.13 `src/quantforge/ai/agent.py` — AI research agent (headline feature)

- **Function:** The propose → backtest → read metrics → refine loop over the MCP tools, on
  train/val data only, under iteration and budget caps; produces a final strategy + params scored
  once on the holdout (by the guardrails, not the agent).
- **Requirements:** FR-9, RG-4, SF-1, SF-3, SF-7, SF-8, AR-6 (Sonnet for reasoning), DL-3.
- **Depends on:** mcp_server, guardrails, budget. **Status:** stub. **Week:** 8.

### 3.14 `app/streamlit_app.py` — interactive UI

- **Function:** Streamlit app with: strategy/params/date config panel; Plotly equity curve,
  drawdown, and efficient-frontier charts; metrics display; AI chat panel (NL interface); Research
  mode (agent, gated); engine selector (Python / R / KNIME); one-click prebuilt cached scenarios.
- **Requirements:** FR-10, DL-2, SF-2/SF-4 (renders cached fallback and passcode gate states),
  AR-1 (engine-agnostic via the interfaces).
- **Depends on:** everything upstream. **Status:** stub shell. **Weeks:** 5 (shell) → 9 (complete).

### 3.15 `infra/` — Terraform / AWS deploy

- **Function:** Dockerfile + Terraform for ECR, ECS Fargate service, ALB, and the AWS Budgets cost
  alarm. Streamlit Community Cloud config as fallback.
- **Requirements:** FR-11, SF-6, DL-2. Pre-public checklist: `PUBLIC_MODE=on`, `DEMO_PASSCODE`
  set, `AI_BUDGET_USD_*` set, Budgets alarm confirmed, cached-fallback confirmed.
- **Depends on:** the app. **Status:** stub. **Week:** 9 (mentor-assisted).

### 3.16 `tests/` — correctness, rigor, and safety proof

- **Function:** The proof layer. Key suites: `test_smoke_vertical_slice.py` (working, the pattern),
  `test_engine_vs_backtestingpy.py` (RG-5), cost-accounting + no-look-ahead tests (RG-1/RG-2),
  `test_holdout_isolation.py` (SF-8), `test_public_mode_no_codegen.py` (SF-3), R-vs-Python
  tolerance check (RG-6).
- **Requirements:** QG-1 (green at every step), RG-5, RG-6, SF-3, SF-8.
- **Status:** smoke test green; the rest stubs. **Weeks:** continuous (3, 6, 8, 9 milestones).

### 3.17 `.github/workflows/` — continuous integration

- **Function:** GitHub Actions workflow running `ruff check .` and `pytest` on every push/PR, with a
  status badge in the README. Mirrors the local `.claude/` hooks so quality gates hold even when
  changes land outside a Claude Code session.
- **Requirements:** QG-1 (CI half of the gate), DL-1 (green CI is part of done).
- **Depends on:** tests. **Status:** missing. **Week:** 1.

### 3.18 `.env` / `.env.example` — runtime configuration

- **Function:** The single documented home for runtime settings: `ANTHROPIC_API_KEY` (server-side
  only, never committed — `.env` is gitignored), `AI_BUDGET_USD_DAILY` / `AI_BUDGET_USD_TOTAL`,
  `PUBLIC_MODE`, `DEMO_PASSCODE`. `.env.example` documents every variable with safe placeholders.
- **Requirements:** SF-1, SF-3, SF-4, SF-7. The infra pre-public checklist (§3.15) validates these
  are set before the URL goes live.
- **Depends on:** nothing (consumed by budget/guardrails/infra). **Status:** `.env.example` present,
  key still a placeholder. **Weeks:** 7–9 (vars added as their features land).

### 3.19 `docs/` — methodology writeup + demo assets

- **Function:** The final methodology writeup (assumptions, results, limitations, AI guardrails,
  interop story), the 60–90s demo video/GIF, and the architecture diagram. This is where the
  survivorship caveat and the holdout's one-shot result are stated honestly.
- **Requirements:** DL-5, DL-6, RG-3 (the caveat's documented home).
- **Depends on:** everything (written last). **Status:** missing. **Week:** 10.

### 3.20 Stretch components (build only if ahead — first to drop)

- **`engine/knime_engine.py` + `knime/`** (FR-12): the same backtest as a visual KNIME workflow
  behind the `Engine` interface, invoked headless (batch mode), interoperating via Parquet; metrics
  verified against `PythonEngine`.
- **`strategies/pairs.py`** (FR-13): pairs trading, same `Strategy` interface and rigor bar.
- **`ai/sentiment.py`** (FR-14): LLM news-sentiment signal — still budget-metered (SF-1) and
  vetted/whitelisted like any strategy input (SF-3).
- **vectorbt sweeps** (FR-15): parameter sweeps; results reported on train/val only (RG-4 —
  sweeps are exactly the multiple-comparisons risk the guardrails exist for).

---

## 4. Component dependency graph

```
interchange ──► loader ──────────────┐
     │                               ▼
     │        base (Strategy/Engine) ──► python_engine ──► metrics ──► portfolio
     │              │                        ▲                            │
     │              └── strategies ──────────┘                            ▼
     │                                                            analytics_r (R)
     │
     └──► [everything crossing a stage/language boundary]

budget ──► guardrails ──► mcp_server ──► nl_interface
                              │
                              └────────► agent

app/streamlit_app ◄── all of the above        infra ◄── app
```

---

## 5. Resolved product decisions (settled with Kent, 2026-07-21)

1. **The fixed universe** (FR-1/RG-3): ~28 large-cap names across sectors, heavier tech, daily
   bars, **2010-01-01 → 2026-06-30** (fixed cutoff — never "up to today", so every backtest is
   reproducible from the Parquet cache). Includes ~4–6 **US-listed ADRs** (e.g. TSM, ASML) for
   international exposure while staying on the US trading calendar in USD — no FX or
   calendar-alignment code. Direct foreign listings (e.g. `2330.TW`) are excluded. The exact
   ticker list is finalized in `data/loader.py` in week 1 and then frozen.
2. **Train / validation / holdout split dates** (RG-4): chronological **60/20/20** —
   train 2010-01-01 → 2019-12-31, validation 2020-01-01 → 2022-12-31,
   holdout 2023-01-01 → 2026-06-30. The agent sees train+validation only; the most recent
   ~3.5 years stay untouched until the one-shot holdout scoring.
3. **Default transaction-cost assumption** (RG-2): **10 bps per unit turnover**, user-overridable.
4. **Agent iteration cap** (SF-7): **10 iterations** per research session.
5. **Metric tolerance for cross-checks** (RG-5/RG-6): relative tolerance of **1e-6** for
   engine-vs-backtesting.py where assumptions align exactly; **~1%** for R cross-checks where
   library conventions (annualization, compounding) may differ — with differences explained, not
   hidden.
