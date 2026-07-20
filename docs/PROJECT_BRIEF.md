# Project Brief — Day 1

## What you're building

An open-source, **AI-driven quant research pipeline** with a snazzy interactive UI and a
cloud-hosted, budget-capped demo. The end state, in one sentence:

> *Type "backtest a momentum strategy on tech stocks, 2015–2020, 10bps costs" — an AI agent parses
> it, runs the pipeline through tools, shows the equity curve, drawdown, Sharpe and an optimized
> portfolio on the efficient frontier, and explains the result in plain English. A "Research mode"
> lets the agent propose a strategy, backtest it, read its own metrics, and iterate — with
> overfitting guardrails — live, in the cloud.*

## Why it's built the way it is (the things that make it credible)

1. **Rigor over flash.** A naive backtest impresses no one who knows quant. What signals competence:
   - No **look-ahead bias** (signals use only information available at decision time).
   - No **survivorship bias** (a fixed historical universe; document the caveat honestly).
   - **Transaction costs** modeled.
   - **Out-of-sample** evaluation — an untouched holdout you never tune against.
   - A custom engine you wrote yourself, **validated against `backtesting.py`** so you can prove it's correct.
2. **One interface, many engines.** Strategies and the backtest engine sit behind a thin interface
   (`src/quantforge/engine/base.py`). That's what lets R, KNIME, or another engine slot in later
   without rewriting the app.
3. **Polyglot by design.** Stages exchange data as **Arrow/Parquet**, so Python does the engine, **R
   does the analytics/risk layer**, and each tool is used where it's best. "Right tool for the right
   piece, and back again."
4. **AI done responsibly.** The agent is genuinely useful *and* the project shows you understand its
   failure mode: an LLM iterating against the same test set is p-hacking. The guardrails (train/val/
   holdout, iteration cap) are the point — be ready to explain them.

## Scope: core vs stretch

**Core (must ship):** data + Parquet interchange → custom vectorized engine → 2 strategies
(momentum, mean-reversion) → portfolio optimization (PyPortfolioOpt) → metrics → **R analytics/risk
layer** → MCP tools → AI research agent + natural-language interface → Streamlit UI → publicly-safe,
budget-capped cloud deploy.

**Stretch (only if ahead):** KNIME alt-engine, pairs trading, LLM news-sentiment alt-data signal,
`vectorbt` parameter sweeps.

## Hard requirements (don't skip)

- **Budget caps on all AI calls** (`AI_BUDGET_USD_DAILY` / `_TOTAL`) with a kill-switch.
- **Public demo safety:** `PUBLIC_MODE=on` ⇒ parameter-only (no execution of LLM-generated code),
  passcode-gated live AI, rate limits, cached fallback, and an independent AWS Budgets alarm.
- **Defensibility:** if you didn't write it or can't explain it, it doesn't go in. (Use Claude Code to
  learn and accelerate, not to ship code you can't defend in an interview.)

## Who helps with what

- **Cloud / infra / deploy:** mentor-supported (the AWS + Terraform + container work).
- **Statistical rigor:** there's a stats-savvy family advisor available to pressure-test the
  methodology (multiple-comparisons / overfitting). Lean on them for the guardrails and the R layer.
- **Everything else (the build):** you.

## Definition of done (v1.0)

- Public GitHub repo, clean README, green CI.
- Live hosted demo URL with one-click prebuilt scenarios, budget-capped.
- AI research agent + NL interface working over MCP tools, with documented guardrails.
- R analytics tearsheet matching Python metrics via the Parquet hand-off.
- A short methodology writeup (assumptions, results, limitations, the guardrails) + a 60–90s demo video.
- The documented "next engine drops in here" seam.

Start with [`TEN_WEEK_PLAN.md`](TEN_WEEK_PLAN.md).
