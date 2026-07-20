# 10-Week Plan (checklist)

Front-loaded so the credible quant core lands first. Stretch items are the first to drop. It's a
flexible home internship — the timeline can stretch a little rather than cutting the R or AI core.

## Week 1 — Foundations + interchange contract
- [ ] Repo, venv, `requirements.txt`, ruff + pytest green in CI.
- [ ] `data/loader.py`: yfinance → **Parquet** cache; fixed historical universe.
- [ ] **Arrow/Parquet interchange schema** (`interchange.py`) — the polyglot backbone.
- [ ] README skeleton; plot a price series end-to-end.

## Week 2 — Engine + strategy #1
- [ ] Custom **vectorized backtester** (`engine/python_engine.py`).
- [ ] **Momentum** strategy (`strategies/momentum.py`).
- [ ] Metrics (`metrics/performance.py`): total/CAGR, Sharpe, max drawdown.

## Week 3 — Rigor pass
- [ ] Transaction costs; explicit **look-ahead** and **survivorship** handling (documented).
- [ ] **Validate the engine vs `backtesting.py`** on momentum (`tests/test_engine_vs_backtestingpy.py`).
- [ ] pytest suite covering cost accounting + no-look-ahead.

## Week 4 — Strategy #2 + lock the seam
- [ ] **Mean-reversion** strategy.
- [ ] Refactor strategies + engine behind the `Strategy`/`Engine` interfaces (`engine/base.py`). Freeze the interface.

## Week 5 — Portfolio layer + UI shell
- [ ] PyPortfolioOpt mean-variance + efficient frontier (`portfolio/optimize.py`).
- [ ] Combine strategies into a portfolio.
- [ ] Stand up the Streamlit shell + core charts (build the UI incrementally from here).

## Week 6 — R analytics/risk layer (core; first polyglot interop)
- [ ] `analytics_r/tearsheet.R`: read the Parquet hand-off; tidyquant + PerformanceAnalytics tearsheet.
- [ ] PortfolioAnalytics as a second optimizer; cross-check vs Python (within tolerance).
- [ ] (Pressure-test the stats methodology with the rigor advisor.)

## Week 7 — MCP + natural-language interface
- [ ] `ai/mcp_server.py`: expose `load_data` / `run_backtest` / `optimize_portfolio` / `get_metrics`.
- [ ] `ai/nl_interface.py`: Claude tool-use parses a request → runs → explains in English.
- [ ] Wire `ai/budget.py` caps + `ai/guardrails.py` from day one (don't bolt on later).

## Week 8 — AI research agent (headline)
- [ ] `ai/agent.py`: propose → backtest → read metrics → refine loop.
- [ ] **Overfitting guardrails:** train / validation / **untouched holdout**, iteration cap, budget cap.
- [ ] `tests/test_holdout_isolation.py`: prove the agent cannot read the holdout.

## Week 9 — Finish UI + deploy + harden (mentor-assisted cloud)
- [ ] AI chat panel + "Research mode" + **engine selector (Python / R / KNIME)**.
- [ ] Containerize; Terraform → AWS ECS Fargate; live URL (Streamlit Community Cloud fallback).
- [ ] **`PUBLIC_MODE=on`:** global budget ledger, passcode gate, rate limits, **AWS Budgets alarm**,
      no codegen exec (`tests/test_public_mode_no_codegen.py`).

## Week 10 — Polish + writeup + buffer
- [ ] Demo GIF/video; one-click prebuilt scenarios.
- [ ] Methodology writeup (assumptions, results, **limitations + AI guardrails + interop**).
- [ ] Architecture diagram; tag **v1.0**; rehearse the demo narration.

---

### Stretch (only if ahead)
- [ ] KNIME alt-engine (`engine/knime_engine.py` + `knime/`).
- [ ] Pairs trading strategy.
- [ ] LLM news-sentiment alt-data signal (`ai/sentiment.py`).
- [ ] `vectorbt` parameter sweeps.
