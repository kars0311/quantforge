# Component 20 — Stretch components (only if ahead of schedule; first to drop)

**Status:** deliberately unplanned in detail — this doc records only the **interface
obligations** each stretch item must honor, so core designs never close a door on them.
Full design docs get written if/when an item is actually started.

## Interface obligations

- **KNIME engine** (`engine/knime_engine.py` + `knime/`, FR-12): implements `Engine` exactly —
  `run_backtest(prices, positions, params) -> BacktestResult`. Adapter shape: write
  prices/positions via `interchange.write_frame` → invoke the `.knwf` workflow headless (KNIME
  batch CLI) → read returns back via `read_frame` → wrap in `BacktestResult` with
  `meta["engine"]="knime"`. Registered in `mcp_server.ENGINES` — nothing else changes (that's
  the AR-4 proof). Must match `PythonEngine` metrics within documented tolerance.
- **Pairs trading** (`strategies/pairs.py`, FR-13): implements `Strategy`; params enter
  `PARAM_WHITELIST` + `STRATEGIES` like any vetted strategy; same no-look-ahead bar (spread
  z-score on backward-looking windows). Long-short by nature → inherits the gross-exposure ≤ 1
  convention and the borrow-cost caveat from mean-reversion.
- **LLM news sentiment** (`ai/sentiment.py`, FR-14): every call through `budget.allow/charge`
  (SF-1); output is a *signal input* to vetted strategies — it never executes code and its
  params are whitelisted like any other (SF-3).
- **vectorbt sweeps** (FR-15): reported on train/validation ONLY — a parameter sweep is exactly
  the multiple-comparisons risk RG-4 exists for; sweep results never touch the holdout.

## Done when (per item, if built)

The item's tests reach the same bar as core (correctness vs reference, whitelist enforcement,
budget metering), and a full design doc is added here first.
