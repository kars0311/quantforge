# Component 04 — `src/quantforge/engine/python_engine.py` (custom vectorized backtester)

**Weeks:** 2–3 · **Status:** working (vertical slice); harden in week 3 · **Depends on:** base, metrics

## Function

The core `Engine`: turns (prices, target weights) into net returns, an equity curve, and metrics —
fully vectorized, no per-row Python loops. The heart of the project and the strongest rigor
signal; every line must be explainable in an interview.

## Requirements satisfied

- **FR-2** — the custom vectorized engine.
- **RG-1** — `held = positions.shift(1)`: a weight decided at close of t earns t+1's return.
- **RG-2** — costs = `turnover × cost_bps/10_000`, turnover = Σ|Δweight| per day.
- **RG-5** — validated against `backtesting.py` within 1e-6 relative tolerance (see tests doc).
- **AR-1** — implements the frozen `Engine` interface.

## Interface (implemented — the algorithm, step by step)

```python
class PythonEngine(Engine):
    name = "python"
    def run_backtest(self, prices, positions, params: dict | None = None) -> BacktestResult
        # params: {"cost_bps": float = 0.0}
        # 1. sort prices by date; align positions to prices' index/columns (missing -> 0.0)
        # 2. asset_returns = prices.pct_change()          # day-over-day simple returns
        # 3. held = positions.shift(1)                    # THE no-look-ahead line (RG-1)
        # 4. gross = (held * asset_returns).sum(axis=1)   # portfolio return before costs
        # 5. turnover = held.diff().abs().sum(axis=1)     # total weight traded each day
        # 6. net = gross - turnover * cost_bps / 10_000   # RG-2
        # 7. equity = (1 + net).cumprod(); metrics = compute_metrics(net)
```

Long-short works with zero changes: negative weights flip the sign of the earned return in
step 4, and turnover in step 5 already charges |Δw| symmetrically for buys, sells, and shorts.
(Borrow fees for shorts are NOT modeled — documented limitation, see strategies doc.)

## Week-3 hardening (planned, still integral)

- Input validation: raise `ValueError` on non-monotonic index, non-numeric data, or gross
  exposure Σ|w| > 1 + ε in `positions` (catch buggy strategies at the boundary).
- NaN policy made explicit: NaN prices (pre-IPO) → 0 return contribution; document why.
- `meta` enriched: engine name, cost_bps, params, start/end, n_days, total turnover.

## Validation strategy vs `backtesting.py` (RG-5)

`backtesting.py` is an event-driven, **single-asset** framework — so the comparison runs a
single-ticker long/flat momentum rule through both engines on the same series (cost-free and
with costs), asserting final equity and Sharpe agree within 1e-6 relative where assumptions
align exactly. Multi-asset correctness is then covered by hand-computed pytest cases (a 2-asset,
5-day example checked by hand in the test).

## Done when

- `test_engine_vs_backtestingpy.py` un-skipped and green; hand-computed cost/no-look-ahead tests
  green; input validation raises on malformed inputs.
