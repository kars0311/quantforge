# Component 06 — `src/quantforge/metrics/performance.py` (performance metrics)

**Week:** 2 · **Status:** working (vertical slice) · **Depends on:** numpy/pandas only

## Function

Single source of truth for metric definitions. `BacktestResult.metrics`, the UI, the MCP
`get_metrics` tool, and the R cross-check all use these definitions — nobody redefines a metric.

## Requirements satisfied

- **FR-5** — the metric set. **AR-3** — single source of truth. **RG-6** — R must reproduce these
  within ~1% with convention differences explained.

## Interface (implemented — definitions are the contract)

```python
ANN = 252          # trading days/year (annualization factor)
RISK_FREE = 0.0    # simplifying assumption, documented

def compute_metrics(returns: pd.Series) -> dict[str, float]
    # Input: periodic (daily) net returns. NaNs dropped. Empty -> all-NaN dict.
    # Output keys (_KEYS — this exact set is the cross-language contract):
    #   total_return  = prod(1+r) − 1
    #   cagr          = (1+total_return)^(ANN/n) − 1
    #   ann_vol       = std(r, ddof=0) · √ANN          # POPULATION std — see R note
    #   sharpe        = (mean(r) − RISK_FREE/ANN) / std(r, ddof=0) · √ANN
    #   max_drawdown  = min(equity/cummax(equity) − 1)  # ≤ 0
    #   hit_rate      = share of days with r > 0
```

## Cross-language note (feeds the R tearsheet, component 08)

R's `sd()` and PerformanceAnalytics use the **sample** std (ddof=1); this module uses
**population** std (ddof=0). At n ≈ 2500 daily observations the difference is ~0.02% — well
inside the 1% tolerance — but the R script states it explicitly rather than letting the
tolerance silently absorb it (RG-6: "differences explained, not hidden").

## Extensions (optional, only if the UI/writeup wants them — same pattern)

`sortino`, `calmar`, rolling Sharpe. Each addition must be mirrored in `_KEYS`, the R layer,
and the tests in the same change.

## Done when

- A hand-computed fixture test pins each metric's value (guards against silent redefinition);
  R cross-check green within tolerance.
