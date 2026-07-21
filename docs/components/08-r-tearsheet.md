# Component 08 — `analytics_r/tearsheet.R` (R analytics/risk layer)

**Week:** 6 · **Status:** stub · **Depends on:** interchange (files), metrics (as reference), portfolio

## Function

The first concrete polyglot interop. A headless `Rscript` that reads the Parquet hand-off,
produces an institutional-style tearsheet (tables + plots), recomputes the headline metrics with
PerformanceAnalytics, runs PortfolioAnalytics as an independent second optimizer, and writes both
back as Parquet so Python tests can assert agreement.

## Requirements satisfied

- **FR-6** — the R tearsheet + second optimizer. **RG-6** — metrics/optimizer agreement within
  ~1%, conventions explained. **AR-2** — all I/O via the interchange files. **AR-3** — R matches
  Python's metric *definitions*, it does not invent variants. **DL-4** — a deliverable.

## Contract (CLI, headless — no interactive R)

```
Rscript analytics_r/tearsheet.R [data_dir=data_cache] [out_dir=analytics_r/output]

reads:   data_dir/returns.parquet         (portfolio returns — interchange `returns` kind)
         data_dir/asset_returns.parquet   (per-asset — for the optimizer cross-check)
writes:  out_dir/tearsheet.png            (charts.PerformanceSummary: cumulative return,
                                           daily returns, drawdown — the visual deliverable)
         data_dir/metrics_r.parquet       (interchange `metrics` kind, same _KEYS names)
         data_dir/weights_r.parquet       (interchange `weights` kind, PortfolioAnalytics optimum)
exit:    0 on success, non-zero + message on any failure (so pytest can shell out to it)
```

## Internal structure (integral steps, mirroring the metric contract)

1. `arrow::read_parquet` → build an `xts` series (dates as index — R's time-series convention).
2. Tearsheet: `PerformanceAnalytics::charts.PerformanceSummary`, `table.AnnualizedReturns`,
   `table.Drawdowns`.
3. `compute_metrics_r(returns) -> data.frame(name, value)` — reimplements the six `_KEYS` with
   Python's exact conventions: ANN=252, risk-free 0, **population** std (`sd(r)*sqrt((n-1)/n)`,
   since R's `sd` is sample-std) — the divergence is stated in a comment, not absorbed silently.
4. `PortfolioAnalytics` mean-variance optimum on asset returns (long-only, full investment —
   the same constraints as component 07) → weights.
5. `arrow::write_parquet` both outputs.

## Design notes

- Kent is new to R: the script is one linear file, heavily commented, no custom S4/classes —
  each step maps 1:1 to a Python concept already built.
- The Python side of the cross-check lives in `tests/` (runs `Rscript` via subprocess,
  `skipif` Rscript or the R packages are unavailable, e.g. in CI).

## Done when

- Runs headless from a fresh `data_cache` produced by the Python pipeline; pytest cross-check
  green: metrics within ~1%, optimizer weights/vol within ~1%, differences documented.
