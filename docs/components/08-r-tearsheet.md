# Component 08 — `analytics_r/tearsheet.R` (R analytics/risk layer)

**Week:** 6 · **Status:** built (wk 6) — cross-check suites green · **Depends on:** interchange (files), metrics (as reference), portfolio

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

## Cross-language conventions and caveats

The week-6 methodology pressure-test notes (this section satisfies the plan's "pressure-test
the stats methodology" item). The rule, per `docs/product.md` decision 5: cross-language
**differences are explained, not hidden**. Every convention below is either transcribed exactly
into R or, where a library insists on its own convention, quarantined to display output.

- **Population vs sample standard deviation.** Python's `numpy.std` defaults to the
  *population* std (ddof=0); R's `sd()` is the *sample* std (ddof=1). `compute_metrics_r`
  corrects with `sd(r) * sqrt((n-1)/n)` so volatility and Sharpe match Python's definition
  exactly. The divergence is stated in a comment in `tearsheet.R`, not absorbed silently. At
  n≈250 the two differ by ~0.2% — small enough to hide inside a loose tolerance, which is
  precisely why the transcription is exact instead. Edge case verified both sides: at n=1,
  R's `sd()` is `NA` where numpy's ddof=0 gives 0; the R code branches to match Python.
- **Annualization: ANN=252, observation-count CAGR.** Both sides annualize with 252 trading
  days (vol × √252, Sharpe × √252) and compute CAGR from the **observation count** —
  `(1+total)^(252/n) − 1` — not from calendar elapsed time. On a daily series with no gaps the
  two agree; on a sparse or truncated series calendar-time CAGR would differ. Observation-count
  is the convention `metrics/performance.py` (the single source of truth) uses, so R transcribes
  it.
- **Risk-free rate = 0.** Sharpe is excess return over a zero risk-free rate on both sides — a
  deliberate simplification (documented, defensible, and constant across languages so it cannot
  cause drift). PerformanceAnalytics functions that accept `Rf` are not given a different value.
- **Why the optimizer cross-check objective is min-variance, not max-Sharpe.** Long-only
  full-investment minimum variance is a **convex QP with a unique global optimum** — two
  independent implementations (PyPortfolioOpt and PortfolioAnalytics/ROI/quadprog) *must* land
  on the same weights, so a disagreement is a real bug. Max-Sharpe involves search internals
  (and, in pypfopt, a transformation) that can legitimately differ between libraries without
  either being wrong — it would test implementation trivia, not correctness. Annualization
  cannot skew the check either: Python minimizes over `sample_cov × 252`, R over the daily
  sample covariance, and scaling by a positive constant never moves the argmin.
- **Why the tolerance is ~1% — and why the observed agreement is ~1e-9.** The RG-6 contract
  tolerance (~1%) exists for *library-convention drift*: had we compared Python metrics to
  PerformanceAnalytics' own functions (sample std, geometric mean annualization), differences
  up to that order would be legitimate. But `compute_metrics_r` **transcribes** Python's
  formulas rather than calling a library, so `tests/test_r_cross_check.py` asserts both bounds:
  the ~1% contract *and* ~1e-9 actual agreement (weights agree to ~5e-6, the residual being
  pypfopt's `clean_weights` 5-dp rounding). Asserting only the loose bound would let a
  transcription bug (e.g. dropping the population-std correction, a ~1e-4 error) pass silently.
- **PerformanceAnalytics display tables keep their own conventions (AR-3 note).**
  `table.AnnualizedReturns` / `table.Drawdowns` printed to stdout use PerformanceAnalytics'
  native conventions (sample std, geometric annualization) and are **display-only** — their
  numbers may differ from ours in the third decimal and that is expected. Nothing from those
  tables crosses the Parquet boundary; the interchange `metrics_r.parquet` carries only
  `compute_metrics_r`'s exact transcriptions, which is what keeps AR-3 ("R matches Python's
  metric *definitions*, it does not invent variants") intact.

## Done when

- Runs headless from a fresh `data_cache` produced by the Python pipeline; pytest cross-check
  green: metrics within ~1%, optimizer weights/vol within ~1%, differences documented.

**Status vs done-when (week 6): met.** `tests/test_r_cross_check.py` runs the real Python
pipeline (both vetted strategies → engine → panel → optimizer), hands off via the interchange
files, shells out to `Rscript tearsheet.R`, and asserts metric agreement at both ~1% and ~1e-9
and weight agreement within 1%/achieved-vol 1% (see the conventions section above). Supporting
suites: `test_r_interchange.py` (Python → R → Python Parquet round trip) and the verifier
suites in `tests/`. All R-dependent tests `skipif` when Rscript or the R packages are absent
(e.g. bare CI) — the cross-check is machine-local by nature.
