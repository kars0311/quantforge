# Component 07 — `src/quantforge/portfolio/optimize.py` (portfolio optimization)

**Week:** 5 · **Status:** stub · **Depends on:** metrics, interchange

## Function

PyPortfolioOpt mean-variance layer: turn a panel of return streams (individual assets *or*
strategy return streams — the math is identical) into optimal weights and efficient-frontier
points, and combine weighted streams into one portfolio return series.

## Requirements satisfied

- **FR-4** — mean-variance weights + frontier; strategy combination.
- **RG-6** — independently cross-checked by R PortfolioAnalytics within ~1%.
- **AR-2** — weights and frontier are emitted via the interchange `weights`/`frontier` kinds for
  the UI and the R layer.

## Interface (integral functions)

```python
def optimize_weights(returns: pd.DataFrame,          # WIDE: index=date, columns=asset-or-strategy
                     params: dict | None = None) -> pd.Series
    # params: {"objective": "max_sharpe" | "min_volatility" = "max_sharpe",
    #          "weight_bounds": (0.0, 1.0)}           # long-only at the portfolio level
    # Wraps pypfopt: expected_returns.mean_historical_return + risk_models.sample_cov ->
    # EfficientFrontier -> clean_weights(). Returns weights indexed by column name, Σw = 1.

def frontier(returns: pd.DataFrame, n_points: int = 50) -> pd.DataFrame
    # Sweeps target returns between the min-vol and max-return portfolios; solves min-vol at
    # each. Returns DataFrame (risk, ret) — annualized vol and annualized return — conforming
    # to the interchange `frontier` kind, ready for the UI scatter + the max-Sharpe marker.

def combine_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series
    # Fixed-weight combination: (returns[weights.index] * weights).sum(axis=1).
    # Used to blend momentum + mean-reversion streams into "the portfolio" (FR-4).
```

## Design notes

- Estimator assumptions are deliberately the textbook defaults (historical mean, sample
  covariance) — defensible and matching what R PortfolioAnalytics will do; fancier estimators
  (shrinkage) are out of scope and would complicate the cross-check.
- Portfolio-level weights stay long-only (bounds (0,1)) even though the mean-reversion *strategy*
  may short internally — the optimizer allocates capital across strategies/assets, it does not
  create shorts.

## Done when

- Weights sum to 1 within 1e-8, respect bounds; frontier risk column is monotonically
  increasing with ret after the min-vol point; R cross-check within ~1% (weights and achieved
  vol); results written via `write_frame` and rendered by the UI.
