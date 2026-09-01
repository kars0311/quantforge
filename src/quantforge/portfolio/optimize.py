"""Portfolio optimization via PyPortfolioOpt (week 5).

Turns a WIDE panel of periodic (daily) simple returns — columns are assets *or* strategy return
streams, the math is identical — into mean-variance optimal weights, and blends weighted streams
into a single portfolio return series (FR-4).

Why the textbook estimators (and nothing fancier):

- ``expected_returns.mean_historical_return`` (historical mean, annualized at 252) and
  ``risk_models.sample_cov`` (sample covariance, annualized at 252) are the plain-vanilla
  estimators every reference derivation of mean-variance optimization assumes. They are what the
  R PortfolioAnalytics layer computes in week 6, so the cross-check (RG-6, within ~1%) compares
  like with like. Shrinkage estimators (Ledoit-Wolf etc.) would trade a little estimation error
  for a cross-check headache and a harder interview defense — deliberately out of scope.
- Weights are LONG-ONLY at the portfolio level (default bounds (0, 1)) even though a strategy in
  the panel may short internally: the optimizer allocates *capital across streams*, it does not
  create shorts. Allowing negative portfolio weights would let the optimizer bet against a
  strategy, which is a different (and harder to defend) product.

``frontier`` sweeps target returns from the min-volatility portfolio up to (just below) the max
attainable expected return, solving a fresh minimum-variance problem at each target, and emits the
(risk, ret) points in the interchange ``frontier`` schema (AR-2) for the UI scatter and the R layer.
Both it and ``optimize_weights`` estimate (mu, S) through the same private helper so the frontier
can never be drawn from different estimators than the weights plotted on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, exceptions, expected_returns, risk_models

from quantforge.interchange import validate_frame

# The only params optimize_weights understands. Unknown keys are rejected loudly rather than
# ignored — same whitelist spirit as the vetted strategy-params gate (SF-3): a typo like
# "objectve" must fail, not silently fall back to the default.
_ALLOWED_PARAM_KEYS = {"objective", "weight_bounds"}
_OBJECTIVES = {"max_sharpe", "min_volatility"}
_DEFAULT_BOUNDS = (0.0, 1.0)

# Relative margin by which the frontier sweep stops short of max(mu). The exact max-return
# portfolio sits on a corner of the feasible set (all capital in the single best asset), where
# pypfopt's own target check and the QP solver both get flaky; stopping 1e-6 (relative) below it
# is visually indistinguishable on any plot while keeping every sweep point reliably solvable.
_FRONTIER_TOP_MARGIN = 1e-6


def _estimate_moments(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualized (mu, S) via the textbook estimators — the ONE place they are chosen.

    ``optimize_weights`` and ``frontier`` must both call this helper: if each picked its own
    estimators they could silently drift apart, and a frontier drawn from different (mu, S)
    than the weights plotted on it would be meaningless. Historical mean + sample covariance
    (252-day annualization) are deliberate — see the module docstring for the R cross-check
    rationale.

    The input is deep-copied before pypfopt sees it: pypfopt is not documented to mutate its
    inputs, but our no-mutation guarantee to callers should not depend on a third-party
    implementation detail.
    """
    panel = returns.copy(deep=True)
    mu = expected_returns.mean_historical_return(panel, returns_data=True)
    cov = risk_models.sample_cov(panel, returns_data=True)
    return mu, cov


def _validate_returns_panel(returns: pd.DataFrame) -> None:
    """Reject inputs the optimizer would mishandle silently.

    Each check names the offending input so a failure is diagnosable from the message alone.
    NaNs are rejected rather than dropped/filled because how to treat a missing return (asset
    not yet listed? bad data?) is a data-loading decision, not an optimizer default.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError(
            f"returns must be a pandas DataFrame (wide panel), got {type(returns).__name__}"
        )
    if returns.shape[1] < 2:
        raise ValueError(
            f"returns must have at least 2 columns to optimize over, got {returns.shape[1]}"
        )
    dupes = returns.columns[returns.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"returns has duplicated column labels: {dupes}")
    if returns.isna().any().any():
        bad = returns.columns[returns.isna().any()].tolist()
        raise ValueError(f"returns contains NaN values in columns: {bad}")
    if len(returns) < 2:
        raise ValueError(
            f"returns needs at least 2 rows to estimate a covariance, got {len(returns)}"
        )


def _validate_params(
    params: dict[str, Any] | None, n_assets: int
) -> tuple[str, tuple[float, float]]:
    """Resolve (objective, weight_bounds) from params, rejecting anything off-whitelist."""
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"params must be a dict or None, got {type(params).__name__}")

    unknown = set(params) - _ALLOWED_PARAM_KEYS
    if unknown:
        raise ValueError(
            f"unknown params keys: {sorted(unknown)}; allowed: {sorted(_ALLOWED_PARAM_KEYS)}"
        )

    objective = params.get("objective", "max_sharpe")
    if objective not in _OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; allowed: {sorted(_OBJECTIVES)}")

    bounds = params.get("weight_bounds", _DEFAULT_BOUNDS)
    if (
        not isinstance(bounds, (tuple, list))
        or len(bounds) != 2
        or not all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in bounds)
    ):
        raise ValueError(f"weight_bounds must be a (low, high) pair of numbers, got {bounds!r}")
    lo, hi = float(bounds[0]), float(bounds[1])
    if not lo < hi:
        raise ValueError(f"weight_bounds low must be < high, got weight_bounds=({lo}, {hi})")
    # Long-only at the portfolio level by design (see module docstring / component doc 07):
    # the optimizer allocates capital across streams, it never bets against one.
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"weight_bounds must lie within the long-only range [0, 1], got ({lo}, {hi})"
        )
    # Fully-invested feasibility: with n assets each in [lo, hi], the sum can only reach 1 if
    # n*lo <= 1 <= n*hi. Rejecting here gives a named error instead of an opaque solver failure.
    if n_assets * lo > 1.0 or n_assets * hi < 1.0:
        raise ValueError(
            f"weight_bounds ({lo}, {hi}) make sum(weights)=1 infeasible for "
            f"{n_assets} assets (need n*low <= 1 <= n*high)"
        )
    return objective, (lo, hi)


def optimize_weights(returns: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    """Mean-variance optimal weights for a wide panel of periodic returns.

    Parameters
    ----------
    returns : pd.DataFrame
        WIDE panel of periodic (daily) simple returns: index = DatetimeIndex,
        columns = asset-or-strategy labels. No NaNs, no duplicate labels.
    params : dict, optional
        ``{"objective": "max_sharpe" | "min_volatility"}`` (default ``"max_sharpe"``) and
        ``{"weight_bounds": (low, high)}`` (default ``(0.0, 1.0)``, long-only). Any other key
        raises ``ValueError`` — whitelist, not best-effort.

    Returns
    -------
    pd.Series
        Weights indexed by column label, in the input's column order, summing to 1.

    Why it is built this way
    ------------------------
    - Estimators are the textbook defaults (historical mean + sample covariance, both annualized
      at 252 by pypfopt) so the week-6 R PortfolioAnalytics cross-check compares like with like;
      see the module docstring for the full rationale.
    - ``clean_weights()`` rounds to 5 decimals and zeroes dust below 1e-4 — the raw solver output
      carries meaningless 1e-9 residues that would differ between solvers and confuse the R
      cross-check. Rounding can leave the sum a few 1e-4 off 1, so we renormalize by the sum:
      for non-negative weights ``w / w.sum()`` keeps every weight inside [0, 1] (no weight can
      exceed the sum) while restoring the fully-invested budget exactly.
    - Neither ``returns`` nor ``params`` is mutated; pypfopt gets its own copies.
    """
    _validate_returns_panel(returns)
    objective, bounds = _validate_params(params, n_assets=returns.shape[1])

    mu, cov = _estimate_moments(returns)  # shared with frontier(); 252-day annualized

    ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
    if objective == "max_sharpe":
        ef.max_sharpe()
    else:
        ef.min_volatility()

    cleaned = ef.clean_weights()  # OrderedDict {label: weight}, rounded to 5 dp, dust zeroed
    weights = pd.Series(cleaned, dtype=float).reindex(returns.columns)
    total = float(weights.sum())
    if total <= 0.0:  # cannot happen for a solved long-only problem; guard the division anyway
        raise ValueError(f"optimizer produced non-positive total weight {total}")
    weights = weights / total
    weights.name = "weight"
    return weights


def frontier(returns: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
    """Efficient-frontier sweep: ``n_points`` (risk, ret) portfolios from min-vol to max-return.

    Parameters
    ----------
    returns : pd.DataFrame
        WIDE panel of periodic (daily) simple returns, same contract as ``optimize_weights``.
    n_points : int
        Number of target returns to sweep (default 50). Must be >= 2 — one point is not a
        frontier — else ``ValueError``.

    Returns
    -------
    pd.DataFrame
        Exactly the interchange ``frontier`` schema: columns ``(risk, ret)``, float64,
        annualized volatility and annualized expected return, sorted by ``ret`` ascending,
        all values finite. ``validate_frame(df, "frontier")`` passes by construction, so the
        result can go straight to ``interchange.write_frame`` for the UI / R layer (AR-2).

    Why it is built this way
    ------------------------
    - (mu, S) come from ``_estimate_moments`` — the *same* helper ``optimize_weights`` uses —
      so the frontier and any weights plotted on it are always drawn from identical estimators.
    - Targets sweep linearly from the min-volatility portfolio's expected return (the leftmost
      point of the efficient frontier — anything below it is the inefficient lower branch) up
      to just below max(mu), the best attainable expected return under long-only fully-invested
      bounds (all capital in the single best asset). The top is clipped by a 1e-6 relative
      margin because the exact corner sits on the boundary of the feasible set where the QP
      solver is unreliable; see ``_FRONTIER_TOP_MARGIN``.
    - Each target gets a FRESH ``EfficientFrontier``: pypfopt optimizer objects are effectively
      single-solve (they accumulate objective/constraint state), so reusing one across targets
      is exactly the kind of subtle misuse this module avoids.
    - A target the solver still cannot hit is dropped rather than crashing the whole sweep —
      one missing dot on a 50-point scatter is harmless — but a sweep that loses more than two
      points means (mu, S) are pathological and the caller must hear about it, so fewer than
      ``max(2, n_points - 2)`` solved points raises ``ValueError``.
    """
    _validate_returns_panel(returns)
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError(f"n_points must be an int, got {type(n_points).__name__}")
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2 to trace a frontier, got {n_points}")

    mu, cov = _estimate_moments(returns)

    # Bottom of the sweep: the min-volatility portfolio's expected return. Solved with the same
    # default long-only bounds the sweep uses, so the frontier's first point IS the min-vol
    # portfolio that optimize_weights(objective="min_volatility") returns.
    ef_min = EfficientFrontier(mu, cov, weight_bounds=_DEFAULT_BOUNDS)
    ef_min.min_volatility()
    ret_min, _, _ = ef_min.portfolio_performance()

    max_mu = float(mu.max())
    # max(|max_mu|, 1) keeps the margin meaningful even when max_mu is tiny or zero, where a
    # purely relative margin would vanish and leave the top target on the unsolvable corner.
    top = max_mu - _FRONTIER_TOP_MARGIN * max(abs(max_mu), 1.0)
    # Degenerate panel (all assets share one expected return): collapse the sweep onto the
    # min-vol point instead of sweeping backwards.
    top = max(top, float(ret_min))

    points: list[tuple[float, float]] = []
    for target in np.linspace(ret_min, top, n_points):
        ef = EfficientFrontier(mu, cov, weight_bounds=_DEFAULT_BOUNDS)  # fresh per solve
        try:
            ef.efficient_return(float(target))
        except (ValueError, exceptions.OptimizationError):
            continue  # unreachable corner target: drop the point, keep the sweep
        achieved_ret, achieved_risk, _ = ef.portfolio_performance()
        points.append((float(achieved_risk), float(achieved_ret)))

    min_rows = max(2, n_points - 2)
    if len(points) < min_rows:
        raise ValueError(
            f"frontier sweep solved only {len(points)} of {n_points} targets "
            f"(need at least {min_rows}); the return panel's estimated moments are "
            f"likely degenerate"
        )

    frame = pd.DataFrame(points, columns=["risk", "ret"]).astype("float64")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("frontier sweep produced non-finite risk/ret values")
    # Achieved returns already rise with the targets, but sort explicitly so the ordering is a
    # guarantee of this function, not a side effect of solver behavior.
    frame = frame.sort_values("ret", kind="stable", ignore_index=True)

    validate_frame(frame, "frontier")  # self-check the AR-2 contract before handing it out
    return frame


def combine_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Fixed-weight blend of return streams into one portfolio return series.

    ``(returns[weights.index] * weights).sum(axis=1)`` — alignment is BY LABEL, never by
    position, so the caller cannot silently pair a weight with the wrong stream.

    Why fixed weights: this blends whole strategy return streams (momentum + mean-reversion)
    into "the portfolio" (FR-4). Rebalancing/drift modeling lives inside the strategies and the
    engine, not here — a per-period fixed-weight combination is the standard, defensible reading
    of "portfolio return given weights" and matches what the R layer computes.

    Why no silent renormalization: weights that do not sum to 1 (within 1e-8) mean the caller's
    budget is wrong — leveraged, under-invested, or simply buggy. Fixing that quietly would hide
    the bug and desynchronize us from the R cross-check, so it raises instead.

    Raises ``ValueError`` if a weight label is missing from ``returns``, labels are duplicated,
    weights or the used return columns contain NaN, or the weights do not sum to 1.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError(
            f"returns must be a pandas DataFrame (wide panel), got {type(returns).__name__}"
        )
    if not isinstance(weights, pd.Series):
        raise ValueError(f"weights must be a pandas Series, got {type(weights).__name__}")

    dupes = weights.index[weights.index.duplicated()].tolist()
    if dupes:
        raise ValueError(f"weights has duplicated labels: {dupes}")
    missing = [label for label in weights.index if label not in returns.columns]
    if missing:
        raise ValueError(f"weights labels missing from returns columns: {missing}")
    if weights.isna().any():
        bad = weights.index[weights.isna()].tolist()
        raise ValueError(f"weights contains NaN values at labels: {bad}")

    used = returns[list(weights.index)]
    if used.isna().any().any():
        bad = used.columns[used.isna().any()].tolist()
        raise ValueError(f"returns contains NaN values in used columns: {bad}")

    total = float(weights.sum())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"weights must sum to 1 (within 1e-8), got sum={total!r}")

    portfolio = used.mul(weights, axis=1).sum(axis=1)
    portfolio.name = "portfolio"
    return portfolio
