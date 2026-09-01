"""Independent verifier for the FR-4 strategy-combination milestone (week 5, milestone 3).

Written adversarially, sharing no fixtures with tests/test_portfolio_combination.py: every
expected value here is either a hand-computed literal or a re-derivation from raw numpy on
independently constructed inputs. The goal is to catch the failure modes a friendly test could
paper over:

- a blend that is *closeness* rather than the exact fixed-weight arithmetic (hand-computed panel);
- weight/stream pairing by POSITION instead of by LABEL (reversed-order weights must blend
  identically — positional pairing would silently swap the streams);
- look-ahead sneaking in anywhere in the combination pipeline (truncation invariance: strategy
  return streams computed from a 300-day price prefix must equal the first 300 rows of the
  streams computed from the full 400 days, byte-for-byte — any dependence on future data breaks
  this);
- transaction costs vanishing between the engine and the blend (cost_bps=10 blend must end
  strictly below the cost-free blend under identical weights);
- malformed inputs accepted silently (bad budget, bogus labels, NaNs, off-whitelist params);
- the builder's "interior weights" claim being vacuous (min_volatility on the milestone's exact
  seeded setup must give both strategies material weight, or the weighted-sum proof proves
  nothing).
"""

import numpy as np
import pandas as pd
import pytest

from quantforge.data.loader import get_split_bounds
from quantforge.engine.python_engine import PythonEngine
from quantforge.metrics.performance import compute_metrics
from quantforge.portfolio.optimize import combine_returns, optimize_weights
from quantforge.strategies import STRATEGIES, validate_params

_STRATEGY_NAMES = ["momentum", "mean_reversion"]


def _synthetic_prices(n: int = 400, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Same generator family as tests/test_smoke_vertical_slice.py, anchored at train start."""
    train_start = get_split_bounds()["train"][0]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(train_start, periods=n)
    rets = rng.normal(0.0004, 0.01, size=(n, k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(k)])


def _strategy_panel(prices: pd.DataFrame, cost_bps: float = 10) -> pd.DataFrame:
    """Registry-driven momentum + mean-reversion return panel, built independently of the
    builder's helper (dict-of-Series constructor rather than concat) so a bug in either
    construction path cannot hide in both suites."""
    engine = PythonEngine()
    cols = {}
    for name in _STRATEGY_NAMES:
        params = validate_params(name, {})
        positions = STRATEGIES[name]().generate_signals(prices, params)
        cols[name] = engine.run_backtest(prices, positions, {"cost_bps": cost_bps}).returns
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Hand-computed ground truth for combine_returns + compute_metrics
# ---------------------------------------------------------------------------


def _hand_panel() -> pd.DataFrame:
    """Six-day two-stream panel with round numbers so the blend is checkable by hand."""
    dates = pd.bdate_range("2010-01-04", periods=6)
    return pd.DataFrame(
        {
            "momentum": [0.0, 0.01, -0.02, 0.015, 0.0, 0.005],
            "mean_reversion": [0.0, -0.005, 0.01, 0.0, 0.02, -0.01],
        },
        index=dates,
    )


def test_combine_returns_matches_hand_computed_blend():
    """0.6/0.4 blend of the hand panel, every element written out as a literal:
    e.g. day 2: 0.6*0.01 + 0.4*(-0.005) = 0.006 - 0.002 = 0.004."""
    panel = _hand_panel()
    weights = pd.Series([0.6, 0.4], index=["momentum", "mean_reversion"])
    blended = combine_returns(panel, weights)
    expected = [0.0, 0.004, -0.008, 0.009, 0.008, -0.001]
    np.testing.assert_allclose(blended.to_numpy(), expected, rtol=0.0, atol=1e-12)


def test_metrics_on_hand_blend_match_pen_and_paper():
    """compute_metrics on the hand blend cross-checked against raw-numpy formulas applied to
    the literal expected values — total_return as the compounded product, ann_vol as the
    population std (ddof=0, the module's documented convention) times sqrt(252)."""
    panel = _hand_panel()
    weights = pd.Series([0.6, 0.4], index=["momentum", "mean_reversion"])
    metrics = compute_metrics(combine_returns(panel, weights))

    literal = np.array([0.0, 0.004, -0.008, 0.009, 0.008, -0.001])
    assert metrics["total_return"] == pytest.approx(np.prod(1.0 + literal) - 1.0, abs=1e-12)
    assert metrics["ann_vol"] == pytest.approx(literal.std(ddof=0) * np.sqrt(252), abs=1e-12)
    assert metrics["hit_rate"] == pytest.approx((literal > 0).mean(), abs=1e-12)


def test_weights_align_by_label_not_position():
    """Adversarial mispairing: the same weights handed over in REVERSED index order must give an
    identical blend. An implementation that zipped weights to columns positionally would swap
    the two streams here and disagree on every non-symmetric day."""
    panel = _hand_panel()
    forward = pd.Series([0.6, 0.4], index=["momentum", "mean_reversion"])
    reversed_order = pd.Series([0.4, 0.6], index=["mean_reversion", "momentum"])
    pd.testing.assert_series_equal(
        combine_returns(panel, forward),
        combine_returns(panel, reversed_order),
        check_exact=True,
    )


def test_extra_panel_column_is_ignored_not_blended():
    """A distractor stream in the panel but absent from the weights must not leak into the
    blend — combine_returns selects columns by weights.index, nothing more."""
    panel = _hand_panel()
    weights = pd.Series([0.6, 0.4], index=["momentum", "mean_reversion"])
    baseline = combine_returns(panel, weights)
    with_distractor = panel.assign(distractor=999.0)
    pd.testing.assert_series_equal(
        combine_returns(with_distractor, weights), baseline, check_exact=True
    )


# ---------------------------------------------------------------------------
# Malformed inputs must be rejected loudly
# ---------------------------------------------------------------------------


def test_combine_rejects_bad_budget_and_labels():
    panel = _hand_panel()
    with pytest.raises(ValueError, match="sum"):
        combine_returns(panel, pd.Series([0.6, 0.3], index=["momentum", "mean_reversion"]))
    with pytest.raises(ValueError, match="sum"):  # leverage is a budget bug too, not a feature
        combine_returns(panel, pd.Series([0.7, 0.4], index=["momentum", "mean_reversion"]))
    with pytest.raises(ValueError, match="missing"):
        combine_returns(panel, pd.Series([0.6, 0.4], index=["momentum", "bogus"]))
    with pytest.raises(ValueError, match="NaN"):
        combine_returns(panel, pd.Series([0.6, np.nan], index=["momentum", "mean_reversion"]))


def test_combine_rejects_nan_in_used_stream():
    panel = _hand_panel()
    panel.loc[panel.index[2], "mean_reversion"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        combine_returns(panel, pd.Series([0.5, 0.5], index=["momentum", "mean_reversion"]))


def test_optimize_rejects_off_whitelist_params_and_nan_panel():
    """The optimizer gate mirrors the strategy-params whitelist spirit: typos and unknown
    objectives fail loudly instead of silently falling back to a default."""
    panel = _strategy_panel(_synthetic_prices(n=80))
    with pytest.raises(ValueError, match="objective"):
        optimize_weights(panel, {"objective": "max_returns"})
    with pytest.raises(ValueError, match="unknown params"):
        optimize_weights(panel, {"objectve": "max_sharpe"})  # the typo case, verbatim
    nan_panel = panel.copy()
    nan_panel.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        optimize_weights(nan_panel)


# ---------------------------------------------------------------------------
# Look-ahead and cost accounting through the WHOLE pipeline
# ---------------------------------------------------------------------------


def test_panel_is_truncation_invariant_no_lookahead():
    """Chop off the last 100 days of prices and rebuild the panel: the first 300 rows must be
    IDENTICAL to the full-history panel. Every operation upstream of the blend (signals,
    shift(1), costs) may only look backward, so removing future data cannot change the past.
    A centered window, an unshifted signal, or any full-sample normalization would fail this."""
    full_prices = _synthetic_prices(n=400)
    panel_full = _strategy_panel(full_prices)
    panel_trunc = _strategy_panel(full_prices.iloc[:300])
    pd.testing.assert_frame_equal(panel_trunc, panel_full.iloc[:300], check_exact=True)


def test_costs_survive_into_the_blend():
    """cost_bps=10 must make the blended equity end strictly below the cost-free blend under
    IDENTICAL fixed weights: positions are cost-independent, so net = gross - costs pointwise,
    and with nonneg weights the blend inherits the drag. If costs were dropped anywhere between
    the engine and combine_returns, the two blends would tie."""
    prices = _synthetic_prices()
    weights = pd.Series([0.5, 0.5], index=_STRATEGY_NAMES)  # fixed: isolate the cost effect
    blended_costly = combine_returns(_strategy_panel(prices, cost_bps=10), weights)
    blended_free = combine_returns(_strategy_panel(prices, cost_bps=0), weights)

    assert (blended_costly <= blended_free + 1e-15).all()  # pointwise drag, never a boost
    final_costly = float((1.0 + blended_costly).prod())
    final_free = float((1.0 + blended_free).prod())
    assert final_costly < final_free  # strict: both strategies demonstrably traded


def test_min_volatility_weights_are_interior_on_milestone_setup():
    """Guards the milestone's own proof against vacuity: on the exact seeded setup the main
    suite uses (seed 0, 400 days, train-anchored), min_volatility must give BOTH strategies
    material weight. If a code or estimator change ever corners these weights, the
    'blend equals the weighted sum' proof degenerates to a one-strategy identity and this
    test says so explicitly."""
    panel = _strategy_panel(_synthetic_prices())
    weights = optimize_weights(panel, {"objective": "min_volatility"})
    assert weights.min() > 0.05
    assert abs(float(weights.sum()) - 1.0) <= 1e-8


def test_full_pipeline_blend_reproduced_from_raw_arrays():
    """End-to-end re-derivation with no pandas in the arithmetic: run the real pipeline, then
    rebuild the blend from .to_numpy() arrays and float weights alone; compare at 1e-12."""
    panel = _strategy_panel(_synthetic_prices())
    weights = optimize_weights(panel, {"objective": "min_volatility"})
    blended = combine_returns(panel, weights)

    raw = (
        float(weights["momentum"]) * panel["momentum"].to_numpy()
        + float(weights["mean_reversion"]) * panel["mean_reversion"].to_numpy()
    )
    np.testing.assert_allclose(blended.to_numpy(), raw, rtol=0.0, atol=1e-12)
    metrics = compute_metrics(blended)
    assert all(np.isfinite(v) for v in metrics.values())
