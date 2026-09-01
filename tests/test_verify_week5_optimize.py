"""Independent verification suite for week-5 milestone 1 (portfolio/optimize.py).

Written by the adversarial verifier, deliberately NOT sharing helpers with
tests/test_portfolio_optimize.py. The teeth:

- A 3-asset min-volatility answer is checked against the analytic global minimum-variance
  portfolio ``w = S^-1 1 / (1' S^-1 1)`` computed here with raw numpy (annualization cancels).
- A symmetric 2-asset panel (asset Q is asset P's values reversed, so the sample variances are
  EXACTLY equal) must produce an exact 50/50 min-vol split — a hand-derivable answer with no
  estimation slack at all.
- max_sharpe is checked against the analytic tangency portfolio ``w ∝ S^-1 (mu - rf)`` built
  from independently computed estimators (geometric annualized mean, sample cov * 252,
  pypfopt's default rf = 0.02). Near the optimum the Sharpe surface is flat, so the weights are
  only loosely pinned — the sharp assertion is on the OBJECTIVE: the module's weights must
  achieve the analytic optimum's Sharpe to within 0.1% relative. A wrong estimator (arithmetic
  vs geometric mean, missing annualization, ddof drift) or wrong objective wiring moves the
  achieved Sharpe far more than that.
- Adversarial input probes: NaN bounds, boolean bounds, an empty weights Series, a binding
  upper bound, NaN in an UNUSED returns column (must be accepted), and list-typed bounds that
  must not be mutated.
"""

import numpy as np
import pandas as pd
import pytest

from quantforge.portfolio.optimize import combine_returns, optimize_weights

RF_PYPFOPT_DEFAULT = 0.02  # pypfopt EfficientFrontier.max_sharpe() default risk-free rate


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", periods=n)


def _panel3(seed: int = 11, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "A": 0.0009 + 0.011 * rng.standard_normal(n),
            "B": 0.0011 + 0.014 * rng.standard_normal(n),
            "C": 0.0008 + 0.009 * rng.standard_normal(n),
        },
        index=_dates(n),
    )


# ---------------------------------------------------------------------------
# analytic cross-checks (no module code involved in the expected values)
# ---------------------------------------------------------------------------


def test_min_vol_three_assets_matches_analytic_global_minimum():
    panel = _panel3()
    cov = panel.cov().values  # daily; the 252x annualization cancels in the weight ratio
    inv_ones = np.linalg.solve(cov, np.ones(3))
    w_expected = inv_ones / inv_ones.sum()
    assert (w_expected > 0).all() and (w_expected < 1).all()  # interior: bounds must not bind

    w = optimize_weights(panel, {"objective": "min_volatility"})
    np.testing.assert_allclose(w.values, w_expected, atol=1e-4)  # clean_weights rounds to 5 dp


def test_min_vol_symmetric_pair_is_exact_fifty_fifty():
    """Q holds P's values in reverse order, so var(P) == var(Q) EXACTLY: the closed-form
    min-variance weight is 1/2 with zero estimation slack. Any asymmetry bug (ddof drift
    between assets, column-order sensitivity) breaks the tie."""
    rng = np.random.default_rng(5)
    a = 0.001 + 0.01 * rng.standard_normal(300)
    panel = pd.DataFrame({"P": a, "Q": a[::-1]}, index=_dates(300))

    w = optimize_weights(panel, {"objective": "min_volatility"})
    assert w["P"] == pytest.approx(0.5, abs=1e-4)
    assert w["Q"] == pytest.approx(0.5, abs=1e-4)


def test_max_sharpe_achieves_analytic_tangency_sharpe():
    panel = _panel3()

    # Independent estimators mirroring pypfopt's textbook defaults, built from scratch:
    mu = (1.0 + panel).prod() ** (252.0 / len(panel)) - 1.0  # geometric, annualized
    cov = panel.cov().values * 252.0
    w_tan = np.linalg.solve(cov, (mu - RF_PYPFOPT_DEFAULT).values)
    w_tan = w_tan / w_tan.sum()
    assert (w_tan > 0).all()  # interior: the long-only bounds must not bind

    def sharpe(w: np.ndarray) -> float:
        return float((w @ mu.values - RF_PYPFOPT_DEFAULT) / np.sqrt(w @ cov @ w))

    w = optimize_weights(panel)  # default objective is max_sharpe
    # The Sharpe surface is flat near the optimum, so weights are loosely pinned (the numeric
    # solver stops within ~2e-2 of the analytic point) — but the achieved objective must sit
    # within 0.1% of the true optimum. A wrong mean (arithmetic vs geometric), a missing 252x,
    # or a min-vol/max-sharpe mix-up all fail this by a wide margin.
    assert sharpe(w.values) >= sharpe(w_tan) * (1.0 - 1e-3)
    np.testing.assert_allclose(w.values, w_tan, atol=0.02)


def test_binding_upper_bound_is_respected():
    """A dominant asset capped at 0.5 must sit AT the cap, not above it — including after the
    post-clean_weights renormalization inside optimize_weights."""
    rng = np.random.default_rng(3)
    n = 400
    panel = pd.DataFrame(
        {
            "DOM": 0.0040 + 0.005 * rng.standard_normal(n),
            "X": 0.0002 + 0.020 * rng.standard_normal(n),
            "Y": 0.0001 + 0.025 * rng.standard_normal(n),
        },
        index=_dates(n),
    )
    w = optimize_weights(panel, {"weight_bounds": (0.0, 0.5)})
    assert abs(float(w.sum()) - 1.0) <= 1e-8
    assert (w <= 0.5 + 1e-4).all() and (w >= -1e-12).all()
    assert w["DOM"] == pytest.approx(0.5, abs=1e-4)  # the cap binds for the dominant asset


# ---------------------------------------------------------------------------
# adversarial input probes
# ---------------------------------------------------------------------------


def test_rejects_nan_weight_bounds():
    with pytest.raises(ValueError, match="weight_bounds"):
        optimize_weights(_panel3(), {"weight_bounds": (float("nan"), 1.0)})


def test_rejects_boolean_weight_bounds():
    # bool is an int subclass; (False, True) must not sneak through as (0, 1).
    with pytest.raises(ValueError, match="weight_bounds"):
        optimize_weights(_panel3(), {"weight_bounds": (False, True)})


def test_rejects_non_dict_params():
    with pytest.raises(ValueError, match="params"):
        optimize_weights(_panel3(), [("objective", "max_sharpe")])


def test_list_typed_bounds_accepted_and_not_mutated():
    panel = _panel3()
    bounds = [0.0, 1.0]  # a list is a reasonable caller spelling of a pair
    params = {"weight_bounds": bounds}
    w_list = optimize_weights(panel, params)
    assert bounds == [0.0, 1.0]  # untouched
    assert params == {"weight_bounds": [0.0, 1.0]}
    w_tuple = optimize_weights(panel, {"weight_bounds": (0.0, 1.0)})
    pd.testing.assert_series_equal(w_list, w_tuple)


def test_combine_uses_only_weighted_columns():
    """Columns absent from weights.index must be ignored entirely — even a NaN there is fine,
    because that stream is not part of the blend."""
    frame = pd.DataFrame(
        {
            "MOM": [0.01, -0.02, 0.03],
            "MRV": [0.00, 0.01, -0.01],
            "JUNK": [np.nan, 99.0, -99.0],  # must not contaminate or trigger the NaN check
        },
        index=_dates(3),
    )
    out = combine_returns(frame, pd.Series({"MOM": 0.6, "MRV": 0.4}))
    expected = pd.Series([0.006, -0.008, 0.014], index=frame.index, name="portfolio")
    pd.testing.assert_series_equal(out, expected, atol=1e-12, rtol=0.0)


def test_combine_rejects_empty_weights():
    frame = pd.DataFrame({"A": [0.01], "B": [0.02]}, index=_dates(1))
    with pytest.raises(ValueError, match="sum to 1"):
        combine_returns(frame, pd.Series(dtype=float))


def test_combine_rejects_leveraged_weights():
    frame = pd.DataFrame({"A": [0.01, 0.0], "B": [0.02, 0.0]}, index=_dates(2))
    with pytest.raises(ValueError, match="sum to 1"):
        combine_returns(frame, pd.Series({"A": 1.0, "B": 1.0}))  # 2x leverage, no silent fix


def test_optimizer_output_is_directly_combinable_and_hand_checked():
    """End-to-end: weights out of the optimizer blend the panel, and every row of the blend
    equals the row-wise dot product computed by hand with raw numpy."""
    panel = _panel3(seed=42, n=120)
    w = optimize_weights(panel, {"objective": "min_volatility"})
    blended = combine_returns(panel, w)
    expected = panel.values @ w.values  # positional is safe: w is in panel column order
    np.testing.assert_allclose(blended.values, expected, atol=1e-12)
