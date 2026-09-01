"""Week-5 proof suite: mean-variance weights + fixed-weight combination (FR-4).

All tests are offline on hand-built or seeded synthetic data, matching the pattern of the
week-3/4 suites. The teeth of the suite:

- The min-volatility answer on a 2-asset panel is checked against the CLOSED-FORM minimum-
  variance weights, computed here from first principles with plain pandas/numpy and no shared
  code with the module: for covariance [[s11, s12], [s12, s22]],

      w1 = (s22 - s12) / (s11 + s22 - 2*s12),   w2 = 1 - w1

  (the annualization factor 252 cancels, so the sample covariance of daily returns suffices).
  A wrong estimator (ddof, annualization asymmetry, mean/cov mix-up) or a wrong objective wiring
  in the module moves the weights and fails the assertion.
- combine_returns is checked against a fully hand-computed 3-day blend at atol 1e-12, including
  a shuffled-weight-order call that proves alignment is by LABEL, not position.
- Every documented ValueError rejection class is exercised with a message match, so validation
  cannot silently rot into a bare `raise ValueError`.
- The frontier's lowest point is cross-checked against optimize_weights(min_volatility) with the
  (risk, ret) of those weights recomputed here via pypfopt's own portfolio_performance on
  independently re-estimated (mu, S) — a drift between the two functions' estimators or an
  off-frontier bottom point fails the assertion.
- AR-2 emission is proven end-to-end: the frontier frame and the reshaped weights frame both
  pass interchange validation and round-trip through write_frame/read_frame unchanged.
"""

import numpy as np
import pandas as pd
import pytest
from pypfopt import expected_returns as _pypfopt_expected_returns
from pypfopt import risk_models as _pypfopt_risk_models
from pypfopt.base_optimizer import portfolio_performance as _pypfopt_performance

from quantforge import interchange
from quantforge.portfolio.optimize import combine_returns, frontier, optimize_weights


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _seeded_panel_4() -> pd.DataFrame:
    """4-asset daily-return panel, seeded. Positive drift keeps max_sharpe feasible
    (pypfopt's max_sharpe needs at least one asset beating its risk-free default)."""
    rng = np.random.default_rng(7)
    n = 300
    data = {
        "A": 0.0010 + 0.010 * rng.standard_normal(n),
        "B": 0.0008 + 0.015 * rng.standard_normal(n),
        "C": 0.0012 + 0.020 * rng.standard_normal(n),
        "D": 0.0006 + 0.012 * rng.standard_normal(n),
    }
    return pd.DataFrame(data, index=_dates(n))


def _seeded_panel_2() -> pd.DataFrame:
    """2-asset panel whose unconstrained min-variance weights land strictly inside (0, 1):
    a low-vol and a high-vol asset with only mild correlation."""
    rng = np.random.default_rng(21)
    n = 400
    common = rng.standard_normal(n)
    lo = 0.0004 + 0.008 * (0.9 * rng.standard_normal(n) + 0.1 * common)
    hi = 0.0006 + 0.020 * (0.9 * rng.standard_normal(n) + 0.1 * common)
    return pd.DataFrame({"LOWVOL": lo, "HIGHVOL": hi}, index=_dates(n))


# ---------------------------------------------------------------------------
# optimize_weights — correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("objective", ["max_sharpe", "min_volatility"])
def test_weights_sum_to_one_and_respect_bounds(objective):
    panel = _seeded_panel_4()
    w = optimize_weights(panel, {"objective": objective})

    assert isinstance(w, pd.Series)
    assert list(w.index) == list(panel.columns)  # input column order preserved
    assert abs(float(w.sum()) - 1.0) <= 1e-8
    assert (w >= 0.0).all() and (w <= 1.0).all()


def test_min_volatility_matches_closed_form_two_assets():
    panel = _seeded_panel_2()

    # Independent closed-form minimum-variance solution (no module code involved).
    cov = panel.cov()  # daily sample covariance; 252x annualization cancels in the ratio
    s11 = cov.iloc[0, 0]
    s22 = cov.iloc[1, 1]
    s12 = cov.iloc[0, 1]
    w1_expected = (s22 - s12) / (s11 + s22 - 2.0 * s12)
    assert 0.0 < w1_expected < 1.0  # sanity: bounds must not bind for the comparison to be fair

    w = optimize_weights(panel, {"objective": "min_volatility"})
    # clean_weights rounds to 5 decimals, so exact rtol 1e-6 is unreachable; atol 1e-4 covers
    # the rounding while still catching any real estimator/objective error.
    assert w["LOWVOL"] == pytest.approx(w1_expected, abs=1e-4)
    assert w["HIGHVOL"] == pytest.approx(1.0 - w1_expected, abs=1e-4)


def test_max_sharpe_prefers_dominant_asset():
    """An asset with higher mean, lower vol, and low correlation to the rest must take the
    plurality of the max-Sharpe portfolio — the qualitative property any correct
    mean-variance implementation must reproduce."""
    rng = np.random.default_rng(3)
    n = 400
    panel = pd.DataFrame(
        {
            "DOM": 0.0040 + 0.005 * rng.standard_normal(n),  # high mean, low vol
            "X": 0.0002 + 0.020 * rng.standard_normal(n),
            "Y": 0.0001 + 0.025 * rng.standard_normal(n),
        },
        index=_dates(n),
    )
    w = optimize_weights(panel)  # default objective is max_sharpe
    assert w.idxmax() == "DOM"


def test_default_params_equal_explicit_max_sharpe():
    panel = _seeded_panel_4()
    pd.testing.assert_series_equal(
        optimize_weights(panel),
        optimize_weights(panel, {"objective": "max_sharpe", "weight_bounds": (0.0, 1.0)}),
    )


def test_optimize_is_deterministic():
    panel = _seeded_panel_4()
    for objective in ("max_sharpe", "min_volatility"):
        w1 = optimize_weights(panel, {"objective": objective})
        w2 = optimize_weights(panel, {"objective": objective})
        pd.testing.assert_series_equal(w1, w2)


def test_optimize_does_not_mutate_inputs():
    panel = _seeded_panel_4()
    panel_before = panel.copy(deep=True)
    params = {"objective": "min_volatility", "weight_bounds": (0.0, 1.0)}
    params_before = {"objective": "min_volatility", "weight_bounds": (0.0, 1.0)}

    optimize_weights(panel, params)

    pd.testing.assert_frame_equal(panel, panel_before)
    assert params == params_before
    assert isinstance(params["weight_bounds"], tuple)


# ---------------------------------------------------------------------------
# optimize_weights — rejection classes (each names the offending input)
# ---------------------------------------------------------------------------


def test_rejects_non_dataframe():
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        optimize_weights(pd.Series([0.01, 0.02]))


def test_rejects_single_column():
    panel = _seeded_panel_4()[["A"]]
    with pytest.raises(ValueError, match="at least 2 columns"):
        optimize_weights(panel)


def test_rejects_nan_returns():
    panel = _seeded_panel_4()
    panel.iloc[5, panel.columns.get_loc("B")] = np.nan
    with pytest.raises(ValueError, match=r"NaN.*B"):
        optimize_weights(panel)


def test_rejects_duplicated_columns():
    panel = _seeded_panel_4()
    panel.columns = ["A", "B", "B", "D"]
    with pytest.raises(ValueError, match=r"duplicated column labels.*B"):
        optimize_weights(panel)


def test_rejects_too_few_rows():
    panel = _seeded_panel_4().iloc[:1]
    with pytest.raises(ValueError, match="at least 2 rows"):
        optimize_weights(panel)


def test_rejects_unknown_objective():
    with pytest.raises(ValueError, match="unknown objective 'max_profit'"):
        optimize_weights(_seeded_panel_4(), {"objective": "max_profit"})


def test_rejects_unknown_param_keys():
    with pytest.raises(ValueError, match="unknown params keys.*objectve"):
        optimize_weights(_seeded_panel_4(), {"objectve": "max_sharpe"})


@pytest.mark.parametrize(
    "bounds, pattern",
    [
        ((0.0, 1.0, 2.0), "pair of numbers"),  # not a 2-tuple
        (0.5, "pair of numbers"),  # not a tuple at all
        ((0.5, 0.5), "low must be < high"),  # lo == hi
        ((0.8, 0.2), "low must be < high"),  # lo > hi
        ((-0.5, 1.0), r"long-only range \[0, 1\]"),  # shorting at the portfolio level
        ((0.0, 1.5), r"long-only range \[0, 1\]"),  # leverage
        ((0.0, 0.2), "infeasible"),  # 4 assets * 0.2 = 0.8 < 1: cannot fully invest
        ((0.3, 0.9), "infeasible"),  # 4 assets * 0.3 = 1.2 > 1: cannot sum to 1
    ],
)
def test_rejects_malformed_weight_bounds(bounds, pattern):
    with pytest.raises(ValueError, match=pattern):
        optimize_weights(_seeded_panel_4(), {"weight_bounds": bounds})


# ---------------------------------------------------------------------------
# combine_returns
# ---------------------------------------------------------------------------


def _hand_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"MOM": [0.01, -0.02, 0.03], "MRV": [0.00, 0.01, -0.01]},
        index=_dates(3),
    )


def test_combine_matches_hand_computation():
    frame = _hand_frame()
    weights = pd.Series({"MOM": 0.6, "MRV": 0.4})
    out = combine_returns(frame, weights)

    # Hand arithmetic: 0.6*0.01+0.4*0.00 = 0.006; 0.6*-0.02+0.4*0.01 = -0.008;
    #                  0.6*0.03+0.4*-0.01 = 0.014
    expected = pd.Series([0.006, -0.008, 0.014], index=frame.index, name="portfolio")
    pd.testing.assert_series_equal(out, expected, atol=1e-12, rtol=0.0)
    assert out.name == "portfolio"


def test_combine_aligns_by_label_not_position():
    frame = _hand_frame()
    shuffled = pd.Series({"MRV": 0.4, "MOM": 0.6})  # reversed order, same labels
    out = combine_returns(frame, shuffled)
    expected = pd.Series([0.006, -0.008, 0.014], index=frame.index, name="portfolio")
    pd.testing.assert_series_equal(out, expected, atol=1e-12, rtol=0.0)


def test_combine_does_not_mutate_inputs():
    frame = _hand_frame()
    frame_before = frame.copy(deep=True)
    weights = pd.Series({"MOM": 0.6, "MRV": 0.4})
    weights_before = weights.copy(deep=True)

    combine_returns(frame, weights)

    pd.testing.assert_frame_equal(frame, frame_before)
    pd.testing.assert_series_equal(weights, weights_before)


def test_combine_rejects_missing_label():
    with pytest.raises(ValueError, match=r"missing from returns columns.*GHOST"):
        combine_returns(_hand_frame(), pd.Series({"MOM": 0.5, "GHOST": 0.5}))


def test_combine_rejects_nan_weight():
    with pytest.raises(ValueError, match=r"weights contains NaN.*MRV"):
        combine_returns(_hand_frame(), pd.Series({"MOM": 1.0, "MRV": np.nan}))


def test_combine_rejects_nan_in_used_returns():
    frame = _hand_frame()
    frame.iloc[1, frame.columns.get_loc("MOM")] = np.nan
    with pytest.raises(ValueError, match=r"NaN.*MOM"):
        combine_returns(frame, pd.Series({"MOM": 0.6, "MRV": 0.4}))


def test_combine_rejects_weights_not_summing_to_one():
    # No silent renormalization: a wrong budget is the caller's bug, not ours to hide.
    with pytest.raises(ValueError, match="must sum to 1"):
        combine_returns(_hand_frame(), pd.Series({"MOM": 0.6, "MRV": 0.5}))


def test_combine_rejects_duplicated_weight_labels():
    dup = pd.Series([0.5, 0.5], index=["MOM", "MOM"])
    with pytest.raises(ValueError, match=r"duplicated labels.*MOM"):
        combine_returns(_hand_frame(), dup)


def test_combine_rejects_non_series_weights():
    with pytest.raises(ValueError, match="weights must be a pandas Series"):
        combine_returns(_hand_frame(), {"MOM": 0.6, "MRV": 0.4})


# ---------------------------------------------------------------------------
# frontier — shape, contract, monotonicity, min-vol consistency
# ---------------------------------------------------------------------------


def _independent_moments(panel: pd.DataFrame):
    """Re-estimate (mu, S) here with pypfopt's textbook estimators, sharing no module code,
    so a silent estimator change inside optimize.py fails the consistency tests below."""
    mu = _pypfopt_expected_returns.mean_historical_return(panel, returns_data=True)
    cov = _pypfopt_risk_models.sample_cov(panel, returns_data=True)
    return mu, cov


def test_frontier_shape_columns_dtypes_and_contract():
    panel = _seeded_panel_4()
    n_points = 25
    f = frontier(panel, n_points=n_points)

    assert isinstance(f, pd.DataFrame)
    assert list(f.columns) == ["risk", "ret"]
    assert (f.dtypes == "float64").all()
    assert max(2, n_points - 2) <= len(f) <= n_points
    assert np.isfinite(f.to_numpy()).all()
    interchange.validate_frame(f, "frontier")  # AR-2 contract: must not raise

    # The sweep's top must reach (just below) the best attainable expected return, which under
    # long-only fully-invested bounds is max(mu): all capital in the single best asset.
    mu, _ = _independent_moments(panel)
    assert f["ret"].iloc[-1] == pytest.approx(float(mu.max()), rel=1e-4)
    assert f["ret"].iloc[-1] <= float(mu.max()) + 1e-12


def test_frontier_sorted_by_ret_and_risk_monotone():
    """Doc-07 done-when: risk is monotonically non-decreasing with ret at/after the min-vol
    point. The sweep starts AT the min-vol return, so every row must satisfy it. The -1e-9
    tolerance absorbs QP solver noise on annualized vols of order 0.1 without letting any
    real non-convexity (which would mean a broken frontier) through."""
    panel = _seeded_panel_4()
    f = frontier(panel, n_points=25)

    assert (np.diff(f["ret"].to_numpy()) >= -1e-12).all()  # sorted by ret ascending
    assert (np.diff(f["risk"].to_numpy()) >= -1e-9).all()  # risk non-decreasing with ret


def test_frontier_min_vol_point_matches_optimize_weights():
    """The frontier's lowest-ret point must BE the min-volatility portfolio that
    optimize_weights returns — evaluated through pypfopt's portfolio_performance on
    independently re-estimated moments, so the two functions cannot drift apart."""
    panel = _seeded_panel_4()
    f = frontier(panel, n_points=20)
    w = optimize_weights(panel, {"objective": "min_volatility"})

    mu, cov = _independent_moments(panel)
    ret_w, vol_w, _ = _pypfopt_performance(w.to_dict(), mu, cov)

    assert f["ret"].iloc[0] == pytest.approx(ret_w, rel=1e-4)
    assert f["risk"].iloc[0] == pytest.approx(vol_w, rel=1e-3)
    # And the min-vol point is the risk minimum of the whole sweep.
    assert f["risk"].iloc[0] == pytest.approx(float(f["risk"].min()), rel=1e-9)


def test_frontier_minimal_two_points():
    f = frontier(_seeded_panel_4(), n_points=2)
    assert len(f) == 2
    assert f["ret"].iloc[1] > f["ret"].iloc[0]  # min-vol bottom, near-max-mu top


def test_frontier_is_deterministic():
    panel = _seeded_panel_4()
    pd.testing.assert_frame_equal(frontier(panel, n_points=10), frontier(panel, n_points=10))


def test_frontier_does_not_mutate_input():
    panel = _seeded_panel_4()
    before = panel.copy(deep=True)
    frontier(panel, n_points=5)
    pd.testing.assert_frame_equal(panel, before)


@pytest.mark.parametrize("n_points", [1, 0, -3])
def test_frontier_rejects_too_few_points(n_points):
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        frontier(_seeded_panel_4(), n_points=n_points)


def test_frontier_rejects_non_int_n_points():
    with pytest.raises(ValueError, match="n_points must be an int"):
        frontier(_seeded_panel_4(), n_points=10.0)


def test_frontier_rejects_nan_panel():
    panel = _seeded_panel_4()
    panel.iloc[3, panel.columns.get_loc("A")] = np.nan
    with pytest.raises(ValueError, match=r"NaN.*A"):
        frontier(panel)


def test_frontier_rejects_single_column():
    with pytest.raises(ValueError, match="at least 2 columns"):
        frontier(_seeded_panel_4()[["A"]])


# ---------------------------------------------------------------------------
# AR-2 emission: weights + frontier round-trip through the interchange contract
# ---------------------------------------------------------------------------


def test_weights_frame_validates_and_round_trips(tmp_path):
    """optimize_weights output, reshaped to the interchange `weights` kind (ticker, weight),
    must survive write_frame -> read_frame byte-for-value — this is exactly the artifact the
    UI and the week-6 R layer consume (AR-2)."""
    w = optimize_weights(_seeded_panel_4())
    weights_frame = pd.DataFrame(
        {"ticker": [str(label) for label in w.index], "weight": w.to_numpy(dtype="float64")}
    )
    interchange.validate_frame(weights_frame, "weights")  # must not raise

    path = tmp_path / "weights.parquet"
    interchange.write_frame(weights_frame, str(path), "weights")
    back = interchange.read_frame(str(path), "weights")

    pd.testing.assert_frame_equal(weights_frame, back, check_exact=False, rtol=1e-12)
    assert float(back["weight"].sum()) == pytest.approx(1.0, abs=1e-8)


def test_frontier_frame_round_trips(tmp_path):
    f = frontier(_seeded_panel_4(), n_points=15)

    path = tmp_path / "frontier.parquet"
    interchange.write_frame(f, str(path), "frontier")
    back = interchange.read_frame(str(path), "frontier")

    pd.testing.assert_frame_equal(f, back, check_exact=False, rtol=1e-12)


# ---------------------------------------------------------------------------
# integration: optimizer output feeds directly into the combiner
# ---------------------------------------------------------------------------


def test_optimizer_weights_feed_combine():
    panel = _seeded_panel_4()
    w = optimize_weights(panel, {"objective": "min_volatility"})
    blended = combine_returns(panel, w)
    assert len(blended) == len(panel)
    # Spot-check one row by hand: the blend at row 0 is the dot product of row 0 with weights.
    assert blended.iloc[0] == pytest.approx(float((panel.iloc[0] * w).sum()), abs=1e-12)
