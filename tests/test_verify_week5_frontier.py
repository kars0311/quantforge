"""Independent verification suite for week-5 milestone 2 (frontier + AR-2 emission).

Written by the adversarial verifier, deliberately sharing no helpers with
tests/test_portfolio_optimize.py. The teeth:

- POINTWISE closed-form cross-check on a 2-asset panel: with two assets under full investment,
  the portfolio is uniquely determined by its expected return, so for EVERY (risk, ret) row the
  frontier emits, the weight w1 = (ret - mu2) / (mu1 - mu2) is recoverable and the risk must
  equal sqrt(w' S w) computed here from first principles (geometric annualized mean, sample
  cov * 252 — pypfopt's documented estimators, re-derived with raw numpy/pandas). Any estimator
  drift, annualization bug, wrong objective, or a swapped (risk, ret) column pair fails this on
  every row at once.
- The bottom point is checked against the ANALYTIC global-minimum-variance portfolio
  ``w = S^-1 1 / (1' S^-1 1)`` (no module or pypfopt code in the expected value), and against
  optimize_weights(objective=min_volatility) — the done-when (c) consistency.
- The shared-estimator requirement ("factor mu/S into one private helper so the two functions
  can never drift") is enforced structurally: _estimate_moments is monkeypatched with a counting
  wrapper and both public functions must route through it.
- Adversarial probes: a boolean n_points (bool is an int subclass — must still be rejected),
  a numpy integer n_points (must be accepted), string n_points, and the done-when (e) cases.
- AR-2 is re-proven end-to-end with this file's own reshape and tmp_path round-trips, and the
  frontier/weights interchange schemas are pinned so a silent contract edit fails here too.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import quantforge.portfolio.optimize as optimize_module
from quantforge import interchange
from quantforge.portfolio.optimize import frontier, optimize_weights


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2019-01-02", periods=n)


def _panel2(n: int = 350) -> pd.DataFrame:
    """Two assets with WELL-SEPARATED expected returns (denominator mu1 - mu2 stays far from 0
    so recovering w1 from ret is numerically stable) and mild correlation."""
    rng = np.random.default_rng(99)
    common = rng.standard_normal(n)
    hi_mu = 0.0012 + 0.012 * (0.85 * rng.standard_normal(n) + 0.15 * common)
    lo_mu = 0.0002 + 0.007 * (0.85 * rng.standard_normal(n) + 0.15 * common)
    return pd.DataFrame({"HIMU": hi_mu, "LOMU": lo_mu}, index=_dates(n))


def _panel5(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cols = {}
    for i, (drift, vol) in enumerate(
        [(0.0011, 0.010), (0.0007, 0.013), (0.0013, 0.019), (0.0005, 0.011), (0.0009, 0.016)]
    ):
        cols[f"S{i}"] = drift + vol * rng.standard_normal(n)
    return pd.DataFrame(cols, index=_dates(n))


def _independent_mu_cov(panel: pd.DataFrame):
    """pypfopt's documented estimators re-derived from first principles with pandas/numpy only:
    geometric annualized mean (1+r).prod()**(252/n) - 1 and sample covariance * 252."""
    mu = (1.0 + panel).prod() ** (252.0 / len(panel)) - 1.0
    cov = panel.cov() * 252.0
    return mu, cov


# ---------------------------------------------------------------------------
# closed-form cross-checks (no module / pypfopt code in the expected values)
# ---------------------------------------------------------------------------


def test_every_frontier_point_matches_two_asset_closed_form():
    """In a 2-asset fully-invested portfolio, ret pins the weights, and the weights pin the
    risk. Check EVERY emitted row against that closed form."""
    panel = _panel2()
    mu, cov = _independent_mu_cov(panel)
    mu1, mu2 = float(mu["HIMU"]), float(mu["LOMU"])
    s11 = float(cov.loc["HIMU", "HIMU"])
    s22 = float(cov.loc["LOMU", "LOMU"])
    s12 = float(cov.loc["HIMU", "LOMU"])
    assert abs(mu1 - mu2) > 0.05  # the recovery below must be well-conditioned to be fair

    f = frontier(panel, n_points=15)
    assert len(f) >= 13  # max(2, 15 - 2)

    for risk, ret in f.itertuples(index=False):
        w1 = (ret - mu2) / (mu1 - mu2)
        assert -1e-6 <= w1 <= 1.0 + 1e-6  # long-only: recovered weight must be feasible
        w2 = 1.0 - w1
        expected_risk = np.sqrt(w1 * w1 * s11 + 2.0 * w1 * w2 * s12 + w2 * w2 * s22)
        assert risk == pytest.approx(expected_risk, rel=1e-5), (
            f"frontier point (risk={risk}, ret={ret}) is off the closed-form 2-asset frontier"
        )


def test_bottom_point_is_analytic_global_minimum_variance():
    panel = _panel5()
    mu, cov = _independent_mu_cov(panel)
    inv_ones = np.linalg.solve(cov.to_numpy(), np.ones(len(cov)))
    w_gmv = inv_ones / inv_ones.sum()
    assert (w_gmv > 0).all()  # interior: (0,1) bounds must not bind for a fair comparison

    ret_gmv = float(w_gmv @ mu.to_numpy())
    vol_gmv = float(np.sqrt(w_gmv @ cov.to_numpy() @ w_gmv))

    f = frontier(panel, n_points=12)
    assert f["ret"].iloc[0] == pytest.approx(ret_gmv, rel=1e-4)
    assert f["risk"].iloc[0] == pytest.approx(vol_gmv, rel=1e-3)


def test_bottom_point_consistent_with_optimize_weights_min_volatility():
    """Done-when (c): the lowest-ret frontier point IS the min_volatility portfolio that
    optimize_weights returns, evaluated with this file's independent (mu, S)."""
    panel = _panel5()
    mu, cov = _independent_mu_cov(panel)
    w = optimize_weights(panel, {"objective": "min_volatility"})

    ret_w = float(w.to_numpy() @ mu.to_numpy())
    vol_w = float(np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy()))

    f = frontier(panel, n_points=20)
    assert f["ret"].iloc[0] == pytest.approx(ret_w, rel=1e-4)
    assert f["risk"].iloc[0] == pytest.approx(vol_w, rel=1e-3)


def test_top_point_reaches_but_never_exceeds_max_mu():
    panel = _panel5()
    mu, _ = _independent_mu_cov(panel)
    f = frontier(panel, n_points=10)
    top_ret = float(f["ret"].iloc[-1])
    assert top_ret == pytest.approx(float(mu.max()), rel=1e-4)  # sweep reaches the corner
    assert top_ret <= float(mu.max()) + 1e-12  # but never claims more than attainable


# ---------------------------------------------------------------------------
# done-when (b) and (d): monotonicity, shape, sortedness, finiteness
# ---------------------------------------------------------------------------


def test_risk_monotone_nondecreasing_with_ret_five_assets():
    """Doc-07 done-when on a >=4-asset seeded panel: the sweep starts AT the min-vol point,
    so every consecutive pair must have non-decreasing risk as ret rises."""
    f = frontier(_panel5(), n_points=30)
    ret = f["ret"].to_numpy()
    risk = f["risk"].to_numpy()
    assert (np.diff(ret) >= -1e-12).all()
    assert (np.diff(risk) >= -1e-9).all()
    # ...and the bottom point is the global risk minimum of the sweep.
    assert risk[0] <= risk.min() + 1e-9


@pytest.mark.parametrize("n_points", [2, 7, 50])
def test_shape_dtypes_row_count_and_contract(n_points):
    f = frontier(_panel5(), n_points=n_points)
    assert list(f.columns) == ["risk", "ret"]
    assert str(f["risk"].dtype) == "float64"
    assert str(f["ret"].dtype) == "float64"
    assert isinstance(f.index, pd.RangeIndex)
    assert max(2, n_points - 2) <= len(f) <= n_points
    assert np.isfinite(f.to_numpy()).all()
    interchange.validate_frame(f, "frontier")  # done-when (a): must not raise


# ---------------------------------------------------------------------------
# structural guarantee: one shared moment estimator (the spec's no-drift rule)
# ---------------------------------------------------------------------------


def test_both_functions_route_through_the_shared_moment_helper(monkeypatch):
    """The spec requires (mu, S) to come from ONE private helper for both functions. Enforce
    the wiring, not just the current numerical agreement: wrap _estimate_moments with a counter
    and require both public entry points to pass through it."""
    calls = []
    real = optimize_module._estimate_moments

    def counting(returns):
        calls.append(len(returns))
        return real(returns)

    monkeypatch.setattr(optimize_module, "_estimate_moments", counting)
    panel = _panel5()
    optimize_weights(panel, {"objective": "min_volatility"})
    assert len(calls) == 1, "optimize_weights must estimate (mu, S) via _estimate_moments"
    frontier(panel, n_points=3)
    assert len(calls) == 2, "frontier must estimate (mu, S) via the same _estimate_moments"


# ---------------------------------------------------------------------------
# adversarial input probes / done-when (e)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_points", [1, 0])
def test_rejects_n_points_below_two(n_points):
    with pytest.raises(ValueError, match="n_points"):
        frontier(_panel2(), n_points=n_points)


def test_rejects_boolean_n_points():
    """bool subclasses int, so a naive isinstance(int) check would accept True (== 1) and
    silently produce a nonsense one-point 'frontier'. It must be rejected as a type error."""
    with pytest.raises(ValueError, match="n_points must be an int"):
        frontier(_panel2(), n_points=True)


def test_rejects_string_and_float_n_points():
    with pytest.raises(ValueError, match="n_points must be an int"):
        frontier(_panel2(), n_points="10")
    with pytest.raises(ValueError, match="n_points must be an int"):
        frontier(_panel2(), n_points=10.0)


def test_accepts_numpy_integer_n_points():
    """np.int64 is what naturally falls out of numpy arithmetic; rejecting it would be a trap."""
    f = frontier(_panel2(), n_points=np.int64(4))
    assert max(2, 4 - 2) <= len(f) <= 4


def test_rejects_nan_panel_before_touching_the_solver():
    panel = _panel2()
    panel.iloc[7, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        frontier(panel)


def test_frontier_input_not_mutated():
    panel = _panel2()
    before = panel.copy(deep=True)
    frontier(panel, n_points=4)
    pd.testing.assert_frame_equal(panel, before)


# ---------------------------------------------------------------------------
# AR-2 emission + frozen contract (done-when a, f)
# ---------------------------------------------------------------------------


def test_interchange_schemas_for_weights_and_frontier_are_unchanged():
    """The interchange contract is frozen; the milestone must not have touched it. Pin the two
    kinds this milestone emits so any 'convenient' schema edit fails here as well as in the
    dedicated freeze suite."""
    assert interchange.SCHEMAS["frontier"].names == ["risk", "ret"]
    assert all(t == pa.float64() for t in interchange.SCHEMAS["frontier"].types)
    assert interchange.SCHEMAS["weights"].names == ["ticker", "weight"]
    assert interchange.SCHEMAS["weights"].types == [pa.string(), pa.float64()]


def test_frontier_round_trips_through_interchange(tmp_path):
    f = frontier(_panel5(), n_points=9)
    path = str(tmp_path / "verify_frontier.parquet")
    interchange.write_frame(f, path, "frontier")
    back = interchange.read_frame(path, "frontier")
    pd.testing.assert_frame_equal(f, back, check_exact=False, rtol=1e-12)


def test_weights_reshape_validates_and_round_trips(tmp_path):
    w = optimize_weights(_panel5(), {"objective": "min_volatility"})
    weights_frame = pd.DataFrame(
        {"ticker": [str(t) for t in w.index], "weight": w.to_numpy(dtype="float64")}
    )
    interchange.validate_frame(weights_frame, "weights")  # done-when (a): must not raise

    path = str(tmp_path / "verify_weights.parquet")
    interchange.write_frame(weights_frame, path, "weights")
    back = interchange.read_frame(path, "weights")
    pd.testing.assert_frame_equal(weights_frame, back, check_exact=False, rtol=1e-12)
    assert float(back["weight"].sum()) == pytest.approx(1.0, abs=1e-8)
    assert (back["weight"].to_numpy() >= -1e-12).all()  # long-only survives the round-trip
