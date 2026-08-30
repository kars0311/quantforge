"""Independent verifier tests for RG-1 (no look-ahead), complementing test_no_lookahead.py.

test_no_lookahead.py proves the shift statistically (a same-day peeker earns ~0). These tests
pin the *same* property from three different angles so a regression cannot slip past all of
them at once:

1. A fully hand-computed 4-day case: every gross return, turnover charge, and equity value is
   derived on paper in the docstring and asserted to 1e-12 — the shift and the cost accounting
   are both checked against arithmetic done by a human, not by the engine's own formulas.
2. A causality perturbation: bumping one *future* price must leave every earlier engine return
   bit-identical (and must change the perturbed day, proving the probe actually bites).
3. The engine's payment convention pinned exactly: positions[t] earns precisely day t+1's
   return, shown by feeding the engine tomorrow's sign and matching its net returns to
   mean(|asset returns|) day by day. This is the identity that makes the docs' original
   "sign-of-tomorrow earns ~0" phrasing wrong and the implemented same-day cheat right —
   pinning it here keeps that derivation from regressing silently.

All offline, seeded synthetic or hand-built data, same pattern as test_smoke_vertical_slice.py.
"""

import numpy as np
import pandas as pd

from quantforge.engine.python_engine import PythonEngine


def _iid_prices(n: int, k: int, seed: int) -> pd.DataFrame:
    """Zero-drift iid lognormal prices (no edge exists for any causal signal)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n)
    rets = rng.normal(0.0, 0.01, size=(n, k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"S{i}" for i in range(k)])


def test_hand_computed_four_day_case_exact():
    """Every number the engine produces on a tiny book matches paper arithmetic to 1e-12.

    Prices (1 asset): 100 -> 110 -> 99 -> 108.9, so pct_change = [0, +0.10, -0.10, +0.10].
    Positions (target weights decided at each close): [1, 0, 1, 0].
    Therefore held = shift(1) = [0, 1, 0, 1]:
      gross    = held * ret            = [0, 0.10, 0, 0.10]
      turnover = |diff(held)|          = [0, 1, 1, 1]
      costs    = turnover * 25bp       = [0, 0.0025, 0.0025, 0.0025]
      net                              = [0, 0.0975, -0.0025, 0.0975]
      equity   = cumprod(1 + net)      = [1, 1.0975, 1.09475625, 1.201494984375]

    Note the shift is visible in the numbers themselves: the day-1 gain comes from the weight
    decided on day 0, and the weight decided on day 1 (flat) is what dodges day 2's -10%.
    A same-day engine would instead produce gross = [0, 0, -0.10, 0].
    """
    dates = pd.bdate_range("2020-01-06", periods=4)
    prices = pd.DataFrame({"X": [100.0, 110.0, 99.0, 108.9]}, index=dates)
    positions = pd.DataFrame({"X": [1.0, 0.0, 1.0, 0.0]}, index=dates)

    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 25})

    expected_net = np.array([0.0, 0.0975, -0.0025, 0.0975])
    expected_equity = np.array([1.0, 1.0975, 1.09475625, 1.201494984375])
    np.testing.assert_allclose(result.returns.to_numpy(), expected_net, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.equity_curve.to_numpy(), expected_equity, rtol=0, atol=1e-12)
    assert abs(result.meta["total_turnover"] - 3.0) < 1e-12


def test_perturbing_a_future_price_never_changes_the_past():
    """Bump one price deep in the sample: all engine output before that day is bit-identical.

    Complements the tail-truncation test in test_no_lookahead.py: truncation removes future
    rows, this probe *alters* one — if any full-sample statistic, smoothing, or reversed shift
    leaked future data into earlier results, the prefix would move. The probe is validated by
    also asserting the perturbed day itself DOES change (so a vacuously-passing comparison of
    two identical runs cannot happen).
    """
    prices = _iid_prices(n=120, k=3, seed=21)
    rng = np.random.default_rng(22)
    raw = rng.uniform(0.0, 1.0, size=prices.shape)
    positions = pd.DataFrame(
        raw / raw.sum(axis=1, keepdims=True), index=prices.index, columns=prices.columns
    )

    engine = PythonEngine()
    base = engine.run_backtest(prices, positions, {"cost_bps": 10})

    p = 80  # perturbation day, well inside the sample
    bumped = prices.copy()
    bumped.iloc[p, 1] *= 1.05
    alt = engine.run_backtest(bumped, positions, {"cost_bps": 10})

    pd.testing.assert_series_equal(alt.returns.iloc[:p], base.returns.iloc[:p], check_exact=True)
    pd.testing.assert_series_equal(
        alt.equity_curve.iloc[:p], base.equity_curve.iloc[:p], check_exact=True
    )
    assert alt.returns.iloc[p] != base.returns.iloc[p], "probe must bite on the perturbed day"


def test_positions_are_paid_exactly_the_next_days_return():
    """Feeding the engine tomorrow's sign is paid |return| — pinning the t -> t+1 convention.

    positions[t] = sign(asset_returns[t+1]) / k means held[t] = sign(asset_returns[t]) / k, so
    net[t] must equal mean over assets of |asset_returns[t]| for every day after day 0. This
    day-by-day identity (not just a mean) proves positions[t] earns exactly day t+1's return —
    the derivation behind why test_no_lookahead.py peeks at the *same-day* return rather than
    the milestone spec's literal shift(-1) form, which this identity shows is genuinely
    prescient under the engine's convention and hence correctly profitable.
    """
    prices = _iid_prices(n=800, k=3, seed=13)
    asset_returns = prices.pct_change().fillna(0.0)
    k = prices.shape[1]

    tomorrow_sign = np.sign(asset_returns.shift(-1)).fillna(0.0) / k
    result = PythonEngine().run_backtest(prices, tomorrow_sign, {"cost_bps": 0})

    expected = asset_returns.abs().mean(axis=1)
    expected.iloc[0] = 0.0  # no day -1 decision exists, so day 0 is flat
    np.testing.assert_allclose(result.returns.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)
    # And the statistical corollary: a genuinely prescient signal IS strongly profitable.
    se = result.returns.std() / np.sqrt(len(result.returns))
    assert result.returns.mean() > 10.0 * se
