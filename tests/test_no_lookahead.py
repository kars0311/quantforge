"""RG-1 proof suite: the engine's `positions.shift(1)` makes look-ahead profits impossible.

All tests are offline, on seeded synthetic data (same pattern as test_smoke_vertical_slice.py).

The headline test plants a deliberately cheating signal and shows the engine pays it nothing.
Indexing, spelled out so the claim is defensible:

- `asset_returns[t]` (``pct_change``) is the close-to-close return that *prints on* day t.
- The cheat sets ``positions[t] = sign(asset_returns[t])`` — it "knows" day t's return at the
  moment it sets day t's weight. A leaky engine that applies weights same-day would award it
  ``|asset_returns[t]|`` every single day: spectacular fake profits.
- The real engine applies ``held = positions.shift(1)``, so what actually gets earned is
  ``sign(asset_returns[t-1]) * asset_returns[t]``: yesterday's sign against today's iid return,
  which has zero edge. The shift is the *only* thing standing between the cheat and its
  profits — which is exactly what makes this a proof of no-look-ahead (and what makes the test
  fail loudly if anyone ever removes the shift).

The remaining tests pin the shift mechanics directly: a weight set on day t earns exactly day
t+1's return and nothing else; day 0 is always flat; and truncating the future never changes
the past.
"""

import numpy as np
import pandas as pd

from quantforge.engine.python_engine import PythonEngine


def _synthetic_prices(n: int = 1500, k: int = 4, seed: int = 7) -> pd.DataFrame:
    """Zero-drift iid lognormal prices: under H0 no causal signal has an edge on them.

    Drift is deliberately 0 (unlike the smoke test's 4bp/day): any nonzero mean the engine
    hands a signal here must come from information leakage, not market beta, so the
    "earns ~0" assertion below is clean.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    rets = rng.normal(0.0, 0.01, size=(n, k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(k)])


def test_prescient_signal_earns_nothing_through_engine():
    """A signal that peeks at the same-day return earns ~0 net through the engine.

    Two-sided proof:
    1. Sanity that the signal really is clairvoyant: applied same-day (no shift, computed
       directly here), it earns a hugely positive mean — |return| every day.
    2. Through the engine, the shift(1) delay reduces it to yesterday's sign vs today's iid
       return: mean statistically indistinguishable from zero (|mean| < 3*SE).

    cost_bps=0 so the statistical zero is exactly about information, not cost drag (a
    sign-flipping signal churns heavily; costs would push the mean negative and muddy H0).
    """
    prices = _synthetic_prices()
    asset_returns = prices.pct_change().fillna(0.0)
    k = prices.shape[1]

    # The cheat: weight_t = sign of the return printing on day t, scaled to gross exposure 1.
    positions = np.sign(asset_returns) / k

    # (1) Applied same-day the signal is wildly profitable -- each term is |ret|/k.
    cheat_daily = (positions * asset_returns).sum(axis=1)
    cheat_se = cheat_daily.std() / np.sqrt(len(cheat_daily))
    assert cheat_daily.mean() > 10.0 * cheat_se, "cheat signal should be clairvoyant same-day"

    # (2) Through the engine the shift kills it: mean net return is statistically zero.
    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 0})
    net = result.returns
    se = net.std() / np.sqrt(len(net))
    assert abs(net.mean()) < 3.0 * se, (
        f"engine paid the prescient signal mean={net.mean():.2e} "
        f"(3*SE={3.0 * se:.2e}) — look-ahead leak?"
    )
    # The engine must also pay it far less than same-day application would have.
    assert net.mean() < 0.1 * cheat_daily.mean()


def test_single_day_position_earns_exactly_next_days_return():
    """A weight of 1.0 held only on day t shows up as exactly day t+1's pct_change.

    This pins the shift mechanically (no statistics): decided at close of t, the position
    earns the t -> t+1 move and nothing else. cost_bps=0 so the equality is exact.
    """
    prices = _synthetic_prices(n=12, k=1, seed=3)
    t = 5
    positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    positions.iloc[t, 0] = 1.0

    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 0})
    expected = prices.iloc[:, 0].pct_change()

    assert abs(result.returns.iloc[t + 1] - expected.iloc[t + 1]) < 1e-12
    others = result.returns.drop(result.returns.index[t + 1])
    assert (others == 0.0).all(), "position on day t must affect day t+1 only"


def test_day_zero_is_flat():
    """Day 0's return is exactly 0 even with a full weight on day 0.

    There is no day -1 on which a position could have been decided, so shift(1) leaves the
    book empty on day 0 — the engine can never book a return before the first decision.
    """
    prices = _synthetic_prices(n=50, k=3, seed=11)
    positions = pd.DataFrame(1.0 / 3.0, index=prices.index, columns=prices.columns)

    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 10})
    assert result.returns.iloc[0] == 0.0
    assert result.equity_curve.iloc[0] == 1.0


def test_tail_truncation_does_not_change_earlier_returns():
    """Chopping off the last k days leaves every earlier engine return bit-identical.

    If any future row influenced a past result (a smoothing, a full-sample normalization, a
    reversed shift), removing the tail would perturb the prefix. Run with nonzero costs so
    the invariance covers the turnover/cost path too, not just gross returns.
    """
    prices = _synthetic_prices(n=300, k=4, seed=5)
    rng = np.random.default_rng(6)
    raw = rng.uniform(0.0, 1.0, size=prices.shape)
    positions = pd.DataFrame(
        raw / raw.sum(axis=1, keepdims=True), index=prices.index, columns=prices.columns
    )

    engine = PythonEngine()
    full = engine.run_backtest(prices, positions, {"cost_bps": 10})

    k_tail = 30
    truncated = engine.run_backtest(
        prices.iloc[:-k_tail], positions.iloc[:-k_tail], {"cost_bps": 10}
    )

    pd.testing.assert_series_equal(truncated.returns, full.returns.iloc[:-k_tail], check_exact=True)
    pd.testing.assert_series_equal(
        truncated.equity_curve, full.equity_curve.iloc[:-k_tail], check_exact=True
    )
