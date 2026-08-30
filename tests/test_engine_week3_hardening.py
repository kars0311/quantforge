"""Week-3 hardening tests for PythonEngine (offline, synthetic data).

Proves the three hardening claims of docs/components/04-python-engine.md:
1. Input validation — malformed inputs raise ValueError at the engine boundary.
2. Explicit NaN policy — NaN prices (pre-IPO) contribute exactly 0 return.
3. Enriched meta — engine/cost_bps/params/start/end/n_days/total_turnover, exact values.

Plus adversarial rigor cases: a prescient same-day signal must earn ~0 (shift(1) is doing its
job), and a fully hand-computed 2-asset run pins cost accounting to the documented algorithm.
"""

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.python_engine import PythonEngine
from quantforge.strategies.momentum import MomentumStrategy


def _synthetic_prices(n: int = 400, k: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    rets = rng.normal(0.0004, 0.01, size=(n, k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(k)])


def _flat_positions(prices: pd.DataFrame, weight: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(weight, index=prices.index, columns=prices.columns)


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------


def test_rejects_levered_positions_row():
    prices = _synthetic_prices(n=20, k=4)
    positions = _flat_positions(prices, 0.25)  # gross 1.0 everywhere...
    positions.iloc[7] = 0.375  # ...except one levered row: sum(|w|) = 1.5
    with pytest.raises(ValueError, match="gross exposure"):
        PythonEngine().run_backtest(prices, positions, {})


def test_rejects_levered_long_short_book():
    # |longs| + |shorts| = 1.3 > 1: levered even though net exposure is only 0.1.
    prices = _synthetic_prices(n=20, k=2)
    positions = _flat_positions(prices)
    positions["A0"] = 0.7
    positions["A1"] = -0.6
    with pytest.raises(ValueError, match="gross exposure"):
        PythonEngine().run_backtest(prices, positions, {})


def test_accepts_gross_one_long_short_and_float_noise():
    # Long-short within gross 1.0 is explicitly fine; so is 1/n float noise a few ULPs over.
    prices = _synthetic_prices(n=20, k=2)
    positions = _flat_positions(prices)
    positions["A0"] = 0.5
    positions["A1"] = -0.5
    PythonEngine().run_backtest(prices, positions, {})  # must not raise

    noisy = _flat_positions(prices, (1.0 + 5e-10) / 2)  # gross = 1 + 5e-10, inside eps
    PythonEngine().run_backtest(prices, noisy, {})  # must not raise


def test_rejects_shuffled_prices_index():
    prices = _synthetic_prices(n=30, k=3)
    positions = _flat_positions(prices, 0.1)
    shuffled = prices.sample(frac=1.0, random_state=1)
    assert not shuffled.index.is_monotonic_increasing  # the shuffle actually shuffled
    with pytest.raises(ValueError, match="monotonic"):
        PythonEngine().run_backtest(shuffled, positions, {})


def test_rejects_duplicated_dates_in_prices_index():
    prices = _synthetic_prices(n=10, k=2)
    dup = pd.concat([prices, prices.iloc[[4]]]).sort_index()  # sorted, but one repeated label
    positions = _flat_positions(prices, 0.1)
    with pytest.raises(ValueError, match="duplicated"):
        PythonEngine().run_backtest(dup, positions, {})


def test_rejects_non_numeric_columns():
    prices = _synthetic_prices(n=10, k=2)
    positions = _flat_positions(prices, 0.1)

    bad_prices = prices.copy()
    bad_prices["A0"] = "oops"
    with pytest.raises(ValueError, match=r"prices.*non-numeric.*A0"):
        PythonEngine().run_backtest(bad_prices, positions, {})

    bad_positions = positions.copy()
    bad_positions["A1"] = "oops"
    with pytest.raises(ValueError, match=r"positions.*non-numeric.*A1"):
        PythonEngine().run_backtest(prices, bad_positions, {})


# ---------------------------------------------------------------------------
# 2. NaN policy: NaN prices (pre-IPO) contribute exactly 0 return
# ---------------------------------------------------------------------------


def test_nan_prices_contribute_zero_return():
    # B is unlisted (NaN) for the first 4 days. Holding B through its NaN stretch — and
    # through its first listed day — must earn exactly 0: no fabricated listing-day return.
    dates = pd.bdate_range("2020-01-01", periods=8)
    prices = pd.DataFrame(
        {"A": [100.0] * 8, "B": [np.nan] * 4 + [50.0, 55.0, 55.0, 55.0]}, index=dates
    )
    positions = pd.DataFrame({"A": [0.0] * 8, "B": [1.0] * 8}, index=dates)

    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 0})
    # held B from day 1 onward; B's only real return is +10% on day 5 (50 -> 55).
    expected = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], index=dates)
    pd.testing.assert_series_equal(result.returns, expected, check_names=False)
    assert result.equity_curve.iloc[-1] == pytest.approx(1.1, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Enriched meta
# ---------------------------------------------------------------------------


def test_meta_contents_and_exact_turnover():
    prices = _synthetic_prices()
    params = {"cost_bps": 10, "note": "meta-test"}
    positions = MomentumStrategy().generate_signals(prices, {"lookback": 60, "top_n": 2})
    result = PythonEngine().run_backtest(prices, positions, params)

    meta = result.meta
    for key in ("engine", "cost_bps", "params", "start", "end", "n_days", "total_turnover"):
        assert key in meta, f"meta missing {key!r}"
    assert meta["engine"] == "python"
    assert meta["cost_bps"] == 10.0
    assert meta["params"] == params
    assert meta["start"] == str(prices.index[0])
    assert meta["end"] == str(prices.index[-1])
    assert meta["n_days"] == len(prices.index)

    # Hand-recompute total turnover from the documented algorithm: sum of |delta held|.
    held = (
        positions.reindex(prices.index)
        .reindex(columns=prices.columns)
        .fillna(0.0)
        .shift(1)
        .fillna(0.0)
    )
    hand_total = float(held.diff().abs().sum(axis=1).fillna(0.0).sum())
    assert meta["total_turnover"] == pytest.approx(hand_total, abs=1e-12)
    assert meta["total_turnover"] > 0.0  # momentum actually trades


# ---------------------------------------------------------------------------
# Adversarial rigor: hardening must not have bent the algorithm
# ---------------------------------------------------------------------------


def test_prescient_same_day_signal_earns_nothing():
    # Asset jumps +10% on exactly one day. A "prescient" strategy that puts weight on ONLY
    # that day (knowing the same-day return it could not have known at decision time) must
    # capture none of the jump: shift(1) applies the weight to the following (flat) day.
    dates = pd.bdate_range("2021-01-01", periods=10)
    px = pd.Series(100.0, index=dates)
    px.iloc[5:] = 110.0  # +10% on day 5, flat elsewhere
    prices = pd.DataFrame({"A": px})

    prescient = pd.DataFrame({"A": 0.0}, index=dates)
    prescient.iloc[5] = 1.0  # bets on the jump day itself

    result = PythonEngine().run_backtest(prices, prescient, {"cost_bps": 0})
    assert result.equity_curve.iloc[-1] == pytest.approx(1.0, abs=1e-12)  # earned nothing

    # Sanity: deciding the day BEFORE the jump (legitimately) does capture it.
    honest = pd.DataFrame({"A": 0.0}, index=dates)
    honest.iloc[4] = 1.0
    result = PythonEngine().run_backtest(prices, honest, {"cost_bps": 0})
    assert result.equity_curve.iloc[-1] == pytest.approx(1.1, abs=1e-12)


def test_hand_computed_two_asset_run_with_costs():
    # Fully hand-checked 5-day, 2-asset run pinning returns, costs, equity, and meta together.
    dates = pd.bdate_range("2022-01-03", periods=5)
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 121.0, 121.0], "B": [100.0, 100.0, 90.0, 90.0, 90.0]},
        index=dates,
    )
    positions = pd.DataFrame(
        {"A": [1.0, 0.0, 0.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0, 0.0, 0.0]}, index=dates
    )
    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 10})

    # held: day1 A=1 (earns A's +10%), day2 B=1 (earns B's -10%), else flat.
    # turnover: day1 |1-0|=1; day2 |0-1|+|1-0|=2; day3 |1-0|=1  -> total 4.
    # cost per unit turnover at 10 bps = 0.001.
    expected_net = pd.Series([0.0, 0.10 - 0.001, -0.10 - 0.002, -0.001, 0.0], index=dates)
    pd.testing.assert_series_equal(result.returns, expected_net, check_names=False)

    expected_equity = (1.0 + 0.099) * (1.0 - 0.102) * (1.0 - 0.001)
    assert result.equity_curve.iloc[-1] == pytest.approx(expected_equity, abs=1e-12)
    assert result.meta["total_turnover"] == pytest.approx(4.0, abs=1e-12)
    assert result.meta["n_days"] == 5


def test_valid_run_matches_documented_algorithm_exactly():
    # Generic guard for "no numerical change for valid inputs": re-derive net returns from the
    # 7 documented algorithm lines and demand exact equality with the engine's output.
    prices = _synthetic_prices(n=250, k=4, seed=7)
    positions = MomentumStrategy().generate_signals(prices, {"lookback": 40, "top_n": 2})
    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 25})

    aligned = positions.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)
    asset_returns = prices.pct_change().fillna(0.0)
    held = aligned.shift(1).fillna(0.0)
    gross = (held * asset_returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * (25.0 / 10_000.0)

    assert (result.returns - net).abs().max() == 0.0
    assert (result.equity_curve - (1.0 + net).cumprod()).abs().max() == 0.0
