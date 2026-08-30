"""Independent verifier tests for MeanReversionStrategy (week 4 milestone).

Adversarial checks beyond the smoke pattern:
- a fully hand-computed oscillation panel (lookback=2 gives the closed form z = ±sqrt(2)/2, so
  every entry/hold/exit row is derivable on paper, not read off the implementation);
- a sequential per-column loop oracle (independent re-implementation of the hysteresis spec)
  matched exactly against the vectorized output on noisy data with injected NaNs;
- entry-beats-exit precedence when the bands overlap (exit_z > entry_z);
- no-look-ahead via trailing-row truncation (earlier weights must be byte-identical);
- NaN prices forcing flat AND severing carried state.
"""

import numpy as np
import pandas as pd
import pytest

import quantforge.strategies.mean_reversion as mr_module
from quantforge.engine.python_engine import PythonEngine
from quantforge.strategies.mean_reversion import MeanReversionStrategy

# lookback=2 closed form: z_t = sign(P_t - P_{t-1}) * sqrt(2)/2 ~ +-0.7071 (NaN if delta = 0,
# because the 2-point sample std is 0 -> 0/0). entry_z=0.6 / exit_z=0.5 straddle 0.7071, so every
# down day is a long entry and every up day is a long exit (mirrored on the short side).
_HAND_PARAMS = {"lookback": 2, "entry_z": 0.6, "exit_z": 0.5}


def _hand_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=8)
    return pd.DataFrame(
        {
            # A deltas: -1, -1, +0.5, -0.5, 0, -1, +2  -> z: NaN,-,-,+,-,NaN,-,+
            "A": [100.0, 99.0, 98.0, 98.5, 98.0, 98.0, 97.0, 99.0],
            # B deltas: +1, -0.5, +0.5, -0.1, 0, -0.9, +1 -> z: NaN,+,-,+,-,NaN,-,+
            "B": [50.0, 51.0, 50.5, 51.0, 50.9, 50.9, 50.0, 51.0],
        },
        index=dates,
    )


def _noisy_panel(n: int = 300, k: int = 4, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, size=(n, k))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)),
        index=pd.bdate_range("2020-01-01", periods=n),
        columns=list("ABCD")[:k],
    )
    prices.iloc[50:53, 1] = np.nan  # a listed name going dark mid-sample
    prices.iloc[120, 3] = np.nan
    return prices


# ---------------------------------------------------------------------------
# Interface / done-when basics
# ---------------------------------------------------------------------------


def test_class_name_and_no_stub():
    strat = MeanReversionStrategy()
    assert strat.name == "mean_reversion"
    out = strat.generate_signals(_hand_panel(), None)  # must not raise NotImplementedError
    assert isinstance(out, pd.DataFrame)


def test_shape_index_columns_and_default_warmup():
    prices = _noisy_panel()
    weights = MeanReversionStrategy().generate_signals(prices)  # defaults: lookback=20
    assert weights.shape == prices.shape
    assert weights.index.equals(prices.index)
    assert list(weights.columns) == list(prices.columns)
    # First lookback-1 rows have no full window -> flat.
    assert (weights.iloc[:19] == 0.0).all().all()
    assert not weights.isna().any().any()


@pytest.mark.parametrize("mode", ["long_flat", "long_short"])
def test_gross_exposure_within_engine_tolerance(mode):
    prices = _noisy_panel()
    weights = MeanReversionStrategy().generate_signals(
        prices, {"lookback": 5, "entry_z": 1.2, "exit_z": 0.3, "mode": mode}
    )
    gross = weights.abs().sum(axis=1)
    assert (gross <= 1.0 + 1e-9).all()
    # Active rows are fully invested (gross == 1 up to float noise), inactive rows all-zero.
    active = (weights != 0.0).any(axis=1)
    assert active.any(), "test panel produced no trades - thresholds too tight to prove anything"
    assert np.allclose(gross[active], 1.0)
    assert (gross[~active] == 0.0).all()
    if mode == "long_flat":
        assert (weights >= 0.0).all().all()


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        MeanReversionStrategy().generate_signals(_hand_panel(), {"mode": "long_leveraged"})


def test_module_docstring_states_short_side_frictions():
    doc = mr_module.__doc__.lower()
    for friction in ("borrow", "locate", "squeeze"):
        assert friction in doc, f"honesty caveat missing {friction!r} in module docstring"


# ---------------------------------------------------------------------------
# Hand-computed entry / hold / exit behavior (both modes)
# ---------------------------------------------------------------------------


def test_hand_built_long_flat_entries_holds_exits():
    prices = _hand_panel()
    weights = MeanReversionStrategy().generate_signals(prices, {**_HAND_PARAMS})
    expected = pd.DataFrame(
        {
            # A: warmup, enter on first z<-0.6 (t1), still-entered t2, exit on first z>-0.5 (t3),
            # re-enter t4, forced flat on NaN-z (zero-delta) t5, enter t6, exit t7.
            "A": [0.0, 1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0],
            # B enters t2/t4/t6 alongside A -> equal weight 1/2 on shared rows.
            "B": [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0],
        },
        index=prices.index,
    )
    pd.testing.assert_frame_equal(weights, expected)


def test_hand_built_long_short_symmetric_shorts():
    prices = _hand_panel()
    weights = MeanReversionStrategy().generate_signals(
        prices, {**_HAND_PARAMS, "mode": "long_short"}
    )
    expected = pd.DataFrame(
        {
            # Up days (z=+0.7071 > entry_z) open shorts; down days cover and go long.
            "A": [0.0, 0.5, 0.5, -0.5, 0.5, 0.0, 0.5, -0.5],
            "B": [0.0, -0.5, 0.5, -0.5, 0.5, 0.0, 0.5, -0.5],
        },
        index=prices.index,
    )
    pd.testing.assert_frame_equal(weights, expected)
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-9).all()


def test_entry_wins_when_bands_overlap():
    # exit_z > entry_z makes entry and exit true simultaneously on a down day
    # (z=-0.7071 < -0.6 AND > -1.5). Documented precedence: entry wins -> long, not flat.
    prices = pd.DataFrame(
        {"A": [100.0, 99.0, 100.0]}, index=pd.bdate_range("2024-01-01", periods=3)
    )
    weights = MeanReversionStrategy().generate_signals(
        prices, {"lookback": 2, "entry_z": 0.6, "exit_z": 1.5}
    )
    assert weights["A"].tolist() == [0.0, 1.0, 0.0]


def test_nan_price_forces_flat_and_severs_state():
    # Long from t1; NaN price at t2 poisons the t2 and t3 windows -> flat both rows, and the
    # position must NOT resume by itself after the gap - t4 is long only via a fresh entry.
    prices = pd.DataFrame(
        {"A": [100.0, 99.0, np.nan, 97.0, 96.0]},
        index=pd.bdate_range("2024-01-01", periods=5),
    )
    weights = MeanReversionStrategy().generate_signals(prices, {**_HAND_PARAMS})
    assert weights["A"].tolist() == [0.0, 1.0, 0.0, 0.0, 1.0]


def test_constant_prices_stay_flat():
    # Zero rolling std -> z = 0/0 = NaN -> the NaN policy keeps a flat book (no divide blowups).
    prices = pd.DataFrame(
        {"A": [100.0] * 30, "B": [55.0] * 30}, index=pd.bdate_range("2024-01-01", periods=30)
    )
    for mode in ("long_flat", "long_short"):
        weights = MeanReversionStrategy().generate_signals(
            prices, {"lookback": 5, "entry_z": 1.0, "exit_z": 0.2, "mode": mode}
        )
        assert (weights == 0.0).all().all()


# ---------------------------------------------------------------------------
# Adversarial oracle: independent sequential re-implementation of the spec
# ---------------------------------------------------------------------------


def _loop_side(z_col, entry_fn, exit_fn):
    """One-sided hysteresis as a plain Python loop (the spec, executed literally)."""
    state, out = 0.0, []
    for zv in z_col:
        if pd.isna(zv):
            state = 0.0  # NaN forces flat and severs state
        else:
            if exit_fn(zv):
                state = 0.0
            if entry_fn(zv):  # checked after exit -> entry wins overlaps
                state = 1.0
        out.append(state)
    return out


def _oracle(prices, lookback, entry_z, exit_z, mode):
    prices = prices.sort_index()
    roll = prices.rolling(window=lookback, min_periods=lookback)
    z = (prices - roll.mean()) / roll.std()
    state = pd.DataFrame(0.0, index=z.index, columns=z.columns)
    for col in z.columns:
        long_leg = _loop_side(z[col], lambda v: v < -entry_z, lambda v: v > -exit_z)
        state[col] = long_leg
        if mode == "long_short":
            short_leg = _loop_side(z[col], lambda v: v > entry_z, lambda v: v < exit_z)
            state[col] = state[col] - pd.Series(short_leg, index=z.index)
    n_active = (state != 0.0).sum(axis=1)
    return state.div(n_active.where(n_active != 0), axis=0).fillna(0.0)


@pytest.mark.parametrize("mode", ["long_flat", "long_short"])
def test_vectorized_matches_sequential_oracle(mode):
    prices = _noisy_panel()
    params = {"lookback": 5, "entry_z": 1.2, "exit_z": 0.3, "mode": mode}
    got = MeanReversionStrategy().generate_signals(prices, params)
    expected = _oracle(prices, 5, 1.2, 0.3, mode)
    assert (got != 0.0).any(axis=1).sum() > 50  # the comparison actually exercises trades
    pd.testing.assert_frame_equal(got, expected)


# ---------------------------------------------------------------------------
# Rigor: no look-ahead, no input mutation, engine compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["long_flat", "long_short"])
def test_no_lookahead_truncating_future_leaves_past_identical(mode):
    prices = _noisy_panel()
    params = {"lookback": 5, "entry_z": 1.2, "exit_z": 0.3, "mode": mode}
    strat = MeanReversionStrategy()
    full = strat.generate_signals(prices, params)
    for k in (1, 25):
        truncated = strat.generate_signals(prices.iloc[:-k], params)
        pd.testing.assert_frame_equal(full.iloc[:-k], truncated)


def test_prices_not_mutated_and_unsorted_input_handled():
    prices = _noisy_panel().iloc[::-1]  # reverse order: strategy must sort, not choke
    before = prices.copy(deep=True)
    weights = MeanReversionStrategy().generate_signals(
        prices, {"lookback": 5, "entry_z": 1.2, "exit_z": 0.3}
    )
    pd.testing.assert_frame_equal(prices, before)  # caller's frame untouched
    assert weights.index.is_monotonic_increasing
    # Same answer as sorted input: order of arrival must not change the signal.
    sorted_weights = MeanReversionStrategy().generate_signals(
        prices.sort_index(), {"lookback": 5, "entry_z": 1.2, "exit_z": 0.3}
    )
    pd.testing.assert_frame_equal(weights, sorted_weights)


@pytest.mark.parametrize("mode", ["long_flat", "long_short"])
def test_weights_accepted_by_python_engine(mode):
    prices = _noisy_panel()
    weights = MeanReversionStrategy().generate_signals(
        prices, {"lookback": 5, "entry_z": 1.2, "exit_z": 0.3, "mode": mode}
    )
    result = PythonEngine().run_backtest(prices, weights, {"cost_bps": 10})  # must not raise
    assert len(result.equity_curve) == len(prices)
    assert not result.returns.isna().any()
