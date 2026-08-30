"""Verifier tests for the week-4 interface freeze (adversarial companion to test_interface_freeze).

WHY a second file: test_interface_freeze.py pins the mechanical surface (fields, signatures,
ABC-ness). This file pins the *documented semantics* the freeze is supposed to lock in — the
frozen-declaration wording itself, the close-of-t timing convention (no stale "t-1" contradiction),
and, behaviorally, that the engine-owned one-day lag actually does what the frozen docstrings say:
a signal that peeks at same-day returns gains nothing, and a hand-computed equity curve comes out
exactly as the contract predicts.
"""

import numpy as np
import pandas as pd
import pytest

import quantforge.engine.base as base
from quantforge.engine.base import BacktestResult, Engine, Strategy
from quantforge.engine.python_engine import PythonEngine
from quantforge.metrics.performance import _KEYS

# ---------------------------------------------------------------------------
# The frozen declaration and timing convention, as written
# ---------------------------------------------------------------------------


def test_module_docstring_declares_week4_freeze():
    doc = base.__doc__
    assert doc is not None
    assert "FROZEN" in doc
    assert "week 4" in doc
    # AR-1 breaking-change rule and AR-4 "next engine drops in here" must both be stated.
    # (Normalize whitespace: the phrase may wrap across docstring lines.)
    flat = " ".join(doc.split())
    assert "BREAKING CHANGE" in flat.upper()
    assert "next engine drops in here" in flat


def test_generate_signals_docstring_says_close_of_t_with_no_stale_tminus1():
    # The week-4 correction: decision uses info through the CLOSE of t; the one-day lag is the
    # ENGINE's job. The old "close of t-1" wording contradicted the engine and must be gone.
    combined = (Strategy.__doc__ or "") + (Strategy.generate_signals.__doc__ or "")
    assert "close of t" in combined.lower()
    for stale in ("t-1", "t−1"):  # ASCII hyphen and Unicode minus spellings
        assert stale not in combined, f"stale '{stale}' decision-timing wording still present"


def test_no_todo_markers_left_in_frozen_docstrings():
    for doc in (
        base.__doc__,
        BacktestResult.__doc__,
        Strategy.__doc__,
        Strategy.generate_signals.__doc__,
        Engine.__doc__,
        Engine.run_backtest.__doc__,
    ):
        assert doc is not None
        assert "TODO" not in doc


# ---------------------------------------------------------------------------
# The frozen semantics, behaviorally (hand-computed, deterministic)
# ---------------------------------------------------------------------------


def test_engine_lag_hand_computed_single_asset():
    # One asset, +10% a day. Full weight from day 0, zero costs. The contract says a weight
    # decided at close of t earns t+1's return, so day 0 earns nothing and the curve is
    # exactly [1.0, 1.1, 1.21, 1.331] — including the "starts at 1.0" pin from the
    # BacktestResult docstring.
    dates = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]}, index=dates)
    positions = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=dates)

    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 0})

    expected = [1.0, 1.1, 1.21, 1.331]
    assert np.allclose(result.equity_curve.to_numpy(), expected, rtol=0, atol=1e-12)
    assert result.equity_curve.iloc[0] == 1.0


def test_prescient_same_day_peek_earns_the_loss_not_the_win():
    # Adversarial: two assets whose daily returns strictly alternate — on each day one is +10%
    # and the other -10%, swapping every day. A cheating signal holds the SAME-DAY winner
    # (information the contract forbids acting on without the lag). If the engine failed to
    # shift, this book would print +10% every day. With the frozen engine-owned lag, the held
    # weight is yesterday's winner, which loses today — so every post-warmup day returns
    # exactly -10%. The lag doesn't just damp the cheat, it inverts it.
    n = 6
    dates = pd.bdate_range("2024-01-01", periods=n)
    ret_a = np.array([0.10 if t % 2 == 0 else -0.10 for t in range(n)])
    ret_b = -ret_a
    prices = pd.DataFrame(
        {
            "A": 100.0 * np.cumprod(1.0 + np.concatenate([[0.0], ret_a[1:]])),
            "B": 100.0 * np.cumprod(1.0 + np.concatenate([[0.0], ret_b[1:]])),
        },
        index=dates,
    )
    # Day t holds whichever asset wins on day t itself (the peek).
    peek = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    for t in range(1, n):
        peek.iloc[t, 0 if ret_a[t] > 0 else 1] = 1.0

    result = PythonEngine().run_backtest(prices, peek, {"cost_bps": 0})

    # Days 0-1 are flat (nothing held yet); every day from 2 on realizes the -10% loss.
    net = result.returns.to_numpy()
    assert np.allclose(net[:2], 0.0, atol=1e-12)
    assert np.allclose(net[2:], -0.10, atol=1e-12)


def test_result_metrics_keys_match_single_source_of_truth():
    # The frozen BacktestResult docstring pins metrics keys to metrics/performance._KEYS.
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": np.linspace(100.0, 104.0, 5)}, index=dates)
    positions = pd.DataFrame({"A": [0.5] * 5}, index=dates)
    result = PythonEngine().run_backtest(prices, positions, {})
    assert list(result.metrics.keys()) == list(_KEYS)


def test_engine_does_not_mutate_inputs():
    # The frozen run_backtest docstring promises no input mutation — callers may reuse frames
    # across engines. Feed unsorted-column, NaN-bearing positions and confirm both frames come
    # back byte-identical.
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0, 104.0], "B": [50.0, 49.0, 50.0, 51.0, 50.5]},
        index=dates,
    )
    positions = pd.DataFrame(
        {"B": [0.5, np.nan, 0.5, 0.5, 0.5], "A": [0.5, 0.5, np.nan, 0.5, 0.5]}, index=dates
    )
    prices_before, positions_before = prices.copy(deep=True), positions.copy(deep=True)

    PythonEngine().run_backtest(prices, positions, {"cost_bps": 10})

    pd.testing.assert_frame_equal(prices, prices_before)
    pd.testing.assert_frame_equal(positions, positions_before)


def test_seam_rejects_direct_use_with_typeerror_mentioning_abstract():
    # Sharper than "raises TypeError": the error must be the ABC guard, not some other TypeError.
    with pytest.raises(TypeError, match="abstract"):
        Strategy()
    with pytest.raises(TypeError, match="abstract"):
        Engine()
