"""Independent verifier tests for the week-4 milestone (tests/test_strategies.py).

These deliberately do NOT reuse the builder's expectations: every number below is derived by
literal fraction arithmetic in comments, so agreement with the pipeline is evidence about the
math, not about two reads of the same constants.

Adversarial angle: the entry-triggering crash itself must earn exactly nothing. On the
hand-built series MR drops 5% on the very row whose z-score fires the long entry — a
look-ahead bug (missing/reversed shift, or a signal peeking at same-day data) would either
capture that -5% or the +2.1% bounce a day early. The engine-level tests here pin the *entire*
net-return vector of the strategy->engine chain to hand-computed fractions, cost included, in
both modes.

All offline/synthetic. No network, no yfinance.
"""

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.python_engine import PythonEngine
from quantforge.strategies import PARAM_WHITELIST, STRATEGIES, validate_params
from quantforge.strategies.mean_reversion import MeanReversionStrategy
from quantforge.strategies.momentum import MomentumStrategy

# Same hand-built series as tests/test_strategies.py (lookback=5, entry_z=1.5, exit_z=0.5):
# MR enters long at row 4 (z = -4/sqrt(5) ~ -1.789), holds rows 5-6 (z in the hysteresis band),
# exits row 7; OB is the short-side mirror in long_short mode. Re-derived here, not imported,
# so a regression in the builder's file cannot silently blind this one.
_PARAMS = {"lookback": 5, "entry_z": 1.5, "exit_z": 0.5}


def _hand_built_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MR": [100.0, 100.0, 100.0, 100.0, 95.0, 97.0, 96.0, 99.0, 95.0, 97.0],
            "OB": [200.0, 200.0, 200.0, 200.0, 210.0, 206.0, 208.0, 202.0, 205.0, 205.0],
        },
        index=pd.bdate_range("2024-01-01", periods=10),
    )


# ---------------------------------------------------------------------------
# Adversarial: hand-computed net returns through the REAL engine, cost included
# ---------------------------------------------------------------------------


def test_long_flat_net_returns_hand_computed_through_engine():
    """Entire net-return vector of strategy->engine pinned to literal fractions (cost_bps=10).

    Weights (from the hand-built design): MR = 1.0 on rows 4-6, else 0; OB always 0.
    held = weights.shift(1) -> MR held on rows 5-7 only.

    Asset returns for MR:  r5 = 97/95 - 1 = 2/95,  r6 = 96/97 - 1 = -1/97,  r7 = 99/96 - 1 = 1/32.
    Turnover on held.diff(): row 5 = |1-0| = 1, row 8 = |0-1| = 1; cost = 1 * 10/10000 = 0.001.

    net = [0, 0, 0, 0, 0, 2/95 - 0.001, -1/97, 1/32, -0.001, 0]
    """
    prices = _hand_built_prices()
    weights = MeanReversionStrategy().generate_signals(prices, {**_PARAMS, "mode": "long_flat"})
    result = PythonEngine().run_backtest(prices, weights, {"cost_bps": 10})

    expected_net = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2 / 95 - 0.001, -1 / 97, 1 / 32, -0.001, 0.0])
    np.testing.assert_allclose(result.returns.to_numpy(), expected_net, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        result.equity_curve.to_numpy(), np.cumprod(1.0 + expected_net), rtol=1e-12
    )
    assert result.meta["total_turnover"] == pytest.approx(2.0, rel=1e-12)  # enter + exit


def test_long_short_net_returns_hand_computed_through_engine():
    """Same pinning for long_short: the short leg's sign and its cost are hand-derived.

    Weights rows 4-6: MR +0.5, OB -0.5; held rows 5-7. OB asset returns:
      r5 = 206/210 - 1 = -2/105, r6 = 208/206 - 1 = 1/103, r7 = 202/208 - 1 = -3/104.
    Gross: row5 = 0.5*(2/95) - 0.5*(-2/105) = 1/95 + 1/105
           row6 = 0.5*(-1/97) - 0.5*(1/103)  = -1/194 - 1/206
           row7 = 0.5*(1/32)  - 0.5*(-3/104) = 1/64 + 3/208
    Turnover: row5 = 0.5 + 0.5 = 1, row8 = 1 -> cost 0.001 each.
    """
    prices = _hand_built_prices()
    weights = MeanReversionStrategy().generate_signals(prices, {**_PARAMS, "mode": "long_short"})
    result = PythonEngine().run_backtest(prices, weights, {"cost_bps": 10})

    expected_net = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1 / 95 + 1 / 105 - 0.001,
            -1 / 194 - 1 / 206,
            1 / 64 + 3 / 208,
            -0.001,
            0.0,
        ]
    )
    np.testing.assert_allclose(result.returns.to_numpy(), expected_net, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("mode", ["long_flat", "long_short"])
def test_entry_day_crash_is_never_captured(mode):
    """The -5% MR move that FIRES the entry earns exactly zero — the anti-look-ahead teeth.

    Row 4 is where z crosses -entry_z, and it is also the series' biggest single-day move
    (95/100 - 1 = -5%). A prescient chain (signal on same-day data, or a missing/reversed
    shift in the engine) would post a large nonzero return on row 4 — long_flat would eat the
    -5%, and a sign-flipped book would bank +5%. The honest chain holds nothing yet: 0.0 exact.
    """
    prices = _hand_built_prices()
    weights = MeanReversionStrategy().generate_signals(prices, {**_PARAMS, "mode": mode})
    result = PythonEngine().run_backtest(prices, weights, {"cost_bps": 10})
    assert result.returns.iloc[4] == 0.0


# ---------------------------------------------------------------------------
# validate_params: boundary acceptance + registry/whitelist coherence
# ---------------------------------------------------------------------------


def test_whitelist_bounds_are_inclusive():
    """Documented bounds are [min, max] inclusive: the exact endpoints must be ACCEPTED.

    The builder's suite proves 4 and 3.5 are rejected; an off-by-one gate (< instead of <=)
    would also reject the legal endpoints, which only this test would catch.
    """
    for param, lo, hi in [("lookback", 5, 60), ("entry_z", 0.5, 3.0), ("exit_z", 0.0, 1.5)]:
        assert validate_params("mean_reversion", {param: lo})[param] == lo
        assert validate_params("mean_reversion", {param: hi})[param] == hi


def test_float_ints_coerced_not_rejected():
    """JSON round-trips float-ify ints: 20.0 must validate as lookback 20 (a real int)."""
    out = validate_params("mean_reversion", {"lookback": 20.0})
    assert out["lookback"] == 20
    assert isinstance(out["lookback"], int) and not isinstance(out["lookback"], bool)
    with pytest.raises(ValueError, match="lookback"):
        validate_params("mean_reversion", {"lookback": 20.5})  # fractional stays rejected


def test_registry_and_whitelist_stay_in_lockstep():
    """Every vetted strategy has a whitelist entry and vice versa (the SF-3 gate's two halves)."""
    assert set(STRATEGIES) == set(PARAM_WHITELIST)
    assert STRATEGIES["mean_reversion"] is MeanReversionStrategy


def test_validated_defaults_equal_inline_defaults():
    """'No params' and 'validated empty params' must mean the SAME backtest, byte for byte.

    The registry docstring promises its defaults are identical to the strategies' inline
    defaults; if either side drifts, the AI/MCP path (always validated) and the direct path
    (often None) would silently run different strategies under one name.
    """
    rng = np.random.default_rng(29)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, size=(120, 3)), axis=0)),
        index=pd.bdate_range("2023-01-02", periods=120),
        columns=["A", "B", "C"],
    )
    for name, cls in [("mean_reversion", MeanReversionStrategy), ("momentum", MomentumStrategy)]:
        via_registry = cls().generate_signals(prices, validate_params(name, {}))
        via_inline = cls().generate_signals(prices, None)
        pd.testing.assert_frame_equal(via_registry, via_inline, check_exact=True)
