"""Independent verifier tests for RG-2 cost accounting (offline, deterministic).

Written by the milestone verifier, deliberately NOT sharing numbers with
tests/test_cost_accounting.py: a fresh hand-computed 2-asset/4-day book plus two adversarial
cases the builder's file does not cover explicitly —

1. A prescient final-day trade must be FREE and EARN NOTHING: a target decided on the last
   close is never held (shift(1) pushes it off the end), so it can neither pay turnover cost
   nor collect return. An engine charging on same-day ``positions.diff()`` would bill it.
2. Rotation turnover must sum absolute per-asset deltas, not net them: moving 0.5 from A to B
   is 1.0 of turnover, not 0. An engine netting signed deltas (``diff().sum().abs()``) would
   charge nothing for the swap.

All expected numbers below are derived by hand in comments from the literal inputs.
"""

import numpy as np
import pandas as pd

from quantforge.engine.python_engine import PythonEngine

ATOL = 1e-12


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    dates = pd.bdate_range("2023-06-01", periods=len(rows))
    return pd.DataFrame(rows, index=dates, columns=["A", "B"])


def test_verifier_hand_computed_two_asset_four_day_case():
    """A second fully hand-computed book, independent of the builder's numbers.

    Prices  A: 80 -> 100 -> 90 -> 108      ret_A = [0, 0.25, -0.10, 0.20]
            B: 25 ->  20 -> 22 ->  11      ret_B = [0, -0.20, 0.10, -0.50]
    Targets d0 (0.4, -0.6)  d1 (-0.25, 0.25)  d2 (0.5, 0.5)  d3 (0, 0)   (gross <= 1 always)
    held = shift(1): d0 (0,0)  d1 (0.4,-0.6)  d2 (-0.25,0.25)  d3 (0.5,0.5)

    gross: d1 0.4*0.25 + (-0.6)*(-0.20)      = 0.10 + 0.12   =  0.22
           d2 -0.25*(-0.10) + 0.25*0.10      = 0.025 + 0.025 =  0.05
           d3 0.5*0.20 + 0.5*(-0.50)         = 0.10 - 0.25   = -0.15
    turnover: d1 0.4+0.6 = 1.0
              d2 |-0.25-0.4| + |0.25-(-0.6)| = 0.65 + 0.85 = 1.5
              d3 |0.5-(-0.25)| + |0.5-0.25|  = 0.75 + 0.25 = 1.0        total = 3.5
    costs at 50 bps (rate 0.005): [0, 0.005, 0.0075, 0.005]
    net = gross - costs:          [0, 0.215, 0.0425, -0.155]
    equity = cumprod(1+net):      [1, 1.215, 1.215*1.0425, 1.215*1.0425*0.845]
    """
    prices = _frame([[80.0, 25.0], [100.0, 20.0], [90.0, 22.0], [108.0, 11.0]])
    positions = _frame([[0.4, -0.6], [-0.25, 0.25], [0.5, 0.5], [0.0, 0.0]])

    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 50})

    expected_net = [0.0, 0.215, 0.0425, -0.155]
    expected_equity = [1.0, 1.215, 1.215 * 1.0425, 1.215 * 1.0425 * 0.845]
    np.testing.assert_allclose(result.returns.to_numpy(), expected_net, rtol=0, atol=ATOL)
    np.testing.assert_allclose(result.equity_curve.to_numpy(), expected_equity, rtol=0, atol=ATOL)
    assert abs(result.meta["total_turnover"] - 3.5) < ATOL


def test_prescient_final_day_trade_is_free_and_earns_nothing():
    """Adversarial: two books identical except for a wild final-day target must backtest identically.

    The final-day decision is never held, so it must contribute zero return AND zero cost —
    same net, equity, and total_turnover. This distinguishes held.diff() (correct) from
    positions.diff() (which would charge the ghost trade) and doubles as a no-look-ahead
    check on the cost leg.
    """
    prices = _frame([[10.0, 10.0], [12.0, 8.0], [9.0, 11.0], [15.0, 5.0]])
    base = _frame([[0.3, 0.3], [0.3, 0.3], [0.3, 0.3], [0.3, 0.3]])
    prescient = base.copy()
    prescient.iloc[-1] = [1.0, -0.0]  # all-in on the final close — unknowable, and unheld

    engine = PythonEngine()
    r_base = engine.run_backtest(prices, base, {"cost_bps": 25})
    r_prescient = engine.run_backtest(prices, prescient, {"cost_bps": 25})

    assert (r_base.returns.to_numpy() == r_prescient.returns.to_numpy()).all()
    assert (r_base.equity_curve.to_numpy() == r_prescient.equity_curve.to_numpy()).all()
    assert r_base.meta["total_turnover"] == r_prescient.meta["total_turnover"]


def test_rotation_turnover_sums_absolute_deltas_not_net_exposure():
    """Adversarial: rotating 0.5 from A into B keeps net exposure flat but is 1.0 of turnover.

    Constant prices isolate the cost leg (gross == 0, so net == -costs). Targets:
        d0 (0.5, 0)  d1 (0, 0.5)  d2 (0, 0.5)
    held: d0 (0,0)  d1 (0.5,0)  d2 (0,0.5)
    turnover: d1 0.5 (entry)   d2 |0-0.5| + |0.5-0| = 1.0 (the rotation)   total 1.5
    costs at 100 bps (rate 0.01): net = [0, -0.005, -0.010]. An engine netting signed deltas
    would see d2 turnover of 0 and charge nothing for the swap.
    """
    prices = _frame([[100.0, 40.0]] * 3)
    positions = _frame([[0.5, 0.0], [0.0, 0.5], [0.0, 0.5]])

    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 100})
    np.testing.assert_allclose(result.returns.to_numpy(), [0.0, -0.005, -0.010], rtol=0, atol=ATOL)
    assert abs(result.meta["total_turnover"] - 1.5) < ATOL
