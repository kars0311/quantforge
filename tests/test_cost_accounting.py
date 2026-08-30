"""RG-2 — transaction-cost accounting proof (offline, deterministic, hand-computed).

Why this file exists: the engine's cost model is four lines of pandas, which makes it easy to
*look* right while charging the wrong day, the wrong magnitude, or forgetting the entry trade.
Every number the main test asserts against is derived BY HAND in comments from literal prices
and weights — not copied from engine output — so any change to the cost arithmetic (halving
costs, charging on positions instead of held, dropping the entry-day charge) breaks a test
whose expected values a human can re-derive on paper.

Engine conventions under test (src/quantforge/engine/python_engine.py):
    asset_returns = prices.pct_change().fillna(0.0)
    held          = positions.shift(1).fillna(0.0)     # decided at t, applied at t+1
    gross[t]      = sum_i held[t, i] * asset_returns[t, i]
    turnover[t]   = sum_i |held[t, i] - held[t-1, i]|  (day 0: NaN diff -> fillna(0.0))
    costs[t]      = turnover[t] * cost_bps / 10_000
    net           = gross - costs
    equity        = cumprod(1 + net)

Default-cost note: the ENGINE default is cost_bps=0.0 (asserted below). The product-level
default of 10 bps (product.md section 5.3) is a caller decision and is deliberately NOT baked
into engine behavior — the engine must not charge costs nobody asked for.
"""

import numpy as np
import pandas as pd

from quantforge.engine.python_engine import PythonEngine

ATOL = 1e-12


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    """Wide 2-column (A, B) frame on a business-day index — keeps the literal numbers readable."""
    dates = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame(rows, index=dates, columns=["A", "B"])


# ---------------------------------------------------------------------------------------------
# The fully hand-computed 2-asset, 5-day case. Every expected series below is derived in the
# comments from these literals; the verifier should be able to redo the arithmetic on paper.
# ---------------------------------------------------------------------------------------------

#           A       B
PRICES = _frame(
    [
        [100.0, 50.0],  # d0
        [110.0, 50.0],  # d1
        [99.0, 60.0],  # d2
        [108.9, 45.0],  # d3
        [108.9, 54.0],  # d4
    ]
)

#             A     B         gross exposure |A|+|B|
POSITIONS = _frame(
    [
        [0.6, 0.4],  # d0   1.0  (long-long)
        [0.5, -0.5],  # d1   1.0  (long-short)
        [0.0, 0.0],  # d2   0.0  (flat)
        [-0.3, 0.3],  # d3   0.6  (short-long)
        [0.2, 0.2],  # d4   0.4  (never held: shift(1) pushes it past the last day)
    ]
)

# Asset returns (pct_change, day 0 NaN -> 0):
#   ret_A: d0 0,  d1 110/100-1 = 0.10,  d2 99/110-1 = -0.10,  d3 108.9/99-1 = 0.10,  d4 0
#   ret_B: d0 0,  d1 50/50-1   = 0.00,  d2 60/50-1  =  0.20,  d3 45/60-1  = -0.25,  d4 54/45-1 = 0.20
#
# held = positions.shift(1) (day 0 -> 0):
#   d0 (0, 0)   d1 (0.6, 0.4)   d2 (0.5, -0.5)   d3 (0, 0)   d4 (-0.3, 0.3)
#
# gross[t] = held_A*ret_A + held_B*ret_B:
#   d0: 0
#   d1: 0.6*0.10  + 0.4*0.00     =  0.06
#   d2: 0.5*(-0.10) + (-0.5)*0.20 = -0.05 - 0.10 = -0.15
#   d3: 0*0.10    + 0*(-0.25)    =  0.00
#   d4: -0.3*0.00 + 0.3*0.20     =  0.06
EXPECTED_GROSS = [0.0, 0.06, -0.15, 0.0, 0.06]

# turnover[t] = |held_A[t]-held_A[t-1]| + |held_B[t]-held_B[t-1]| (day 0 -> 0):
#   d0: 0
#   d1: |0.6-0|    + |0.4-0|      = 1.0    (entering the initial book IS charged)
#   d2: |0.5-0.6|  + |-0.5-0.4|   = 0.1 + 0.9 = 1.0
#   d3: |0-0.5|    + |0-(-0.5)|   = 0.5 + 0.5 = 1.0
#   d4: |-0.3-0|   + |0.3-0|      = 0.3 + 0.3 = 0.6
EXPECTED_TURNOVER = [0.0, 1.0, 1.0, 1.0, 0.6]
EXPECTED_TOTAL_TURNOVER = 3.6  # 0 + 1.0 + 1.0 + 1.0 + 0.6

# costs at cost_bps=25: turnover * 25/10_000 = turnover * 0.0025:
#   d0: 0   d1: 0.0025   d2: 0.0025   d3: 0.0025   d4: 0.6*0.0025 = 0.0015
COST_BPS = 25
EXPECTED_COSTS = [0.0, 0.0025, 0.0025, 0.0025, 0.0015]

# net = gross - costs:
#   d0: 0
#   d1:  0.06 - 0.0025 =  0.0575
#   d2: -0.15 - 0.0025 = -0.1525
#   d3:  0.00 - 0.0025 = -0.0025
#   d4:  0.06 - 0.0015 =  0.0585
EXPECTED_NET = [0.0, 0.0575, -0.1525, -0.0025, 0.0585]

# equity = cumprod(1 + net):
#   d0: 1.0
#   d1: 1.0575
#   d2: 1.0575 * 0.8475  = 0.89623125
#   d3:      ... * 0.9975 = 0.893990671875
#   d4:      ... * 1.0585 = 0.9462891261796875
EXPECTED_EQUITY = [
    1.0,
    1.0575,
    1.0575 * 0.8475,
    1.0575 * 0.8475 * 0.9975,
    1.0575 * 0.8475 * 0.9975 * 1.0585,
]


def _run(cost_bps: float | None = None):
    params = {} if cost_bps is None else {"cost_bps": cost_bps}
    return PythonEngine().run_backtest(PRICES, POSITIONS, params)


def test_hand_computed_case_net_equity_and_turnover():
    """The 2-asset/5-day case: engine output matches paper arithmetic element-wise to 1e-12."""
    result = _run(COST_BPS)
    np.testing.assert_allclose(result.returns.to_numpy(), EXPECTED_NET, rtol=0, atol=ATOL)
    np.testing.assert_allclose(result.equity_curve.to_numpy(), EXPECTED_EQUITY, rtol=0, atol=ATOL)
    assert abs(result.meta["total_turnover"] - EXPECTED_TOTAL_TURNOVER) < ATOL
    assert result.meta["cost_bps"] == float(COST_BPS)


def test_hand_computed_case_gross_and_cost_series():
    """Split net into its two ingredients and check each against the hand numbers.

    The engine only exposes net returns, so gross comes from a zero-cost run of the same
    inputs and the per-day cost series is the difference of the two runs — both are then
    pinned to the hand-derived tables, which also fixes the implied turnover series
    (costs / 0.0025) without re-implementing the engine's diff arithmetic in the test.
    """
    net_costed = _run(COST_BPS).returns.to_numpy()
    gross = _run(0).returns.to_numpy()
    np.testing.assert_allclose(gross, EXPECTED_GROSS, rtol=0, atol=ATOL)
    np.testing.assert_allclose(gross - net_costed, EXPECTED_COSTS, rtol=0, atol=ATOL)
    # implied per-day turnover = costs / (25 bps) — ties the cost series back to |delta held|
    np.testing.assert_allclose(
        (gross - net_costed) / (COST_BPS / 10_000.0), EXPECTED_TURNOVER, rtol=0, atol=1e-8
    )


def test_zero_cost_default_net_equals_gross_exactly():
    """cost_bps=0 (the engine DEFAULT) must be a true no-op: net == gross bit-for-bit.

    Uses seeded random data (not the tiny hand case) so the exactness claim covers hundreds
    of ordinary float values, and recomputes gross independently from prices/positions rather
    than trusting any engine intermediate. Also pins the documented default cost_bps=0.0 —
    the product-level 10 bps default belongs to callers, not the engine.
    """
    rng = np.random.default_rng(7)
    n, k = 250, 4
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(n, k)), axis=0)),
        index=dates,
        columns=[f"A{i}" for i in range(k)],
    )
    raw = rng.normal(size=(n, k))
    weights = raw / np.abs(raw).sum(axis=1, keepdims=True) * 0.9  # gross exposure 0.9 < 1
    positions = pd.DataFrame(weights, index=dates, columns=prices.columns)

    result = PythonEngine().run_backtest(prices, positions)  # no params: documented default
    assert result.meta["cost_bps"] == 0.0

    gross = (positions.shift(1).fillna(0.0) * prices.pct_change().fillna(0.0)).sum(axis=1)
    assert (result.returns.to_numpy() == gross.to_numpy()).all()  # exact, not approx
    assert (result.equity_curve.to_numpy() == (1.0 + gross).cumprod().to_numpy()).all()


def test_cost_deductions_scale_linearly_in_cost_bps():
    """gross - net must be exactly linear in cost_bps: deductions at 20 bps = 2x those at 10.

    Costs are turnover * cost_bps/10_000 and turnover does not depend on cost_bps, so
    doubling the rate must exactly double every per-day deduction. A fixed per-trade fee,
    a rounding step, or a cost floor would all break this.
    """
    gross = _run(0).returns.to_numpy()
    deduct_10 = gross - _run(10).returns.to_numpy()
    deduct_20 = gross - _run(20).returns.to_numpy()
    assert deduct_10.sum() > 0  # sanity: costs actually charged (turnover is nonzero)
    np.testing.assert_allclose(deduct_20, 2.0 * deduct_10, rtol=0, atol=ATOL)


def test_buy_sell_and_short_entry_cost_the_same():
    """Turnover charges |delta held|: a buy, a sell, and a short entry of size 0.7 cost equally.

    Prices are constant so gross is identically 0 and net = -costs, isolating the cost leg.
    Three mini-books each make one 0.7-sized transition:
      buy:   held 0 -> +0.7 on d2   (positions [0, .7, .7, .7] shifted)
      sell:  held +0.7 -> 0 on d3   (positions [.7, .7, 0, 0] shifted; d1 entry is separate)
      short: held 0 -> -0.7 on d2   (positions [0, -.7, -.7, -.7] shifted)
    Each transition day must be charged exactly 0.7 * 25/10_000 = 0.7 * 0.0025 = 0.00175.
    """
    flat = _frame([[100.0, 40.0]] * 4)
    buy = _frame([[0.0, 0.0], [0.7, 0.0], [0.7, 0.0], [0.7, 0.0]])
    sell = _frame([[0.7, 0.0], [0.7, 0.0], [0.0, 0.0], [0.0, 0.0]])
    short = _frame([[0.0, 0.0], [-0.7, 0.0], [-0.7, 0.0], [-0.7, 0.0]])

    engine = PythonEngine()
    net_buy = engine.run_backtest(flat, buy, {"cost_bps": COST_BPS}).returns
    net_sell = engine.run_backtest(flat, sell, {"cost_bps": COST_BPS}).returns
    net_short = engine.run_backtest(flat, short, {"cost_bps": COST_BPS}).returns

    # constant prices: any nonzero net is purely -cost
    cost_buy = -net_buy.iloc[2]  # held goes 0 -> +0.7 between d1 and d2
    cost_sell = -net_sell.iloc[3]  # held goes +0.7 -> 0 between d2 and d3
    cost_short = -net_short.iloc[2]  # held goes 0 -> -0.7 between d1 and d2
    expected = 0.7 * COST_BPS / 10_000.0  # = 0.00175
    for cost in (cost_buy, cost_sell, cost_short):
        assert abs(cost - expected) < ATOL
    assert cost_buy == cost_sell == cost_short  # |+-0.7| is the same trade size, bit-for-bit


def test_first_nonzero_position_charged_on_day_it_becomes_held():
    """Day-1 entry cost: a day-0 target weight is charged the day it becomes HELD (day 1).

    held = positions.shift(1) with a leading fillna(0), so held goes 0 -> 0.5 between d0 and
    d1 and held.diff() charges 0.5 of turnover on d1 — not d0 (nothing was traded yet) and
    not silently never (the classic bug where the leading NaN in diff() eats the entry trade).
    Constant prices again isolate the cost leg: net must be [0, -0.5*0.0025, 0, 0].
    """
    flat = _frame([[100.0, 40.0]] * 4)
    positions = _frame([[0.5, 0.0]] * 4)  # target 0.5 in A from day 0 onward

    result = PythonEngine().run_backtest(flat, positions, {"cost_bps": COST_BPS})
    expected_net = [0.0, -0.5 * COST_BPS / 10_000.0, 0.0, 0.0]  # d1: 0.5 * 0.0025 = 0.00125
    np.testing.assert_allclose(result.returns.to_numpy(), expected_net, rtol=0, atol=ATOL)
    assert abs(result.meta["total_turnover"] - 0.5) < ATOL  # the entry is ALL the turnover
