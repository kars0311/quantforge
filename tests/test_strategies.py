"""Week-4 proof suite: mean-reversion strategy + vetted-params gate (FR-3, SF-3).

All tests are offline on hand-built or seeded synthetic data — no network, no yfinance —
matching the pattern of test_smoke_vertical_slice.py / test_no_lookahead.py.

The core of the suite is a 10-row, 2-asset series engineered so every z-score is hand-computable
(the arithmetic is written out in comments), which pins the *exact* entry row, holding span and
exit row for both modes. That makes the tests teeth-bearing: flipping the entry comparison or
collapsing the hysteresis band inside the strategy changes at least one asserted row.

The expected z-scores are ALSO recomputed here with a plain numpy loop (``_recompute_z``) that
shares no code with the strategy — so the literal expectations are cross-checked against an
independent implementation, never against the strategy's own output.

Design of the hand-built series (lookback=5, entry_z=1.5, exit_z=0.5):

    MR (mean-reverter, drives the LONG book):   100 100 100 100  95  97  96  99  95  97
    OB (overbought, drives the SHORT book):     200 200 200 200 210 206 208 202 205 205

Key identity used for the entry rows: after four equal prices a single move of size d gives
mean = base - d/5, deviations (d/5 x4, -4d/5), sum-of-squares 4d^2/5, sample var d^2/5,
std d/sqrt(5), hence z = (-4d/5)/(d/sqrt(5)) = -4/sqrt(5) ~= -1.7889 — independent of d, and
below -entry_z = -1.5. OB mirrors it upward (+4/sqrt(5) > +1.5).
"""

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.python_engine import PythonEngine
from quantforge.strategies import validate_params
from quantforge.strategies.mean_reversion import MeanReversionStrategy

# Params for the hand-built series. All inside the whitelist bounds
# (lookback [5,60], entry_z [0.5,3.0], exit_z [0.0,1.5]).
LOOKBACK = 5
ENTRY_Z = 1.5
EXIT_Z = 0.5
HAND_PARAMS = {"lookback": LOOKBACK, "entry_z": ENTRY_Z, "exit_z": EXIT_Z}


def _hand_built_prices() -> pd.DataFrame:
    """10-row 2-asset series whose z-scores (lookback=5, ddof=1) are hand-computable.

    Per-row arithmetic for MR (window = the 5 prices ending at that row; var uses n-1=4):

    row 4  [100,100,100,100,95]: mean 99;   devs [1,1,1,1,-4];          ss 20;   var 5;
           z = -4/sqrt(5)        ~= -1.7889  < -1.5           -> ENTER long
    row 5  [100,100,100,95,97]:  mean 98.4; devs [1.6,1.6,1.6,-3.4,-1.4]; ss 21.2; var 5.3;
           z = -1.4/sqrt(5.3)    ~= -0.6081  in (-1.5, -0.5)  -> HOLD (hysteresis band)
    row 6  [100,100,95,97,96]:   mean 97.6; devs [2.4,2.4,-2.6,-0.6,-1.6]; ss 21.2; var 5.3;
           z = -1.6/sqrt(5.3)    ~= -0.6950  in (-1.5, -0.5)  -> HOLD
    row 7  [100,95,97,96,99]:    mean 97.4; devs [2.6,-2.4,-0.4,-1.4,1.6]; ss 17.2; var 4.3;
           z = +1.6/sqrt(4.3)    ~= +0.7716  > -0.5           -> EXIT (and < 1.5: no short)
    row 8  [95,97,96,99,95]:     mean 96.4; devs [-1.4,0.6,-0.4,2.6,-1.4]; ss 11.2; var 2.8;
           z = -1.4/sqrt(2.8)    ~= -0.8367  in the band, but the last event was an exit,
                                              so hysteresis keeps it FLAT (path dependence)
    row 9  [97,96,99,95,97]:     mean 96.8; devs [0.2,-0.8,2.2,-1.8,0.2]; ss 8.8; var 2.2;
           z = +0.2/sqrt(2.2)    ~= +0.1348                   -> stays flat

    OB mirrors the entry/hold/cover shape on the SHORT side:

    row 4  [200,200,200,200,210]: mean 202;   devs [-2,-2,-2,-2,8]; ss 80; var 20;
           z = +8/sqrt(20) = +4/sqrt(5) ~= +1.7889 > +1.5    -> SHORT enter (long_short only)
    row 5  [200,200,200,210,206]: mean 203.2; ss 84.8; var 21.2;
           z = +2.8/sqrt(21.2)   ~= +0.6081  in (0.5, 1.5)   -> HOLD short
    row 6  [200,200,210,206,208]: mean 204.8; ss 84.8; var 21.2;
           z = +3.2/sqrt(21.2)   ~= +0.6950  in (0.5, 1.5)   -> HOLD short
    row 7  [200,210,206,208,202]: mean 205.2; ss 68.8; var 17.2;
           z = -3.2/sqrt(17.2)   ~= -0.7716  < +0.5          -> COVER (and > -1.5: no long)
    rows 8-9: z ~= -0.3956, -0.0923                          -> stays flat

    Rows 0-3 have fewer than lookback observations, so min_periods forces z = NaN -> flat.
    MR's z never exceeds +1.5 and OB's never drops below -1.5, so in long_flat OB is flat
    for the whole series and in long_short each name only ever works one side.
    """
    return pd.DataFrame(
        {
            "MR": [100.0, 100.0, 100.0, 100.0, 95.0, 97.0, 96.0, 99.0, 95.0, 97.0],
            "OB": [200.0, 200.0, 200.0, 200.0, 210.0, 206.0, 208.0, 202.0, 205.0, 205.0],
        },
        index=pd.bdate_range("2024-01-01", periods=10),
    )


# The docstring's arithmetic as literals — the single source the assertions compare against.
# (Entry rows are the exact d-independent identity ±4/sqrt(5); the rest are the row fractions.)
# Kept hand-formatted: the column-aligned per-row comments mirror the docstring's table.
# fmt: off
EXPECTED_Z_MR = [
    np.nan, np.nan, np.nan, np.nan,
    -4.0 / np.sqrt(5.0),      # row 4: ENTER  (-1.7889 < -1.5)
    -1.4 / np.sqrt(5.3),      # row 5: band   (-0.6081)
    -1.6 / np.sqrt(5.3),      # row 6: band   (-0.6950)
    +1.6 / np.sqrt(4.3),      # row 7: EXIT   (+0.7716 > -0.5)
    -1.4 / np.sqrt(2.8),      # row 8: band again, but post-exit -> must stay flat
    +0.2 / np.sqrt(2.2),      # row 9: quiet
]
EXPECTED_Z_OB = [
    np.nan, np.nan, np.nan, np.nan,
    +4.0 / np.sqrt(5.0),      # row 4: SHORT enter (+1.7889 > +1.5)
    +2.8 / np.sqrt(21.2),     # row 5: short band  (+0.6081)
    +3.2 / np.sqrt(21.2),     # row 6: short band  (+0.6950)
    -3.2 / np.sqrt(17.2),     # row 7: COVER       (-0.7716 < +0.5)
    -1.2 / np.sqrt(9.2),      # row 8: quiet       (-0.3956)
    -0.2 / np.sqrt(4.7),      # row 9: quiet       (-0.0923)
]
# fmt: on


def _recompute_z(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Independent z-score recomputation: explicit numpy loop, no pandas rolling, no strategy code.

    Deliberately the dumbest possible implementation — for each row take the trailing
    ``lookback`` prices, numpy mean and std(ddof=1) — so agreement with the literals above is
    evidence about the *math*, not about two calls into the same library path.
    """
    out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for col in prices.columns:
        vals = prices[col].to_numpy()
        for t in range(lookback - 1, len(vals)):
            window = vals[t - lookback + 1 : t + 1]
            out.iloc[t, out.columns.get_loc(col)] = (vals[t] - window.mean()) / window.std(ddof=1)
    return out


def _synthetic_prices(n: int = 300, k: int = 6, seed: int = 17) -> pd.DataFrame:
    """Seeded lognormal walk, same recipe as the smoke test — offline and reproducible."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    rets = rng.normal(0.0002, 0.012, size=(n, k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(k)])


# ---------------------------------------------------------------------------
# (a) exact entry / holding span / exit on the hand-built series, both modes
# ---------------------------------------------------------------------------


def test_hand_arithmetic_matches_independent_recomputation():
    """The literal z expectations equal an independent numpy recomputation to near machine eps.

    This anchors every downstream row-level assertion: if this passes, the entry/hold/exit rows
    asserted below really do follow from the price series and the z definition, not from a typo.
    """
    z = _recompute_z(_hand_built_prices(), LOOKBACK)
    np.testing.assert_allclose(z["MR"].to_numpy(), EXPECTED_Z_MR, rtol=1e-12)
    np.testing.assert_allclose(z["OB"].to_numpy(), EXPECTED_Z_OB, rtol=1e-12)
    # And the threshold relations the strategy must react to (strict, no boundary luck):
    assert EXPECTED_Z_MR[4] < -ENTRY_Z  # long entry fires
    assert -ENTRY_Z < EXPECTED_Z_MR[5] < -EXIT_Z  # inside band -> hold
    assert -ENTRY_Z < EXPECTED_Z_MR[6] < -EXIT_Z  # inside band -> hold
    assert EXPECTED_Z_MR[7] > -EXIT_Z  # exit fires
    assert EXPECTED_Z_OB[4] > ENTRY_Z  # short entry fires
    assert EXIT_Z < EXPECTED_Z_OB[5] < ENTRY_Z  # short band -> hold
    assert EXIT_Z < EXPECTED_Z_OB[6] < ENTRY_Z  # short band -> hold
    assert EXPECTED_Z_OB[7] < EXIT_Z  # cover fires


def test_long_flat_exact_entry_hold_exit_rows():
    """long_flat: MR enters on row 4, holds rows 5-6 through the band, exits row 7; OB never longs.

    Weight while long is exactly 1.0 (one active name, state 1 / n_active 1). Exact frame
    comparison — any shift of the entry row, loss of the band hold, or early/late exit fails.
    """
    prices = _hand_built_prices()
    weights = MeanReversionStrategy().generate_signals(prices, {**HAND_PARAMS, "mode": "long_flat"})

    expected = pd.DataFrame(
        {
            #      r0   r1   r2   r3   r4   r5   r6   r7   r8   r9
            "MR": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "OB": [0.0] * 10,  # OB's z never dips below -entry_z -> never long
        },
        index=prices.index,
    )
    pd.testing.assert_frame_equal(weights, expected, check_exact=True)


def test_long_short_exact_entry_hold_exit_and_short_cover_rows():
    """long_short: symmetric books — MR long AND OB short enter row 4, hold 5-6, close row 7.

    With both sides active n_active = 2, so weights are exactly +0.5 / -0.5 (0.5 is a power of
    two: exact in floats). Covers the required symmetric short entry (z > entry_z) and cover
    (z < exit_z), including the short-side hysteresis hold at rows 5-6.
    """
    prices = _hand_built_prices()
    weights = MeanReversionStrategy().generate_signals(
        prices, {**HAND_PARAMS, "mode": "long_short"}
    )

    expected = pd.DataFrame(
        {
            #      r0   r1   r2   r3    r4    r5    r6   r7   r8   r9
            "MR": [0.0, 0.0, 0.0, 0.0, +0.5, +0.5, +0.5, 0.0, 0.0, 0.0],
            "OB": [0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0],
        },
        index=prices.index,
    )
    pd.testing.assert_frame_equal(weights, expected, check_exact=True)


# ---------------------------------------------------------------------------
# (b) structural invariants: shape, gross exposure, full investment, warmup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["long_flat", "long_short"])
def test_structural_invariants_on_random_data(mode):
    """On generic data: output mirrors prices' shape/index/columns and gross never exceeds 1.

    Gross Sigma|w| <= 1 + 1e-9 is the engine's admission contract (leverage would silently
    invalidate the cost model); long_flat additionally must be long-only.
    """
    prices = _synthetic_prices()
    weights = MeanReversionStrategy().generate_signals(prices, {**HAND_PARAMS, "mode": mode})

    assert weights.shape == prices.shape
    assert weights.index.equals(prices.index)
    assert list(weights.columns) == list(prices.columns)
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-9).all()
    if mode == "long_flat":
        assert (weights >= 0.0).all().all()


def test_long_flat_fully_invested_when_any_long_active():
    """Whenever at least one long is on in long_flat, Sigma w = 1: cash earns nothing here.

    On the hand-built series (n_active = 1) the sum is exactly 1.0 — asserted with ``==``.
    On random data n_active varies; summing n copies of 1/n is 1 ULP off 1.0 for some n
    (e.g. 6, 7 in pairwise summation), so there the assertion is |sum - 1| <= 1e-9, the same
    tolerance the engine admits. Rows with no active longs must be exactly all-zero.
    """
    strategy = MeanReversionStrategy()

    hand = strategy.generate_signals(_hand_built_prices(), {**HAND_PARAMS, "mode": "long_flat"})
    active = hand.sum(axis=1) > 0.0
    assert active.any()
    assert (hand.loc[active].sum(axis=1) == 1.0).all()  # exact: single-name book
    assert (hand.loc[~active] == 0.0).all().all()

    rand = strategy.generate_signals(_synthetic_prices(), {**HAND_PARAMS, "mode": "long_flat"})
    active = rand.sum(axis=1) > 0.0
    assert active.any(), "series should trigger at least one entry for the test to bite"
    assert ((rand.loc[active].sum(axis=1) - 1.0).abs() <= 1e-9).all()
    assert (rand.loc[~active] == 0.0).all().all()


@pytest.mark.parametrize("mode", ["long_flat", "long_short"])
def test_warmup_rows_are_flat(mode):
    """min_periods=lookback: the first lookback-1 rows have no full window, so weights are 0.

    A partial-window z would be a look-ahead cousin (statistics from data that doesn't exist
    yet); the strategy must stay flat until row index lookback-1, where the first full
    window ends.
    """
    lookback = 20  # the whitelist default, larger than the hand series' 5 to make warmup visible
    prices = _synthetic_prices()
    weights = MeanReversionStrategy().generate_signals(prices, {"lookback": lookback, "mode": mode})
    assert (weights.iloc[: lookback - 1] == 0.0).all().all()


# ---------------------------------------------------------------------------
# (c) no look-ahead: the future cannot reach back into earlier weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["long_flat", "long_short"])
def test_tail_truncation_leaves_earlier_weights_identical(mode):
    """Chopping the last 40 rows of prices leaves every earlier weight bit-identical.

    Same proof shape as the engine's no-look-ahead test: if any future row influenced a past
    weight (centered window, full-sample stat, reversed shift), removing the tail would perturb
    the prefix. Exact comparison — even one ULP of drift fails.
    """
    prices = _synthetic_prices()
    strategy = MeanReversionStrategy()
    params = {**HAND_PARAMS, "mode": mode}

    full = strategy.generate_signals(prices, params)
    truncated = strategy.generate_signals(prices.iloc[:-40], params)

    pd.testing.assert_frame_equal(truncated, full.iloc[:-40], check_exact=True)


# ---------------------------------------------------------------------------
# (d) hysteresis is path-dependent state, not a pointwise function of z
# ---------------------------------------------------------------------------


def test_hysteresis_band_weight_depends_on_path():
    """Two rows with z inside the same band carry different weights depending on history.

    MR row 5 (z ~= -0.61) sits in (-entry_z, -exit_z) *after* the row-4 entry -> long (1.0).
    MR row 8 (z ~= -0.84) sits in the same band *after* the row-7 exit -> flat (0.0).
    A memoryless rule (any pointwise function weight = f(z)) cannot produce both, so this pins
    the state machine itself: in-band means "keep whatever the last entry/exit decided".
    """
    prices = _hand_built_prices()
    z = _recompute_z(prices, LOOKBACK)

    # Both rows genuinely inside the band, per the independent recomputation:
    assert -ENTRY_Z < z.loc[prices.index[5], "MR"] < -EXIT_Z
    assert -ENTRY_Z < z.loc[prices.index[8], "MR"] < -EXIT_Z

    weights = MeanReversionStrategy().generate_signals(prices, {**HAND_PARAMS, "mode": "long_flat"})
    assert weights.loc[prices.index[5], "MR"] == 1.0  # in band, post-entry -> still long
    assert weights.loc[prices.index[8], "MR"] == 0.0  # in band, post-exit  -> stays flat


# ---------------------------------------------------------------------------
# (e) validate_params: the SF-3 vetted-set gate
# ---------------------------------------------------------------------------


def test_validate_params_merges_defaults():
    """Empty input returns the full documented default set for both vetted strategies."""
    assert validate_params("mean_reversion", {}) == {
        "lookback": 20,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "mode": "long_flat",
    }
    assert validate_params("momentum", {}) == {"lookback": 126, "top_n": 0}


def test_validate_params_does_not_mutate_input():
    """The caller's dict comes back byte-identical; the merged result is a NEW object.

    Callers hold the raw params for logging/replay — a validator that back-fills defaults into
    the caller's dict would corrupt that record.
    """
    supplied = {"lookback": 10}
    out = validate_params("mean_reversion", supplied)
    assert supplied == {"lookback": 10}
    assert out is not supplied
    assert out["lookback"] == 10 and out["entry_z"] == 2.0  # partial merge worked


def test_validate_params_rejects_unvetted_strategy():
    """A strategy name outside the vetted set is refused — the whole point of SF-3."""
    with pytest.raises(ValueError, match="pairs"):
        validate_params("pairs", {})


def test_validate_params_rejects_unknown_param():
    """Typos and smuggled knobs are rejected by name, not silently dropped."""
    with pytest.raises(ValueError, match="lookbak"):
        validate_params("mean_reversion", {"lookbak": 20})


def test_validate_params_rejects_out_of_range_low():
    """Below the documented minimum (mean_reversion lookback >= 5)."""
    with pytest.raises(ValueError, match="lookback"):
        validate_params("mean_reversion", {"lookback": 4})


def test_validate_params_rejects_out_of_range_high():
    """Above the documented maximum (entry_z <= 3.0)."""
    with pytest.raises(ValueError, match="entry_z"):
        validate_params("mean_reversion", {"entry_z": 3.5})


def test_validate_params_rejects_wrong_type():
    """A bool is never a window length (despite bool subclassing int), nor is a string."""
    with pytest.raises(ValueError, match="lookback"):
        validate_params("mean_reversion", {"lookback": True})
    with pytest.raises(ValueError, match="exit_z"):
        validate_params("mean_reversion", {"exit_z": "0.5"})


def test_validate_params_rejects_invalid_mode_choice():
    """mode is categorical: anything outside {long_flat, long_short} is refused."""
    with pytest.raises(ValueError, match="mode"):
        validate_params("mean_reversion", {"mode": "short_only"})


# ---------------------------------------------------------------------------
# (f) end to end: mean-reversion weights through the real engine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["long_flat", "long_short"])
def test_end_to_end_through_engine(mode):
    """Validated params -> signals -> PythonEngine with cost_bps=10 yields a finite result.

    Uses validate_params for the strategy knobs (exercising the same path the MCP/UI layers
    will), and asserts every downstream artifact is finite: no NaNs or infs in returns, a
    positive equity curve, and non-NaN headline metrics.
    """
    prices = _synthetic_prices()
    params = validate_params("mean_reversion", {"lookback": 10, "mode": mode})
    weights = MeanReversionStrategy().generate_signals(prices, params)

    result = PythonEngine().run_backtest(prices, weights, {"cost_bps": 10})

    assert len(result.equity_curve) == len(prices)
    assert np.isfinite(result.returns).all()
    assert np.isfinite(result.equity_curve).all()
    assert (result.equity_curve > 0.0).all()
    for key in ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "hit_rate"]:
        assert key in result.metrics
        assert np.isfinite(result.metrics[key])
