"""FR-4 end-to-end proof: combine two vetted strategies into a portfolio (week 5, milestone 3).

This suite exercises the WHOLE "combine strategies into a portfolio" pipeline through the frozen
interfaces only (AR-1) — no production shims, no test-only hooks:

    STRATEGIES registry + validate_params defaults
        -> generate_signals -> PythonEngine.run_backtest(cost_bps=10)   (per strategy)
        -> 2-column wide strategy-returns panel
        -> optimize_weights -> combine_returns -> compute_metrics

on seeded synthetic wide prices (same generator pattern as tests/test_smoke_vertical_slice.py),
with the synthetic index deliberately placed inside the loader's train split so the whole flow is
demonstrably driven from get_split_bounds() train+validation bounds and never touches a holdout
date.

Panel alignment rule (documented once, asserted below): each strategy's return stream comes out of
PythonEngine on the SAME price index, and the engine's contract makes those streams NaN-free —
warmup days carry zero positions, hence zero returns, not NaNs. The panel is therefore built as an
inner join on dates followed by dropna(); on these inputs both operations are identities (asserted:
full length, no NaNs), but they state the rule any future stream with genuine warmup NaNs must
follow — keep only dates where EVERY stream has a real return, so no weight is ever applied to a
fabricated number.

The teeth of the suite:
- The blended stream is recomputed independently with plain numpy (w0*r0 + w1*r1 on raw ndarrays,
  no pandas alignment, no shared code with the module) and must match at atol 1e-12.
- Each panel column must equal the engine's BacktestResult.returns VERBATIM — proof that the
  combination layer introduces no re-shifting: look-ahead prevention lives in the engine's
  positions.shift(1) and nothing downstream may shift (or un-shift) again.
- Blended ann_vol <= max individual ann_vol + 1e-12: for long-only convex weights this is a
  theorem (triangle inequality in L2), so a violation can only mean broken weights or broken math.
- combine_returns must do no date filtering of its own: the blend's index is the panel's index,
  verbatim.
"""

import numpy as np
import pandas as pd
import pytest

from quantforge.data.loader import get_split_bounds
from quantforge.engine.python_engine import PythonEngine
from quantforge.metrics.performance import compute_metrics
from quantforge.portfolio.optimize import combine_returns, optimize_weights
from quantforge.strategies import STRATEGIES, validate_params

# Metric keys compute_metrics promises, in its documented order (mirrors the smoke test's list;
# kept literal here so a production-side rename fails a test instead of silently renaming both).
_KEYS = ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "hit_rate"]

#: The two vetted strategies this milestone blends, in panel column order.
_STRATEGY_NAMES = ["momentum", "mean_reversion"]


def _synthetic_prices(n: int = 400, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Seeded synthetic wide prices, same generator as tests/test_smoke_vertical_slice.py.

    The index starts at the TRAIN split's start date (from get_split_bounds, the single source
    of truth for split dates) so 400 business days land deep inside train — the whole pipeline
    run is then provably confined to train+validation, mirroring how the AI agent must operate
    (holdout is scored once, never iterated on).
    """
    train_start = get_split_bounds()["train"][0]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(train_start, periods=n)
    rets = rng.normal(0.0004, 0.01, size=(n, k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(k)])


def _build_panel(results: dict) -> pd.DataFrame:
    """Strategy-returns panel per the documented alignment rule: inner join, then drop any row
    where a stream lacks a real return. See the module docstring for why both steps exist even
    though they are identities for engine-produced (NaN-free, same-index) streams."""
    panel = pd.concat(
        {name: results[name].returns for name in _STRATEGY_NAMES}, axis=1, join="inner"
    )
    panel.columns = list(_STRATEGY_NAMES)
    return panel.dropna()


@pytest.fixture(scope="module")
def pipeline():
    """Run the full FR-4 flow ONCE and share the artifacts across tests.

    Module scope (why): the pipeline is deterministic (seeded data, deterministic engine and
    solver), so re-running it per test would only slow the suite without adding independence.
    Tests treat every artifact as read-only.

    Objective choice (why min_volatility here): on this seed the mean-reversion stream has a
    negative historical mean, so max_sharpe corners the whole budget into momentum — a valid
    answer, but weights of (1, 0) would make the "blend equals the weighted sum" proof vacuous.
    min_volatility (also whitelisted, same estimators) lands strictly interior weights, so the
    combination layer is exercised on a genuinely two-strategy blend. The default max_sharpe
    path is proven separately in test_default_objective_also_flows_through.
    """
    prices = _synthetic_prices()
    engine = PythonEngine()

    results = {}
    for name in _STRATEGY_NAMES:
        params = validate_params(name, {})  # {} -> whitelist defaults, the vetted-path contract
        strategy = STRATEGIES[name]()  # registry-driven construction, not direct imports
        positions = strategy.generate_signals(prices, params)
        results[name] = engine.run_backtest(prices, positions, {"cost_bps": 10})

    panel = _build_panel(results)
    weights = optimize_weights(panel, {"objective": "min_volatility"})
    blended = combine_returns(panel, weights)
    metrics = compute_metrics(blended)

    return {
        "prices": prices,
        "results": results,
        "panel": panel,
        "weights": weights,
        "blended": blended,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Registry-driven construction
# ---------------------------------------------------------------------------


def test_registry_provides_both_strategies():
    """Both blend legs come from the vetted registry — the same gate the AI/MCP layer uses."""
    assert set(_STRATEGY_NAMES) <= set(STRATEGIES)
    for name in _STRATEGY_NAMES:
        instance = STRATEGIES[name]()
        assert instance.name == name


def test_validate_params_defaults_round_trip():
    """validate_params({}) yields the full whitelisted default set for each leg, and validating
    those defaults again is a fixed point — 'defaults' and 'validated defaults' must mean the
    same backtest (the registry module's own documented invariant)."""
    for name in _STRATEGY_NAMES:
        defaults = validate_params(name, {})
        assert defaults  # non-empty: every strategy has at least one whitelisted knob
        assert validate_params(name, defaults) == defaults


# ---------------------------------------------------------------------------
# Panel construction: alignment, warmup, no re-shifting
# ---------------------------------------------------------------------------


def test_panel_shape_and_alignment(pipeline):
    """Inner-join + dropna are identities for engine streams: same index as prices, no NaNs.

    Both streams come from the same engine on the same price grid and the engine emits NaN-free
    returns (warmup = zero position = zero return), so nothing may be lost in alignment. A
    shorter panel here would mean a stream sprouted NaNs or a divergent index — either one is a
    contract regression upstream."""
    panel, prices = pipeline["panel"], pipeline["prices"]
    assert list(panel.columns) == _STRATEGY_NAMES
    assert panel.index.equals(prices.index)
    assert len(panel) == len(prices)
    assert not panel.isna().any().any()


def test_panel_equals_engine_returns_verbatim(pipeline):
    """No re-shifting in the combination layer (look-ahead guard).

    The one-day execution lag is applied exactly once, inside PythonEngine (positions.shift(1)).
    The panel must therefore carry each BacktestResult.returns byte-for-byte: a second shift
    would understate returns by a day of lag, and an un-shift would reintroduce look-ahead. Exact
    equality (check_exact) leaves no room for a 'small' transformation hiding in the middle."""
    for name in _STRATEGY_NAMES:
        pd.testing.assert_series_equal(
            pipeline["panel"][name],
            pipeline["results"][name].returns,
            check_exact=True,
            check_names=False,
        )


def test_streams_are_nontrivial(pipeline):
    """Both legs actually traded: nonzero variance and some turnover. Guards the fixture itself —
    every downstream assertion would pass vacuously on two all-zero streams."""
    for name in _STRATEGY_NAMES:
        result = pipeline["results"][name]
        assert float(result.returns.std()) > 0.0
        assert result.meta["total_turnover"] > 0.0
        assert result.meta["cost_bps"] == 10.0  # the milestone's required cost setting


# ---------------------------------------------------------------------------
# Optimize -> combine -> metrics
# ---------------------------------------------------------------------------


def test_weights_are_convex_over_the_panel(pipeline):
    """Long-only budget: weights indexed by the panel's columns, each in [0, 1], summing to 1."""
    weights = pipeline["weights"]
    assert list(weights.index) == _STRATEGY_NAMES
    assert ((weights >= 0.0) & (weights <= 1.0)).all()
    assert abs(float(weights.sum()) - 1.0) <= 1e-8


def test_blend_matches_independent_numpy_recomputation(pipeline):
    """The core FR-4 arithmetic, re-derived with no shared code: pull raw ndarrays out of the
    panel and weights and compute w_mom*r_mom + w_mr*r_mr with plain numpy — no pandas alignment,
    no module internals. atol 1e-12 leaves room only for float summation-order noise."""
    panel, weights, blended = pipeline["panel"], pipeline["weights"], pipeline["blended"]
    expected = (
        weights.loc["momentum"] * panel["momentum"].to_numpy()
        + weights.loc["mean_reversion"] * panel["mean_reversion"].to_numpy()
    )
    np.testing.assert_allclose(blended.to_numpy(), expected, rtol=0.0, atol=1e-12)


def test_combined_equity_starts_at_one(pipeline):
    """cumprod(1 + blended) opens at exactly 1.0: day one is warmup for both strategies (no
    position held yet — the engine's shift guarantees it), so the first blended return is 0.0
    and the equity curve starts at par. A nonzero first return would mean a position existed
    before any signal could have been formed — a look-ahead symptom."""
    blended = pipeline["blended"]
    equity = (1.0 + blended).cumprod()
    assert float(blended.iloc[0]) == 0.0
    assert float(equity.iloc[0]) == 1.0


def test_metrics_keys_exact_and_finite(pipeline):
    """compute_metrics on the blend returns exactly the promised keys, every value finite. The
    single-source-of-truth metrics module serves the blended portfolio the same way it serves a
    single strategy — no special-casing, no NaN leakage from the combination step."""
    metrics = pipeline["metrics"]
    assert list(metrics.keys()) == _KEYS
    assert all(np.isfinite(v) for v in metrics.values())


def test_blended_vol_bounded_by_max_individual(pipeline):
    """Always-true invariant for long-only convex weights: blended ann_vol <= max individual
    ann_vol (+1e-12 float slack). By the L2 triangle inequality,
    std(w1*r1 + w2*r2) <= w1*std(r1) + w2*std(r2) <= max(std) for w >= 0 summing to 1 — with NO
    assumption on correlation, so this is a structural check, not a performance hope. (The
    stronger 'blend beats the min vol' claim is NOT always true and is deliberately not made.)"""
    individual_vols = [
        compute_metrics(pipeline["results"][name].returns)["ann_vol"] for name in _STRATEGY_NAMES
    ]
    assert pipeline["metrics"]["ann_vol"] <= max(individual_vols) + 1e-12


def test_default_objective_also_flows_through(pipeline):
    """The default optimize_weights path (max_sharpe) also produces a valid convex weighting
    that combine_returns accepts — even when, as on this seed, it corners into one stream. The
    corner case is exactly why the fixture uses min_volatility for the main proof (see the
    fixture docstring), but the default path must still compose end-to-end."""
    panel = pipeline["panel"]
    weights = optimize_weights(panel)  # default params -> max_sharpe, (0, 1) bounds
    assert abs(float(weights.sum()) - 1.0) <= 1e-8
    assert ((weights >= 0.0) & (weights <= 1.0)).all()
    blended = combine_returns(panel, weights)
    expected = panel.to_numpy() @ weights.reindex(panel.columns).to_numpy()
    np.testing.assert_allclose(blended.to_numpy(), expected, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Split discipline: no holdout contact, no hidden date filtering
# ---------------------------------------------------------------------------


def test_flow_never_touches_holdout_dates(pipeline):
    """The synthetic index is driven from get_split_bounds(): it starts at train start and every
    date used anywhere in the flow stays <= validation end, so the holdout is untouched by
    construction — the same discipline the AI agent is held to on real data (RG-4)."""
    bounds = get_split_bounds()
    validation_end = pd.Timestamp(bounds["validation"][1])
    holdout_start = pd.Timestamp(bounds["holdout"][0])
    for frame in (pipeline["prices"], pipeline["panel"]):
        assert frame.index.max() <= validation_end
        assert frame.index.max() < holdout_start
    assert pipeline["blended"].index.max() <= validation_end


def test_combine_layer_does_no_date_filtering(pipeline):
    """combine_returns is date-agnostic: its output index is the panel's index verbatim. Any
    slicing to train/validation is the CALLER's job (done here at price construction) — a
    combination layer that quietly dropped or added dates could hide holdout contamination or
    fabricate a longer track record."""
    assert pipeline["blended"].index.equals(pipeline["panel"].index)
    assert len(pipeline["blended"]) == len(pipeline["panel"])
