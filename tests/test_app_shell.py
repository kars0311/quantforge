"""Week-5 proof suite for the Streamlit UI shell (app/streamlit_app.py).

Everything here runs HEADLESS — no Streamlit server, no network, no price cache. That is the
point: the app module separates pure chart builders / spec helpers from widget code, so pytest
can import it and exercise the analytics without booting a UI. The import-isolation test runs in
a subprocess because this pytest process may already have imported ``quantforge.ai`` modules for
other suites — a fresh interpreter is the only honest way to assert what *this module's* import
pulls in.

What is proven, per the week-5 milestone spec:
- importing app.streamlit_app succeeds headless and imports NO ``quantforge.ai.*`` module;
- plot_equity / plot_drawdown / plot_frontier return plotly Figures with the expected trace
  counts and types on synthetic inputs;
- drawdown values are ≤ 0 everywhere and hit exactly 0 at each running peak;
- the equal-weight benchmark is shift-consistent: one point is pinned against a hand
  computation using the "decided at close of t, earns t+1's return" convention;
- the sidebar date-bound constants equal get_split_bounds()'s train start / validation end
  (holdout unselectable, RG-4);
- every param control range is SOURCED from PARAM_WHITELIST (proven by mutating the whitelist
  and watching the specs follow — introspection, not a duplicated table of numbers).
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

import app.streamlit_app as app
from quantforge.data.loader import get_split_bounds
from quantforge.engine.base import BacktestResult
from quantforge.metrics.performance import _KEYS, compute_metrics
from quantforge.strategies import PARAM_WHITELIST

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------
# Fixtures: synthetic inputs, no cache, no network
# ---------------------------------------------------------------------------------------------


@pytest.fixture()
def prices() -> pd.DataFrame:
    """Tiny wide price panel with hand-checkable returns (2 tickers, 6 business days)."""
    idx = pd.date_range("2015-01-05", periods=6, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 101.0, 103.0, 104.0],
            "BBB": [50.0, 50.0, 51.0, 52.0, 52.0, 53.0],
        },
        index=idx,
    )


@pytest.fixture()
def result() -> BacktestResult:
    """Synthetic BacktestResult whose equity has two peaks with a drawdown between them."""
    idx = pd.date_range("2015-01-05", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.0, 0.02, -0.03, 0.01, 0.04, -0.01], index=idx)
    equity = (1.0 + returns).cumprod()
    return BacktestResult(
        equity_curve=equity, returns=returns, metrics=compute_metrics(returns), meta={}
    )


@pytest.fixture()
def frontier_df() -> pd.DataFrame:
    """Synthetic frontier frame; ret/risk is maximized at row 2 (0.12/0.15 = 0.8)."""
    return pd.DataFrame({"risk": [0.10, 0.12, 0.15, 0.20], "ret": [0.05, 0.08, 0.12, 0.14]})


@pytest.fixture()
def weights() -> pd.Series:
    return pd.Series([0.6, 0.4], index=["momentum", "mean_reversion"], name="weight")


# ---------------------------------------------------------------------------------------------
# Headless import + AI isolation
# ---------------------------------------------------------------------------------------------


def test_headless_import_pulls_no_ai_modules():
    """A fresh interpreter imports the app with zero quantforge.ai.* modules loaded.

    Subprocess (not in-process) because other test files legitimately import the ai package;
    sys.modules in THIS process proves nothing about what the app module itself imports.
    """
    code = (
        "import sys; import app.streamlit_app; "
        "bad = sorted(m for m in sys.modules if m.startswith('quantforge.ai')); "
        "assert not bad, f'app import pulled in AI modules: {bad}'; "
        "print('CLEAN')"
    )
    # No PYTHONPATH help on purpose: cwd = repo root is all a fresh clone gets, so this also
    # proves the app's own src/ bootstrap works (the same way `streamlit run` finds quantforge).
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "CLEAN" in proc.stdout


def test_run_pipeline_is_cache_decorated():
    """run_pipeline must be wrapped by st.cache_data (repeat clicks are free — spec FR-10)."""
    # st.cache_data-wrapped functions expose .clear(); a bare function does not.
    assert hasattr(app.run_pipeline, "clear")


# ---------------------------------------------------------------------------------------------
# Chart builders: figure types, trace counts, invariants
# ---------------------------------------------------------------------------------------------


def test_plot_equity_without_prices_has_one_scatter(result):
    fig = app.plot_equity(result)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter)


def test_plot_equity_with_prices_adds_benchmark_trace(result, prices):
    fig = app.plot_equity(result, prices)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)
    names = [trace.name for trace in fig.data]
    assert names == ["Strategy", "Equal-weight benchmark"]


def test_benchmark_point_matches_hand_computation(result, prices):
    """Pin the shift-consistent convention: the benchmark buys at the close of day 0, so day 0's
    equity is exactly 1.0 and day t's equity compounds the equal-weight MEAN of asset returns
    from day 1 through day t. Computed here from raw price floats, independently of pandas
    pct_change or the engine."""
    fig = app.plot_equity(result, prices)
    bench_y = fig.data[1].y

    r1 = ((101.0 / 100.0 - 1.0) + (50.0 / 50.0 - 1.0)) / 2.0
    r2 = ((102.0 / 101.0 - 1.0) + (51.0 / 50.0 - 1.0)) / 2.0
    expected_day2 = (1.0 + r1) * (1.0 + r2)

    assert bench_y[0] == pytest.approx(1.0, rel=1e-12)  # no look-ahead: day 0 earns nothing
    assert bench_y[2] == pytest.approx(expected_day2, rel=1e-12)


def test_plot_drawdown_invariants(result):
    """Underwater values are ≤ 0 everywhere and exactly 0 at every running peak."""
    fig = app.plot_drawdown(result)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter)

    dd = pd.Series(fig.data[0].y, dtype=float)
    assert (dd <= 1e-12).all()
    assert dd.max() == pytest.approx(0.0, abs=1e-12)  # the curve touches its peak

    # 0 exactly where equity equals its running max (rows 0, 1, and 4 for this fixture),
    # strictly negative in the trough between the peaks.
    equity = result.equity_curve.reset_index(drop=True)
    at_peak = equity == equity.cummax()
    assert at_peak[[0, 1, 4]].all()
    assert dd[at_peak].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert (dd[~at_peak] < 0).all()


def test_plot_frontier_traces_and_max_sharpe_marker(frontier_df, weights):
    fig = app.plot_frontier(frontier_df, weights)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)

    # Trace 0 = the full frontier; trace 1 = a single star at the max ret/risk point (rf = 0).
    assert len(fig.data[0].x) == len(frontier_df)
    star = fig.data[1]
    assert len(star.x) == 1
    assert star.x[0] == pytest.approx(0.15)
    assert star.y[0] == pytest.approx(0.12)
    assert star.marker.symbol == "star"
    # The weights are surfaced as hover text so the chart names the holdings.
    assert "momentum" in star.hovertext[0]


def test_plot_frontier_rejects_empty_frame(weights):
    with pytest.raises(ValueError, match="empty"):
        app.plot_frontier(pd.DataFrame({"risk": [], "ret": []}), weights)


# ---------------------------------------------------------------------------------------------
# Metric row
# ---------------------------------------------------------------------------------------------


def test_render_metrics_runs_bare_over_exact_keys(result):
    """Bare-mode call succeeds with a full metrics dict (streamlit widgets no-op headless)."""
    assert app.render_metrics(result.metrics) is None


def test_render_metrics_missing_key_raises():
    incomplete = {k: 0.0 for k in _KEYS if k != "sharpe"}
    with pytest.raises(ValueError, match="sharpe"):
        app.render_metrics(incomplete)


# ---------------------------------------------------------------------------------------------
# Sidebar bounds: RG-4, sourced from the loader's single source of truth
# ---------------------------------------------------------------------------------------------


def test_sidebar_date_bounds_equal_split_source():
    bounds = get_split_bounds()
    assert app.UI_DATE_MIN == date.fromisoformat(bounds["train"][0])
    assert app.UI_DATE_MAX == date.fromisoformat(bounds["validation"][1])
    # Every holdout date lies strictly beyond the pickers' max — unselectable by construction.
    assert date.fromisoformat(bounds["holdout"][0]) > app.UI_DATE_MAX


# ---------------------------------------------------------------------------------------------
# Param controls: ranges sourced from PARAM_WHITELIST by introspection, never duplicated
# ---------------------------------------------------------------------------------------------


def test_param_control_specs_match_whitelist_for_every_strategy():
    for strategy, whitelist in PARAM_WHITELIST.items():
        specs = app.param_control_specs(strategy)
        assert set(specs) == set(whitelist)
        for param, spec in whitelist.items():
            control = specs[param]
            if "choices" in spec:
                assert control["widget"] == "selectbox"
                assert control["options"] == sorted(spec["choices"])
                assert control["default"] == spec["default"]
            else:
                assert control["widget"] == "slider"
                assert control["min"] == spec["min"]
                assert control["max"] == spec["max"]
                assert control["default"] == spec["default"]
                assert control["step"] == (1 if spec["type"] is int else app._FLOAT_STEP)


def test_param_control_specs_track_whitelist_mutations(monkeypatch):
    """Mutate the whitelist and the control spec must follow — proving the UI derives its
    ranges from PARAM_WHITELIST at call time rather than shipping a hand-copied table that
    could silently drift from the SF-3 choke point."""
    sentinel = {"type": int, "min": 7, "max": 33, "default": 11}
    monkeypatch.setitem(PARAM_WHITELIST["momentum"], "lookback", sentinel)
    control = app.param_control_specs("momentum")["lookback"]
    assert (control["min"], control["max"], control["default"]) == (7, 33, 11)


def test_param_control_specs_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        app.param_control_specs("not_a_strategy")


# ---------------------------------------------------------------------------------------------
# Whole-script smoke: AppTest boots main() with every widget live
# ---------------------------------------------------------------------------------------------


def test_full_script_runs_headless_via_apptest():
    """Streamlit's own AppTest executes the ENTIRE script (main(), sidebar, all five tabs)
    headless — the mechanical version of '`streamlit run` starts without traceback'. With no
    price cache on the test machine the compute tabs must degrade to plain-language warnings,
    never an exception; switching strategies must regenerate the param controls cleanly."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=30)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert [t.label for t in at.tabs] == [
        "Backtest",
        "Portfolio",
        "AI Chat",
        "Research mode",
        "Methodology",
    ]
    # Date pickers land on, and are bounded to, the train+validation window.
    assert [str(d.value) for d in at.date_input] == [str(app.UI_DATE_MIN), str(app.UI_DATE_MAX)]

    at.selectbox[0].select("mean_reversion")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert {s.label for s in at.slider} == {"lookback", "entry_z", "exit_z"}


# ---------------------------------------------------------------------------------------------
# equal_weight_benchmark: engine-backed, therefore shift-consistent by construction
# ---------------------------------------------------------------------------------------------


def test_equal_weight_benchmark_starts_at_one_and_is_costless(prices):
    bench = app.equal_weight_benchmark(prices)
    assert isinstance(bench, pd.Series)
    assert bench.iloc[0] == pytest.approx(1.0, rel=1e-12)
    assert bench.index.equals(prices.index)
    # Costless and always fully invested: every post-warmup day compounds the mean return.
    mean_returns = prices.pct_change().fillna(0.0).mean(axis=1)
    expected = (1.0 + mean_returns).cumprod()
    pd.testing.assert_series_equal(bench, expected, rtol=1e-12, check_names=False)
