"""Independent verifier suite for the week-5 Streamlit shell (app/streamlit_app.py).

Written adversarially, on top of tests/test_app_shell.py, to close the gaps that suite left:

- ``run_pipeline`` had never actually been executed (the builder had no price cache). Here it
  runs END TO END against a synthetic Parquet cache in a temp cwd, with ``yfinance`` poisoned so
  any network attempt fails the test — proving the UI pipeline path is cache-hit-only AND that
  its output is byte-identical to composing loader → to_wide → strategy → engine by hand.
- The equal-weight benchmark is pinned against a full hand computation on a RAGGED panel
  (pre-IPO NaN prices), the case the docstring's NaN policy talks about but no test exercised.
- ``plot_frontier``'s max-Sharpe marker is attacked with a zero-risk row whose naive ret/risk
  is +inf — the star must land on the best finite-Sharpe point, not the divide-by-zero one.
- RG-4 and SF-3 are checked at the RENDERED WIDGET level via streamlit.testing.v1 (not just on
  the pure helpers): the actual date pickers cannot reach the holdout, and the actual slider
  bounds equal PARAM_WHITELIST's for every numeric param of every vetted strategy.

Everything runs headless, offline, on synthetic data.
"""

from __future__ import annotations

import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app.streamlit_app as app
from quantforge import interchange
from quantforge.data.loader import get_split_bounds, load_prices
from quantforge.engine.base import BacktestResult
from quantforge.engine.python_engine import PythonEngine
from quantforge.strategies import PARAM_WHITELIST, STRATEGIES, validate_params

REPO_ROOT = Path(__file__).resolve().parents[1]


def _poison_yfinance(monkeypatch) -> None:
    """Install a yfinance stand-in whose download() fails the test on sight.

    The lazy ``import yfinance`` inside the loader resolves through sys.modules, so this
    guarantees that if any code path under test tries to download, the test fails with a
    named assertion instead of silently hitting the network.
    """

    def _no_network(*args, **kwargs):  # pragma: no cover - reaching this IS the failure
        raise AssertionError("UI pipeline attempted a yfinance download — must be cache-only")

    fake = types.ModuleType("yfinance")
    fake.download = _no_network
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def _write_synthetic_cache(root: Path, wide: pd.DataFrame) -> None:
    """Write a wide price panel to ``root``/data_cache/prices.parquet via the interchange contract."""
    long = interchange.to_long(wide, "prices")
    interchange.write_frame(long, str(root / "data_cache" / "prices.parquet"), "prices")


@pytest.fixture()
def ragged_prices() -> pd.DataFrame:
    """5-day, 3-ticker panel with a pre-IPO NaN run and a large day-1 move — hand-checkable."""
    idx = pd.date_range("2015-03-02", periods=5, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 99.0, 99.0, 121.0],
            "BBB": [50.0, 50.0, 55.0, 55.0, 50.0],
            "CCC": [np.nan, np.nan, 10.0, 11.0, 11.0],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------------------------
# run_pipeline: executed for real, offline, against a primed synthetic cache
# ---------------------------------------------------------------------------------------------


def test_run_pipeline_offline_matches_hand_composed_pipeline(tmp_path, monkeypatch):
    """run_pipeline(cfg) on a primed cache == loader → to_wide → strategy → engine by hand.

    This is the integration test the builder could not run (no cache on the machine): prime a
    synthetic cache in a temp cwd, poison yfinance, and require the cached UI pipeline to
    reproduce the manually composed pipeline exactly. Any hidden transformation, re-ordering,
    or network access inside the UI path breaks this.
    """
    rng = np.random.default_rng(7)
    idx = pd.date_range("2015-01-05", periods=60, freq="B", tz="UTC")
    wide = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=(60, 3)), axis=0)),
        index=idx,
        columns=["AAA", "BBB", "CCC"],
    )
    _write_synthetic_cache(tmp_path, wide)
    monkeypatch.chdir(tmp_path)
    _poison_yfinance(monkeypatch)
    # st.cache_data persists across tests in-process; clear so this test owns what it measures.
    app._load_wide_prices.clear()
    app.run_pipeline.clear()

    cfg = {
        "tickers": ["AAA", "BBB", "CCC"],
        "start": "2015-01-05",
        "end": "2015-12-31",
        "strategy": "momentum",
        "params": {"lookback": 20, "top_n": 1},
        "cost_bps": 10.0,
        "engine": "python",
    }
    result = app.run_pipeline(cfg)
    assert isinstance(result, BacktestResult)

    # Hand-composed reference pipeline over the identical window.
    long = load_prices(cfg["tickers"], start=cfg["start"], end=cfg["end"], cache_dir="data_cache")
    ref_wide = interchange.to_wide(long, "prices")
    params = validate_params("momentum", cfg["params"])
    positions = STRATEGIES["momentum"]().generate_signals(ref_wide, params)
    expected = PythonEngine().run_backtest(ref_wide, positions, {"cost_bps": 10.0})

    pd.testing.assert_series_equal(result.equity_curve, expected.equity_curve)
    pd.testing.assert_series_equal(result.returns, expected.returns)
    assert result.metrics == expected.metrics
    assert len(result.equity_curve) == 60  # the whole cached window survived the slice

    # Second call with the same cfg is a cache hit — identical output (and still no network,
    # which the poisoned yfinance would loudly report).
    again = app.run_pipeline(cfg)
    pd.testing.assert_series_equal(again.equity_curve, result.equity_curve)


def test_run_pipeline_holdout_window_stays_out_of_reach_of_cached_data(tmp_path, monkeypatch):
    """Even with a cache spanning into the holdout, the UI-bounded window slices it out.

    The widgets are the enforcement (RG-4), but this proves the plumbing under them: a cfg
    built from the sidebar bounds can never see a row after UI_DATE_MAX even when the cache
    file physically contains holdout rows.
    """
    idx = pd.date_range("2022-12-01", periods=40, freq="B", tz="UTC")  # spills into 2023
    wide = pd.DataFrame(
        {"AAA": np.linspace(100.0, 110.0, 40), "BBB": np.linspace(50.0, 60.0, 40)}, index=idx
    )
    _write_synthetic_cache(tmp_path, wide)
    monkeypatch.chdir(tmp_path)
    _poison_yfinance(monkeypatch)
    app._load_wide_prices.clear()
    app.run_pipeline.clear()

    cfg = {
        "tickers": ["AAA", "BBB"],
        "start": "2022-12-01",
        "end": app.UI_DATE_MAX.isoformat(),  # the widget's hard maximum
        "strategy": "momentum",
        "params": {},
        "cost_bps": 0.0,
        "engine": "python",
    }
    result = app.run_pipeline(cfg)
    holdout_start = pd.Timestamp(get_split_bounds()["holdout"][0], tz="UTC")
    assert result.equity_curve.index.max() < holdout_start


# ---------------------------------------------------------------------------------------------
# Equal-weight benchmark: full hand computation on a ragged (pre-IPO NaN) panel
# ---------------------------------------------------------------------------------------------


def test_benchmark_hand_computed_on_ragged_panel(ragged_prices):
    """Every point of the benchmark equals the hand-compounded 1/n book under the engine's
    documented NaN policy (NaN price ⇒ 0 return; capital parked on it sits idle), computed
    here from raw price floats without pandas pct_change or the engine."""
    bench = app.equal_weight_benchmark(ragged_prices)

    w = 1.0 / 3.0
    r_aaa = [0.0, 110.0 / 100.0 - 1.0, 99.0 / 110.0 - 1.0, 0.0, 121.0 / 99.0 - 1.0]
    r_bbb = [0.0, 0.0, 55.0 / 50.0 - 1.0, 0.0, 50.0 / 55.0 - 1.0]
    # CCC: NaN→NaN and NaN→10.0 pct_changes are NaN ⇒ 0 under the engine's fillna policy;
    # its first real return is day 3 (11/10 − 1).
    r_ccc = [0.0, 0.0, 0.0, 11.0 / 10.0 - 1.0, 0.0]

    equity = 1.0
    expected = []
    for t in range(5):
        # held weight on day t is the weight decided at close of t−1 ⇒ day 0 earns nothing.
        rp = 0.0 if t == 0 else w * r_aaa[t] + w * r_bbb[t] + w * r_ccc[t]
        equity *= 1.0 + rp
        expected.append(equity)

    assert bench.iloc[0] == pytest.approx(1.0, rel=1e-12)  # the +10% day-1 move is NOT pulled
    # back to day 0 — no look-ahead, and no costs are deducted anywhere (costless benchmark).
    for t in range(5):
        assert bench.iloc[t] == pytest.approx(expected[t], rel=1e-12), f"day {t}"

    # And the plotted trace is the same series, unmodified.
    strategy_result = BacktestResult(
        equity_curve=bench, returns=bench.pct_change().fillna(0.0), metrics={}, meta={}
    )
    fig = app.plot_equity(strategy_result, ragged_prices)
    np.testing.assert_allclose(np.asarray(fig.data[1].y, dtype=float), expected, rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# plot_frontier: the divide-by-zero attack on the max-Sharpe marker
# ---------------------------------------------------------------------------------------------


def test_frontier_star_ignores_zero_risk_row(ragged_prices):
    """A risk-0 row (naive ret/risk = +inf) must never win the max-Sharpe star; the star lands
    on the best finite ratio instead. Guards the documented rf=0 tangency logic against the
    degenerate frame a solver hiccup could emit."""
    frame = pd.DataFrame({"risk": [0.0, 0.10, 0.20], "ret": [0.90, 0.05, 0.06]})
    weights = pd.Series([0.5, 0.5], index=["momentum", "mean_reversion"])
    fig = app.plot_frontier(frame, weights)

    star = fig.data[1]
    assert len(star.x) == 1
    assert star.x[0] == pytest.approx(0.10)  # 0.05/0.10 = 0.5 beats 0.06/0.20 = 0.3
    assert star.y[0] == pytest.approx(0.05)


# ---------------------------------------------------------------------------------------------
# Rendered-widget enforcement (streamlit.testing.v1): RG-4 bounds and SF-3 slider ranges
# ---------------------------------------------------------------------------------------------


def _fresh_apptest():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=30)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_rendered_date_pickers_cannot_reach_the_holdout():
    """The ACTUAL date_input widgets carry min/max equal to the train start / validation end —
    the holdout is unselectable in the rendered UI, not merely in a helper constant."""
    at = _fresh_apptest()
    bounds = get_split_bounds()
    for picker in at.date_input:
        assert picker.min == date.fromisoformat(bounds["train"][0])
        assert picker.max == date.fromisoformat(bounds["validation"][1])
    assert date.fromisoformat(bounds["holdout"][0]) > at.date_input[1].max


def test_rendered_widget_ranges_equal_whitelist_for_every_strategy():
    """For each vetted strategy, the sliders/selectboxes Streamlit actually renders carry
    exactly PARAM_WHITELIST's bounds, defaults, and choices — the rendered-widget half of the
    'introspection, not duplication' requirement (the pure-helper half lives in
    tests/test_app_shell.py)."""
    at = _fresh_apptest()
    for strategy, whitelist in PARAM_WHITELIST.items():
        at.selectbox[0].select(strategy)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        sliders = {s.label: s for s in at.slider}
        boxes = {b.label: b for b in at.selectbox}
        for param, spec in whitelist.items():
            if "choices" in spec:
                assert boxes[param].options == sorted(spec["choices"])
                assert boxes[param].value == spec["default"]
            else:
                s = sliders[param]
                assert float(s.min) == float(spec["min"])
                assert float(s.max) == float(spec["max"])
                assert s.value == spec["default"]


def test_missing_cache_warns_and_never_downloads(monkeypatch, tmp_path):
    """With no price cache in cwd and yfinance poisoned, the full script run degrades to
    plain-language warnings on both compute tabs — zero exceptions, zero network."""
    monkeypatch.chdir(tmp_path)  # guarantees data_cache/prices.parquet is absent
    _poison_yfinance(monkeypatch)
    at = _fresh_apptest()
    cache_warnings = [w for w in at.warning if "Price cache not found" in str(w.value)]
    assert len(cache_warnings) == 2  # Backtest tab + Portfolio tab

    # Engine selector: python is the only selectable option; R/KNIME appear only as
    # labeled-disabled captions (the polyglot story stays visible but unclickable).
    engine_boxes = [b for b in at.selectbox if b.label == "Engine"]
    assert len(engine_boxes) == 1 and len(engine_boxes[0].options) == 1
    assert any("week 6+" in str(c.value) for c in at.caption)
