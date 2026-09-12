"""Independent verification of the four MCP tool functions (component 11) — adversarial cases.

Written separately from ``tests/test_mcp_tools.py`` with its OWN synthetic panel so the two files
cannot share a blind spot. Everything is offline: an autouse fixture points
``mcp_server._price_source`` at a synthetic long frame and empties the handle registries.

What this file proves beyond the builder's suite:

* the tool path keeps the one-day execution lag — a deliberately *prescient* strategy injected
  through the registry earns the hand-computed lagged return, and the unlagged "cheat" dwarfs it;
* transaction costs are actually charged on the tool path (hand-computed drag, 0 vs 100 bps);
* ``n_days`` matches an independent business-day count;
* the SF-8 window is exactly ``[train.start, validation.end]`` inclusive, off-by-one both sides;
* code / path / SQL strings pushed through every string argument are rejected with ValueError;
* a ragged panel (a ticker that IPOs mid-window, like META in the real cache) round-trips;
* the production ``_price_source`` is wired to ``loader.load_prices`` — proved without reading
  the Parquet cache.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange, strategies
from quantforge.ai import mcp_server
from quantforge.data import loader
from quantforge.engine.base import Engine, Strategy
from quantforge.metrics.performance import _KEYS

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
PANEL_START, PANEL_END = "2010-01-01", "2026-06-30"
HANDLE_RE = re.compile(r"^(ds|res)_[0-9a-f]{16}$")


def _make_wide(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(PANEL_START, PANEL_END, tz="UTC")
    rets = rng.normal(0.0003, 0.012, size=(len(dates), len(TICKERS)))
    prices = 50.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=TICKERS)


_WIDE = _make_wide()
_LONG = interchange.to_long(_WIDE, "prices")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _LONG)
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    mcp_server.reset_registry()
    yield
    mcp_server.reset_registry()


@pytest.fixture
def ds_id() -> str:
    return mcp_server.load_data("2015-01-01", "2019-12-31", ["AAPL", "MSFT", "NVDA"])["dataset_id"]


# ---------------------------------------------------------------- registry / wiring


def test_engine_registry_is_exactly_python():
    assert set(mcp_server.ENGINES) == {"python"}
    assert isinstance(mcp_server.ENGINES["python"], Engine)
    assert mcp_server.TOOLS == ["load_data", "run_backtest", "optimize_portfolio", "get_metrics"]


def test_default_price_source_is_the_loader(monkeypatch):
    """Production wiring, proved without touching the cache: the default source is load_prices."""
    calls: list[dict] = []

    def fake_load_prices(*args, **kwargs):
        calls.append(kwargs)
        return _LONG

    monkeypatch.setattr(loader, "load_prices", fake_load_prices)
    # Bypass the autouse patch on purpose: call the *default* source directly.
    out = mcp_server._default_price_source()
    assert out is _LONG
    assert calls == [{}]  # the FULL panel is requested; slicing happens in load_data


def test_reset_registry_is_documented_tests_only():
    assert "TESTS ONLY" in (mcp_server.reset_registry.__doc__ or "")


# ---------------------------------------------------------------- load_data


def test_n_days_matches_independent_business_day_count():
    out = mcp_server.load_data("2015-01-01", "2019-12-31", ["AAPL", "MSFT"])
    expected = len(pd.bdate_range("2015-01-01", "2019-12-31"))  # hand count, no tool involved
    assert out["n_days"] == expected == 1304
    assert set(out) == {"dataset_id", "tickers", "start", "end", "n_days"}
    assert out["tickers"] == ["AAPL", "MSFT"]


def test_window_bounds_are_inclusive_and_off_by_one_rejected():
    bounds = loader.get_split_bounds()
    lo, hi = bounds["train"][0], bounds["validation"][1]
    assert (lo, hi) == ("2010-01-01", "2022-12-31")

    ok = mcp_server.load_data(lo, hi, ["AAPL"])
    assert ok["n_days"] == len(pd.bdate_range(lo, hi))
    wide = mcp_server._get_dataset(ok["dataset_id"])
    # 2022-12-31 is a Saturday: the last row must be the last *business* day of validation.
    assert (
        wide.index[-1]
        == pd.bdate_range(lo, hi, tz="UTC")[-1]
        == pd.Timestamp("2022-12-30", tz="UTC")
    )

    for start, end in [("2009-12-31", hi), (lo, "2023-01-01"), ("2023-01-01", "2023-01-01")]:
        with pytest.raises(ValueError) as excinfo:
            mcp_server.load_data(start, end, ["AAPL"])
        msg = str(excinfo.value)
        assert "holdout" in msg and lo in msg and hi in msg
    assert len(mcp_server._DATASETS) == 1  # only the valid load left a handle behind


def test_load_data_never_truncates_to_the_window():
    """Asking for 2015..2026 must not quietly return 2015..2022 — reject, do not clamp."""
    with pytest.raises(ValueError, match="holdout"):
        mcp_server.load_data("2015-01-01", "2026-06-30", ["AAPL"])
    assert mcp_server._DATASETS == {}


@pytest.mark.parametrize(
    "tickers",
    [
        ["aapl"],  # case-sensitive: not the UNIVERSE symbol
        ["AAPL' OR 1=1 --"],  # SQL-shaped
        ["../data_cache/prices.parquet"],  # path-shaped
        ["__import__('os')"],  # code-shaped
        ("AAPL", "MSFT"),  # tuple, not list
        {"AAPL"},  # set, not list
    ],
)
def test_load_data_rejects_hostile_or_wrong_shaped_tickers(tickers):
    with pytest.raises(ValueError, match="ticker"):
        mcp_server.load_data("2015-01-01", "2015-12-31", tickers)


def test_load_data_rejects_datetime_objects_and_iso_with_time():
    with pytest.raises(ValueError, match="start"):
        mcp_server.load_data(pd.Timestamp("2015-01-01"), "2015-12-31", ["AAPL"])
    with pytest.raises(ValueError, match="end"):
        mcp_server.load_data("2015-01-01", "2015-12-31T00:00:00", ["AAPL"])


def test_load_data_ragged_panel_with_mid_window_ipo(monkeypatch):
    """A ticker with no rows before its IPO (META-style) loads, backtests, and optimizes."""
    ragged = _WIDE.copy()
    ipo = pd.Timestamp("2012-05-18", tz="UTC")
    ragged.loc[ragged.index < ipo, "XOM"] = np.nan
    monkeypatch.setattr(mcp_server, "_price_source", lambda: interchange.to_long(ragged, "prices"))

    out = mcp_server.load_data("2011-01-01", "2013-12-31", ["AAPL", "XOM"])
    assert out["tickers"] == ["AAPL", "XOM"]
    assert out["n_days"] == len(pd.bdate_range("2011-01-01", "2013-12-31"))
    wide = mcp_server._get_dataset(out["dataset_id"])
    assert wide["XOM"].isna().sum() == int((wide.index < ipo).sum()) > 0

    res = mcp_server.run_backtest(out["dataset_id"], "momentum", {"lookback": 60})
    assert set(res["metrics"]) == set(_KEYS)
    assert all(math.isfinite(v) for v in res["metrics"].values())

    port = mcp_server.optimize_portfolio(dataset_id=out["dataset_id"])
    assert math.isclose(sum(port["weights"].values()), 1.0, abs_tol=1e-8)


# ---------------------------------------------------------------- run_backtest rigor


class _PrescientStrategy(Strategy):
    """CHEATING on purpose: at close t, go 100% long the asset with the largest return ON day t.

    If the tool path ever dropped the one-day lag, this would book day t's winner return every
    day and the equity curve would explode. With the lag it merely holds yesterday's winner —
    no edge on a random walk.
    """

    name = "prescient"

    def generate_signals(self, prices: pd.DataFrame, params: dict) -> pd.DataFrame:
        rets = prices.pct_change().fillna(0.0)  # first row all-NaN would break idxmax
        winner = rets.idxmax(axis=1)
        positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for date, ticker in winner.items():
            if isinstance(ticker, str):
                positions.loc[date, ticker] = 1.0
        return positions


def test_tool_path_keeps_one_day_lag_prescient_strategy_earns_no_edge(monkeypatch, ds_id):
    monkeypatch.setitem(mcp_server.STRATEGIES, "prescient", _PrescientStrategy)
    monkeypatch.setitem(strategies.PARAM_WHITELIST, "prescient", {})

    out = mcp_server.run_backtest(ds_id, "prescient", {}, cost_bps=0.0)
    tool_returns = mcp_server._get_result(out["result_id"]).returns

    wide = mcp_server._get_dataset(ds_id)
    asset_rets = wide.pct_change().fillna(0.0)
    positions = _PrescientStrategy().generate_signals(wide, {})
    lagged = (positions.shift(1).fillna(0.0) * asset_rets).sum(axis=1)  # hand-computed, lag 1
    cheat = (positions * asset_rets).sum(axis=1)  # what NO lag would earn

    pd.testing.assert_series_equal(
        tool_returns.reset_index(drop=True), lagged.reset_index(drop=True), check_names=False
    )
    cheat_total = float((1.0 + cheat).prod() - 1.0)
    assert cheat_total > 1e3  # same-day max of 3 assets, compounded over 5y: absurd
    assert out["metrics"]["total_return"] < 10.0  # the lagged (honest) path is not
    assert cheat_total > 100 * (1.0 + out["metrics"]["total_return"])
    assert abs(out["metrics"]["sharpe"]) < 2.0


def test_costs_are_charged_on_the_tool_path(ds_id):
    free = mcp_server.run_backtest(ds_id, "momentum", {"lookback": 60}, cost_bps=0)
    dear = mcp_server.run_backtest(ds_id, "momentum", {"lookback": 60}, cost_bps=100)
    free_res = mcp_server._get_result(free["result_id"])
    dear_res = mcp_server._get_result(dear["result_id"])

    assert free_res.meta["cost_bps"] == 0.0 and dear_res.meta["cost_bps"] == 100.0
    assert free["metrics"]["total_return"] > dear["metrics"]["total_return"]
    # Hand-computed drag: the daily gap between the two net streams is turnover * 1%.
    turnover = (
        mcp_server.STRATEGIES["momentum"]()
        .generate_signals(mcp_server._get_dataset(ds_id), {"lookback": 60, "top_n": 0})
        .shift(1)
        .fillna(0.0)
        .diff()
        .abs()
        .sum(axis=1)
        .fillna(0.0)
    )
    assert turnover.sum() > 0
    gap = (free_res.returns - dear_res.returns).to_numpy()
    np.testing.assert_allclose(gap, turnover.to_numpy() * 0.01, atol=1e-12)


def test_metrics_keys_and_validate_params_message_preserved(ds_id):
    out = mcp_server.run_backtest(ds_id, "momentum", {"lookback": 60})
    assert list(out["metrics"]) == list(_KEYS)  # exactly _KEYS, in the canonical order
    assert all(type(v) is float for v in out["metrics"].values())
    with pytest.raises(ValueError, match=r"momentum: param 'lookback'=999 outside \[20, 252\]"):
        mcp_server.run_backtest(ds_id, "momentum", {"lookback": 999})
    assert len(mcp_server._RESULTS) == 1


@pytest.mark.parametrize("cost_bps", [True, False, 101, -0.001, 100.0001, "10", [10]])
def test_cost_bps_rejected(ds_id, cost_bps):
    with pytest.raises(ValueError, match="cost_bps"):
        mcp_server.run_backtest(ds_id, "momentum", {}, cost_bps=cost_bps)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"strategy": "__import__('os').system('id')"}, "unknown strategy"),
        ({"strategy": "momentum; import os"}, "unknown strategy"),
        ({"strategy": "momentum", "params": {"lookback": "60; import os"}}, "lookback"),
        ({"strategy": "momentum", "params": {"code": "print(1)"}}, "unknown param"),
        ({"strategy": "momentum", "engine": "python; rm -rf /"}, "unknown engine"),
        ({"strategy": "momentum", "engine": "../engine"}, "unknown engine"),
        ({"strategy": None}, "strategy"),
        ({"strategy": "momentum", "params": "lookback=60"}, "params"),
    ],
)
def test_run_backtest_rejects_code_and_path_shaped_inputs(ds_id, kwargs, match):
    with pytest.raises(ValueError, match=match):
        mcp_server.run_backtest(ds_id, **kwargs)
    assert mcp_server._RESULTS == {}


@pytest.mark.parametrize(
    "dataset_id",
    ["../../data_cache/prices.parquet", "ds_" + "0" * 16, "", None, 42, ["ds_x"]],
)
def test_run_backtest_rejects_unknown_or_path_like_dataset_ids(dataset_id):
    with pytest.raises(ValueError, match="unknown dataset_id") as excinfo:
        mcp_server.run_backtest(dataset_id, "momentum", {})
    assert not isinstance(excinfo.value, KeyError)


def test_run_backtest_default_params_equal_explicit_defaults(ds_id):
    """No params and whitelist defaults must mean the same backtest (registry contract)."""
    a = mcp_server.run_backtest(ds_id, "mean_reversion")
    b = mcp_server.run_backtest(
        ds_id,
        "mean_reversion",
        {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "mode": "long_flat"},
    )
    assert a["metrics"] == b["metrics"]


# ---------------------------------------------------------------- get_metrics


def test_get_metrics_unknown_is_valueerror_not_keyerror():
    with pytest.raises(ValueError, match="unknown result_id") as excinfo:
        mcp_server.get_metrics("res_deadbeefdeadbeef")
    assert not isinstance(excinfo.value, KeyError)


def test_get_metrics_output_is_a_copy(ds_id):
    rid = mcp_server.run_backtest(ds_id, "momentum")["result_id"]
    out = mcp_server.get_metrics(rid)
    out["metrics"]["sharpe"] = 99.0
    assert mcp_server.get_metrics(rid)["metrics"]["sharpe"] != 99.0
    assert mcp_server._get_result(rid).metrics["sharpe"] != 99.0


# ---------------------------------------------------------------- optimize_portfolio


def test_optimize_result_ids_matches_direct_optimizer_on_inner_join(ds_id):
    from quantforge.portfolio import optimize

    r1 = mcp_server.run_backtest(ds_id, "momentum", {"lookback": 60})["result_id"]
    r2 = mcp_server.run_backtest(ds_id, "mean_reversion", {"lookback": 20})["result_id"]
    out = mcp_server.optimize_portfolio(result_ids=[r1, r2])

    assert set(out) == {"weights", "frontier"}
    assert set(out["weights"]) == {r1, r2}
    assert math.isclose(sum(out["weights"].values()), 1.0, abs_tol=1e-8)
    assert len(out["frontier"]) == 20
    assert all(set(p) == set(interchange.SCHEMAS["frontier"].names) for p in out["frontier"])

    panel = pd.concat(
        {r1: mcp_server._get_result(r1).returns, r2: mcp_server._get_result(r2).returns},
        axis=1,
        join="inner",
    ).dropna()
    direct = optimize.optimize_weights(panel, {"objective": "max_sharpe"})
    assert out["weights"] == {k: float(v) for k, v in direct.items()}


def test_optimize_result_ids_from_different_windows_inner_joins(ds_id):
    other = mcp_server.load_data("2017-01-01", "2021-12-31", ["AAPL", "MSFT", "NVDA"])
    r1 = mcp_server.run_backtest(ds_id, "momentum", {"lookback": 60})["result_id"]
    r2 = mcp_server.run_backtest(other["dataset_id"], "momentum", {"lookback": 60})["result_id"]
    out = mcp_server.optimize_portfolio(result_ids=[r1, r2], objective="min_volatility")
    assert math.isclose(sum(out["weights"].values()), 1.0, abs_tol=1e-8)


def test_optimize_result_ids_from_disjoint_windows_is_a_valueerror(ds_id):
    later = mcp_server.load_data("2020-01-01", "2022-12-31", ["AAPL", "MSFT"])
    r1 = mcp_server.run_backtest(ds_id, "momentum")["result_id"]
    r2 = mcp_server.run_backtest(later["dataset_id"], "momentum")["result_id"]
    with pytest.raises(ValueError):  # empty inner join must not surface as a crash
        mcp_server.optimize_portfolio(result_ids=[r1, r2])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"result_ids": []}, "got 0"),
        ({"result_ids": ["res_deadbeefdeadbeef", "res_cafebabecafebabe"]}, "unknown result_id"),
        ({"result_ids": ["../x", "../y"]}, "unknown result_id"),
        ({"dataset_id": "../../data_cache/prices.parquet"}, "unknown dataset_id"),
        ({"result_ids": None, "dataset_id": None}, "exactly one"),
        ({"dataset_id": "ds_x", "objective": "max_sharpe; import os"}, "objective"),
    ],
)
def test_optimize_rejections(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mcp_server.optimize_portfolio(**kwargs)


def test_optimize_single_result_id_names_the_count(ds_id):
    rid = mcp_server.run_backtest(ds_id, "momentum")["result_id"]
    with pytest.raises(ValueError, match="got 1"):
        mcp_server.optimize_portfolio(result_ids=[rid])


# ---------------------------------------------------------------- serialization / handles


def test_all_outputs_json_roundtrip_to_plain_python(ds_id):
    r1 = mcp_server.run_backtest(ds_id, "momentum", {"lookback": 60})
    r2 = mcp_server.run_backtest(ds_id, "mean_reversion", {"lookback": 20})
    outputs = {
        "load": mcp_server.load_data("2015-01-01", "2015-12-31", ["AAPL", "MSFT"]),
        "run": r1,
        "metrics": mcp_server.get_metrics(r1["result_id"]),
        "opt_res": mcp_server.optimize_portfolio(result_ids=[r1["result_id"], r2["result_id"]]),
        "opt_ds": mcp_server.optimize_portfolio(dataset_id=ds_id),
    }
    text = json.dumps(outputs, allow_nan=False)  # NaN/inf would also be a contract violation
    assert json.loads(text) == outputs  # nothing but str/float/int/list/dict survived
    assert "close" not in outputs["load"]  # no prices ride in a tool output


def test_handles_are_unique_random_tokens():
    ids = [
        mcp_server.load_data("2015-01-01", "2015-03-31", ["AAPL"])["dataset_id"] for _ in range(25)
    ]
    assert all(HANDLE_RE.match(i) for i in ids)
    assert len(set(ids)) == 25


def test_module_source_has_no_file_or_code_execution():
    import inspect

    src = inspect.getsource(mcp_server)
    assert re.search(r"\b(open|eval|exec)\(|subprocess", src) is None
