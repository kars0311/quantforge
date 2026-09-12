"""Tests for the four MCP tool functions in ``quantforge.ai.mcp_server`` (component 11).

Fully offline: an autouse fixture points ``mcp_server._price_source`` at a synthetic long
"prices" frame (four UNIVERSE tickers, business days 2010-01-01..2026-06-30, geometric random
walk) and clears the handle registries, so nothing here reads ``data_cache/prices.parquet`` or
touches the network. The synthetic panel deliberately extends INTO the holdout years so the
SF-8 rejection tests prove the tool refuses those dates even though the data exists.
"""

from __future__ import annotations

import inspect
import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange
from quantforge.ai import guardrails, mcp_server
from quantforge.data import loader
from quantforge.engine.base import Engine
from quantforge.metrics.performance import _KEYS
from quantforge.portfolio import optimize
from quantforge.strategies import STRATEGIES, validate_params

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM"]
SYN_START, SYN_END = "2010-01-01", "2026-06-30"
HANDLE_RE = re.compile(r"^(ds|res)_[0-9a-f]{16}$")


def _synthetic_long(seed: int = 0) -> pd.DataFrame:
    """Long interchange 'prices' frame on business days, tz-aware UTC, for TICKERS."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(SYN_START, SYN_END, tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), len(TICKERS)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    wide = pd.DataFrame(prices, index=dates, columns=TICKERS)
    return interchange.to_long(wide, "prices")


_SYNTHETIC = _synthetic_long()  # built once; tools never mutate it


@pytest.fixture(autouse=True)
def synthetic_source(monkeypatch):
    """Every test: synthetic prices, empty registries, PUBLIC_MODE off unless a test sets it."""
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _SYNTHETIC)
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    mcp_server.reset_registry()
    yield
    mcp_server.reset_registry()


@pytest.fixture
def dataset() -> dict:
    return mcp_server.load_data("2015-01-01", "2019-12-31", ["AAPL", "MSFT", "NVDA"])


@pytest.fixture
def two_results(dataset) -> tuple[str, str]:
    ds = dataset["dataset_id"]
    r1 = mcp_server.run_backtest(ds, "momentum", {"lookback": 60})["result_id"]
    r2 = mcp_server.run_backtest(ds, "mean_reversion", {"lookback": 20})["result_id"]
    return r1, r2


def _expected_rows(start: str, end: str) -> int:
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    return int(_SYNTHETIC["date"].between(lo, hi).sum() / len(TICKERS))


# ---------------------------------------------------------------- module surface


def test_tools_and_engine_registry():
    assert mcp_server.TOOLS == ["load_data", "run_backtest", "optimize_portfolio", "get_metrics"]
    assert set(mcp_server.ENGINES) == {"python"}
    assert isinstance(mcp_server.ENGINES["python"], Engine)


def test_unknown_handles_raise_valueerror_not_keyerror():
    with pytest.raises(ValueError, match="dataset_id"):
        mcp_server._get_dataset("ds_deadbeefdeadbeef")
    with pytest.raises(ValueError, match="result_id"):
        mcp_server._get_result("res_deadbeefdeadbeef")
    with pytest.raises(ValueError, match="result_id"):
        mcp_server.get_metrics("res_deadbeefdeadbeef")
    with pytest.raises(ValueError, match="dataset_id"):
        mcp_server.run_backtest("ds_deadbeefdeadbeef", "momentum", {})


def test_reset_registry_forgets_handles(dataset):
    ds = dataset["dataset_id"]
    mcp_server._get_dataset(ds)
    mcp_server.reset_registry()
    with pytest.raises(ValueError, match="unknown dataset_id"):
        mcp_server._get_dataset(ds)


# ---------------------------------------------------------------- load_data


def test_load_data_happy_path():
    out = mcp_server.load_data("2015-01-01", "2019-12-31", ["MSFT", "AAPL"])
    assert set(out) == {"dataset_id", "tickers", "start", "end", "n_days"}
    assert HANDLE_RE.match(out["dataset_id"]) and out["dataset_id"].startswith("ds_")
    assert out["tickers"] == ["AAPL", "MSFT"]  # sorted, only those requested
    assert out["start"] == "2015-01-01" and out["end"] == "2019-12-31"
    assert out["n_days"] == _expected_rows("2015-01-01", "2019-12-31")
    json.dumps(out)

    wide = mcp_server._get_dataset(out["dataset_id"])
    assert list(wide.columns) == ["AAPL", "MSFT"]
    assert wide.index[0] >= pd.Timestamp("2015-01-01", tz="UTC")
    assert wide.index[-1] <= pd.Timestamp("2019-12-31", tz="UTC")


def test_load_data_defaults_to_universe_and_reports_present_tickers():
    out = mcp_server.load_data("2015-01-01", "2015-12-31")
    # UNIVERSE has 29 names; the synthetic panel has four — only present names are reported.
    assert out["tickers"] == sorted(TICKERS)
    assert out["n_days"] == _expected_rows("2015-01-01", "2015-12-31")


def test_load_data_full_train_plus_validation_window_is_allowed():
    bounds = loader.get_split_bounds()
    out = mcp_server.load_data(bounds["train"][0], bounds["validation"][1])
    assert out["n_days"] == _expected_rows(bounds["train"][0], bounds["validation"][1])


@pytest.mark.parametrize(
    "tickers, match",
    [
        (["AAPL", "ZZZZ"], "ZZZZ"),
        (["AAPL", 42], "42"),
        ([], "non-empty"),
        (["AAPL", "AAPL"], "duplicate"),
        ("AAPL", "list"),
    ],
)
def test_load_data_rejects_bad_tickers(tickers, match):
    with pytest.raises(ValueError, match=match):
        mcp_server.load_data("2015-01-01", "2016-01-01", tickers)


@pytest.mark.parametrize(
    "start, end, match",
    [
        ("2015/01/01", "2016-01-01", "start"),
        ("2015-01-01", "2016/01/01", "end"),
        ("2015-1-1", "2016-01-01", "start"),
        ("2015-02-30", "2016-01-01", "calendar"),
        (20150101, "2016-01-01", "start"),
        ("2016-01-01", "2015-01-01", "after"),
    ],
)
def test_load_data_rejects_bad_dates(start, end, match):
    with pytest.raises(ValueError, match=match):
        mcp_server.load_data(start, end, ["AAPL"])


@pytest.mark.parametrize(
    "start, end",
    [
        ("2023-01-01", "2023-06-30"),  # entirely inside the holdout
        ("2020-01-01", "2023-01-01"),  # ends one day into the holdout
        ("2015-01-01", "2026-06-30"),  # spans through the holdout
        ("2009-01-01", "2015-01-01"),  # before train.start
    ],
)
def test_load_data_rejects_dates_outside_train_validation(start, end):
    """SF-8: reject, never truncate — even though the synthetic panel HAS those dates."""
    bounds = loader.get_split_bounds()
    with pytest.raises(ValueError) as excinfo:
        mcp_server.load_data(start, end, ["AAPL"])
    msg = str(excinfo.value)
    assert "holdout" in msg
    assert bounds["train"][0] in msg and bounds["validation"][1] in msg
    assert mcp_server._DATASETS == {}  # nothing was stored


def test_load_data_rejects_empty_slice(monkeypatch):
    """A window with no rows must not produce a handle to an empty frame."""
    empty = _SYNTHETIC.iloc[0:0]
    monkeypatch.setattr(mcp_server, "_price_source", lambda: empty)
    with pytest.raises(ValueError, match="no price rows"):
        mcp_server.load_data("2015-01-01", "2015-12-31", ["AAPL"])


def test_load_data_never_reads_loader_on_synthetic_source(monkeypatch):
    """Belt and braces: the loader is unreachable while the source is monkeypatched."""

    def boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("loader.load_prices must not be called")

    monkeypatch.setattr(loader, "load_prices", boom)
    mcp_server.load_data("2015-01-01", "2015-12-31", ["AAPL"])


# ---------------------------------------------------------------- run_backtest


@pytest.mark.parametrize("strategy", ["momentum", "mean_reversion"])
def test_run_backtest_happy_path(dataset, strategy):
    out = mcp_server.run_backtest(dataset["dataset_id"], strategy)
    assert set(out) == {"result_id", "metrics"}
    assert HANDLE_RE.match(out["result_id"]) and out["result_id"].startswith("res_")
    assert set(out["metrics"]) == set(_KEYS)
    assert all(isinstance(v, float) for v in out["metrics"].values())
    json.dumps(out)


def test_run_backtest_pipeline_matches_direct_engine_call(dataset):
    """The tool runs exactly the UI's pipeline: signals -> engine with cost_bps merged in."""
    ds = dataset["dataset_id"]
    out = mcp_server.run_backtest(ds, "momentum", {"lookback": 60}, cost_bps=25)
    wide = mcp_server._get_dataset(ds)
    params = validate_params("momentum", {"lookback": 60})
    positions = STRATEGIES["momentum"]().generate_signals(wide, params)
    direct = mcp_server.ENGINES["python"].run_backtest(
        wide, positions, {"cost_bps": 25.0, "strategy": "momentum", **params}
    )
    assert out["metrics"] == direct.metrics
    stored = mcp_server._get_result(out["result_id"])
    assert stored.meta["cost_bps"] == 25.0
    assert stored.meta["params"]["lookback"] == 60


def test_run_backtest_is_deterministic(dataset):
    ds = dataset["dataset_id"]
    a = mcp_server.run_backtest(ds, "mean_reversion", {"lookback": 15, "entry_z": 1.5})
    b = mcp_server.run_backtest(ds, "mean_reversion", {"lookback": 15, "entry_z": 1.5})
    assert a["metrics"] == b["metrics"]
    assert a["result_id"] != b["result_id"]  # fresh handle per run


def test_run_backtest_rejects_unvetted_strategy(dataset):
    with pytest.raises(ValueError, match="unknown strategy 'pairs'"):
        mcp_server.run_backtest(dataset["dataset_id"], "pairs", {})
    with pytest.raises(ValueError, match="strategy"):
        mcp_server.run_backtest(dataset["dataset_id"], ["momentum"], {})


@pytest.mark.parametrize(
    "strategy, params, match",
    [
        ("momentum", {"lookback": 999}, r"'lookback'=999 outside \[20, 252\]"),
        ("momentum", {"window": 60}, r"unknown param\(s\) \['window'\]"),
        ("momentum", {"lookback": True}, r"got bool True"),
        ("mean_reversion", {"mode": "leveraged"}, r"'mode' must be one of"),
        ("momentum", {"lookback": 60.5}, r"must be an integer"),
    ],
)
def test_run_backtest_preserves_validate_params_messages(dataset, strategy, params, match):
    with pytest.raises(ValueError, match=match):
        mcp_server.run_backtest(dataset["dataset_id"], strategy, params)


def test_run_backtest_rejects_non_dict_params(dataset):
    with pytest.raises(ValueError, match="params"):
        mcp_server.run_backtest(dataset["dataset_id"], "momentum", [("lookback", 60)])


def test_run_backtest_rejects_unknown_engine(dataset):
    with pytest.raises(ValueError, match="unknown engine 'matlab'.*python"):
        mcp_server.run_backtest(dataset["dataset_id"], "momentum", {}, engine="matlab")


@pytest.mark.parametrize("cost_bps", [-1, 101, True, "10", None, float("nan")])
def test_run_backtest_rejects_bad_cost_bps(dataset, cost_bps):
    with pytest.raises(ValueError, match="cost_bps"):
        mcp_server.run_backtest(dataset["dataset_id"], "momentum", {}, cost_bps=cost_bps)


@pytest.mark.parametrize("cost_bps", [0, 100, 10.5])
def test_run_backtest_accepts_boundary_cost_bps(dataset, cost_bps):
    out = mcp_server.run_backtest(dataset["dataset_id"], "momentum", {}, cost_bps=cost_bps)
    assert mcp_server._get_result(out["result_id"]).meta["cost_bps"] == float(cost_bps)


def test_run_backtest_rejections_store_nothing(dataset):
    for bad in ({"lookback": 999}, {"code": "x"}):
        with pytest.raises(ValueError):
            mcp_server.run_backtest(dataset["dataset_id"], "momentum", bad)
    assert mcp_server._RESULTS == {}


def test_run_backtest_public_mode_rejects_nested_params(dataset, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    with pytest.raises(guardrails.PublicModeViolation, match="flat"):
        mcp_server.run_backtest(dataset["dataset_id"], "momentum", {"lookback": {"n": 60}})
    with pytest.raises(guardrails.PublicModeViolation):
        mcp_server.run_backtest(dataset["dataset_id"], "pairs", {})
    # Parameter-only requests still work in public mode.
    out = mcp_server.run_backtest(dataset["dataset_id"], "momentum", {"lookback": 60})
    assert set(out["metrics"]) == set(_KEYS)


# ---------------------------------------------------------------- get_metrics


def test_get_metrics_returns_same_dict_as_run(dataset):
    run = mcp_server.run_backtest(dataset["dataset_id"], "momentum", {"lookback": 60})
    out = mcp_server.get_metrics(run["result_id"])
    assert set(out) == {"metrics"}
    assert out["metrics"] == run["metrics"]
    assert out["metrics"] is not mcp_server._get_result(run["result_id"]).metrics  # a copy
    json.dumps(out)


# ---------------------------------------------------------------- optimize_portfolio


def _check_portfolio_output(out: dict, labels: set[str]) -> None:
    assert set(out) == {"weights", "frontier"}
    assert set(out["weights"]) == labels
    assert all(isinstance(w, float) for w in out["weights"].values())
    assert math.isclose(sum(out["weights"].values()), 1.0, abs_tol=1e-8)
    assert all(0.0 <= w <= 1.0 for w in out["weights"].values())
    assert isinstance(out["frontier"], list) and len(out["frontier"]) == 20
    for point in out["frontier"]:
        assert set(point) == set(interchange.SCHEMAS["frontier"].names) == {"risk", "ret"}
        assert all(isinstance(v, float) and math.isfinite(v) for v in point.values())
    json.dumps(out)


def test_optimize_over_result_ids(two_results):
    r1, r2 = two_results
    out = mcp_server.optimize_portfolio(result_ids=[r1, r2])
    _check_portfolio_output(out, {r1, r2})


def test_optimize_over_dataset(dataset):
    out = mcp_server.optimize_portfolio(
        dataset_id=dataset["dataset_id"], objective="min_volatility"
    )
    _check_portfolio_output(out, {"AAPL", "MSFT", "NVDA"})


def test_optimize_over_dataset_matches_direct_optimizer(dataset):
    """The dataset path is buy-and-hold asset returns fed to portfolio.optimize, nothing else."""
    out = mcp_server.optimize_portfolio(dataset_id=dataset["dataset_id"])
    panel = mcp_server._get_dataset(dataset["dataset_id"]).pct_change().dropna()
    direct = optimize.optimize_weights(panel, {"objective": "max_sharpe"})
    assert out["weights"] == {k: float(v) for k, v in direct.items()}


def test_optimize_rejects_single_result_id(two_results):
    r1, _ = two_results
    with pytest.raises(ValueError, match="at least 2 result_ids.*got 1"):
        mcp_server.optimize_portfolio(result_ids=[r1])


def test_optimize_rejects_duplicate_and_unknown_result_ids(two_results):
    r1, r2 = two_results
    with pytest.raises(ValueError, match="duplicate"):
        mcp_server.optimize_portfolio(result_ids=[r1, r1])
    with pytest.raises(ValueError, match="unknown result_id"):
        mcp_server.optimize_portfolio(result_ids=[r1, "res_deadbeefdeadbeef"])
    with pytest.raises(ValueError, match="result_ids"):
        mcp_server.optimize_portfolio(result_ids=r1 + "," + r2)


def test_optimize_rejects_both_or_neither_selector(dataset, two_results):
    with pytest.raises(ValueError, match="exactly one"):
        mcp_server.optimize_portfolio()
    with pytest.raises(ValueError, match="exactly one"):
        mcp_server.optimize_portfolio(
            result_ids=list(two_results), dataset_id=dataset["dataset_id"]
        )


def test_optimize_rejects_bad_objective(two_results):
    with pytest.raises(ValueError, match="objective 'max_return'"):
        mcp_server.optimize_portfolio(result_ids=list(two_results), objective="max_return")


def test_optimize_rejects_unknown_or_single_ticker_dataset():
    with pytest.raises(ValueError, match="unknown dataset_id"):
        mcp_server.optimize_portfolio(dataset_id="ds_deadbeefdeadbeef")
    single = mcp_server.load_data("2015-01-01", "2016-12-31", ["AAPL"])
    with pytest.raises(ValueError, match="at least 2 tickers"):
        mcp_server.optimize_portfolio(dataset_id=single["dataset_id"])


# ---------------------------------------------------------------- safety invariants


def test_every_tool_output_is_json_and_carries_no_frames(two_results, dataset):
    r1, r2 = two_results
    outputs = [
        dataset,
        mcp_server.run_backtest(dataset["dataset_id"], "momentum"),
        mcp_server.get_metrics(r1),
        mcp_server.optimize_portfolio(result_ids=[r1, r2]),
        mcp_server.optimize_portfolio(dataset_id=dataset["dataset_id"]),
    ]
    text = json.dumps(outputs)  # raises TypeError on any DataFrame/Series/ndarray
    assert "DataFrame" not in text


def test_handles_are_opaque_random_tokens(dataset, two_results):
    ids = [dataset["dataset_id"], *two_results]
    assert all(HANDLE_RE.match(i) for i in ids)
    assert len(set(ids)) == 3
    assert not any("/" in i or "." in i for i in ids)  # not a path


def test_module_contains_no_file_or_code_execution_calls():
    src = inspect.getsource(mcp_server)
    assert not re.search(r"\b(open|eval|exec)\(|subprocess", src)
