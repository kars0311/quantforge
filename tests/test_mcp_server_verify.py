"""Independent verification of the MCP schema/discovery layer (component 11) — wire-level cases.

The builder's ``tests/test_mcp_server.py`` proves ``TOOL_SCHEMAS`` and the in-process
``FastMCP.call_tool`` path. This file goes one layer further and drives a REAL MCP client/server
session over the SDK's in-memory transport (``mcp.shared.memory``), which is the code path an
external client (Claude Desktop over stdio) actually exercises: low-level request handlers,
``inputSchema`` validation, ``isError`` results. Its own synthetic panel (three tickers, extended
through the holdout years) so it cannot share a blind spot with the builder's suite. Fully
offline; nothing here opens stdio.

What this file proves beyond the builder's suite:

* discovery over the protocol advertises exactly ``TOOLS`` with the strict schemas;
* the strict schema is ENFORCED on the wire, not just advertised — an extra top-level key, a
  boolean or string ``cost_bps``, a ``params.code`` key and an unvetted strategy are all refused
  with an ``Input validation error`` before any function runs (FastMCP's default would have
  coerced/dropped them);
* a schema-valid but per-strategy-invalid param (momentum ``lookback=10``, inside the union
  schema's 5..252) is still refused by ``validate_params`` — the two gates compose;
* the SF-8 window is exact on the wire (off-by-one both sides) even though synthetic prices
  EXIST for the rejected dates, so the refusal is policy, not missing data;
* metrics through the transport equal the direct function call, and transaction costs are
  actually charged on that path (0 < 10 < 100 bps ordering on the same signal);
* ``optimize_portfolio`` over two wire-produced result handles returns fully-invested weights,
  and both/neither selector is refused;
* ``TOOL_SCHEMAS`` is a byte-stable prompt prefix (AR-6) across rebuilds and server instances.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable

import numpy as np
import pandas as pd
import pytest
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult

from quantforge import interchange
from quantforge.ai import mcp_server
from quantforge.data import loader
from quantforge.strategies import PARAM_WHITELIST

TICKERS = ["AAPL", "MSFT", "JPM"]
#: Deliberately wider than the split table on both sides so a rejected date has real rows.
SYN_START, SYN_END = "2009-06-01", "2026-06-30"


def _synthetic_long(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(SYN_START, SYN_END, tz="UTC")
    rets = rng.normal(0.0005, 0.012, size=(len(dates), len(TICKERS)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return interchange.to_long(pd.DataFrame(prices, index=dates, columns=TICKERS), "prices")


_SYNTHETIC = _synthetic_long()


@pytest.fixture(autouse=True)
def synthetic_source(monkeypatch):
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _SYNTHETIC)
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    mcp_server.reset_registry()
    yield
    mcp_server.reset_registry()


# ---------------------------------------------------------------- wire helpers


def _run(fn: Callable[[ClientSession], Awaitable[object]]) -> object:
    """Open a fresh server + in-memory client session and run ``fn(client)`` inside it."""

    async def main():
        server = mcp_server.build_server()
        async with create_connected_server_and_client_session(server) as client:
            return await fn(client)

    return asyncio.run(main())


def _parse(result: CallToolResult) -> tuple[dict | None, str | None]:
    """``(payload, None)`` on success, ``(None, error_text)`` on an ``isError`` result.

    Both shapes are read off the *protocol* object a real client receives, so a swallowed error
    (isError False with an error-looking payload) would fail these tests rather than pass them.
    """
    text = "".join(getattr(c, "text", "") for c in result.content)
    if result.isError:
        return None, text
    if result.structuredContent:
        return dict(result.structuredContent), None
    return json.loads(text), None


async def _call(client: ClientSession, name: str, args: dict) -> tuple[dict | None, str | None]:
    return _parse(await client.call_tool(name, args))


# ---------------------------------------------------------------- discovery


def test_wire_list_tools_matches_tools_and_strict_schemas():
    async def go(client):
        return (await client.list_tools()).tools

    listed = _run(go)
    assert [t.name for t in listed] == mcp_server.TOOLS
    by_name = {s["name"]: s for s in mcp_server.TOOL_SCHEMAS}
    for tool in listed:
        assert tool.inputSchema == by_name[tool.name]["input_schema"]
        assert tool.inputSchema["additionalProperties"] is False
        assert tool.description == by_name[tool.name]["description"]


def test_tool_schemas_are_a_byte_stable_prompt_prefix():
    """AR-6: the NL interface caches this prefix; any byte that moves is a cache miss."""
    once = json.dumps(mcp_server.TOOL_SCHEMAS)  # insertion order, no sort_keys on purpose
    assert json.dumps(mcp_server._build_tool_schemas()) == once

    async def go(client):
        return [t.model_dump() for t in (await client.list_tools()).tools]

    assert _run(go) == _run(go)  # two servers advertise identical bytes

    # the literals a model would rely on are the whitelist's, introspected not hard-coded
    params = mcp_server.TOOL_SCHEMAS[1]["input_schema"]["properties"]["params"]["properties"]
    for strategy, specs in PARAM_WHITELIST.items():
        for name, spec in specs.items():
            if "choices" in spec:
                assert set(spec["choices"]) <= set(params[name]["enum"])
            else:
                assert params[name]["minimum"] <= spec["min"]
                assert params[name]["maximum"] >= spec["max"]
                assert f"{spec['min']}..{spec['max']}" in params[name]["description"]
                assert strategy in params[name]["description"]
    assert mcp_server.TOOL_SCHEMAS[0]["input_schema"]["properties"]["tickers"]["items"][
        "enum"
    ] == list(loader.UNIVERSE)


# ---------------------------------------------------------------- round trip + rigor


def test_wire_round_trip_equals_direct_call_and_costs_are_charged():
    async def go(client):
        loaded, err = await _call(
            client, "load_data", {"start": "2015-01-01", "end": "2016-12-31", "tickers": TICKERS}
        )
        assert err is None, err
        ds = loaded["dataset_id"]
        assert loaded["tickers"] == sorted(TICKERS)
        assert loaded["n_days"] == len(pd.bdate_range("2015-01-01", "2016-12-31"))

        params = {"lookback": 60, "top_n": 1}  # top_n=1 on 3 names -> real turnover
        ran, err = await _call(
            client, "run_backtest", {"dataset_id": ds, "strategy": "momentum", "params": params}
        )
        assert err is None, err
        got, err = await _call(client, "get_metrics", {"result_id": ran["result_id"]})
        assert err is None, err
        assert got["metrics"] == ran["metrics"]
        # default cost on the wire == explicit 10 bps through the plain function
        direct = mcp_server.run_backtest(ds, "momentum", params, cost_bps=10.0)
        assert got["metrics"] == direct["metrics"]
        assert all(math.isfinite(v) for v in got["metrics"].values())

        totals = {}
        for bps in (0, 10, 100):
            r, err = await _call(
                client,
                "run_backtest",
                {"dataset_id": ds, "strategy": "momentum", "params": params, "cost_bps": bps},
            )
            assert err is None, err
            totals[bps] = r["metrics"]["total_return"]
        return totals

    totals = _run(go)
    # costs are charged on turnover through the transport: more bps, strictly less return
    assert totals[0] > totals[10] > totals[100]


def test_wire_optimize_over_two_result_handles():
    async def go(client):
        ds = (
            await _call(
                client,
                "load_data",
                {"start": "2015-01-01", "end": "2018-12-31", "tickers": TICKERS},
            )
        )[0]["dataset_id"]
        rids = []
        for strategy in ("momentum", "mean_reversion"):
            r, err = await _call(client, "run_backtest", {"dataset_id": ds, "strategy": strategy})
            assert err is None, err
            rids.append(r["result_id"])
        opt, err = await _call(client, "optimize_portfolio", {"result_ids": rids})
        assert err is None, err
        assert set(opt["weights"]) == set(rids)
        assert math.isclose(sum(opt["weights"].values()), 1.0, abs_tol=1e-6)
        assert all(w >= -1e-9 for w in opt["weights"].values())
        assert len(opt["frontier"]) == 20 and set(opt["frontier"][0]) == {"risk", "ret"}

        _, both = await _call(client, "optimize_portfolio", {"result_ids": rids, "dataset_id": ds})
        _, neither = await _call(client, "optimize_portfolio", {})
        return both, neither

    both, neither = _run(go)
    assert both and "exactly one" in both
    assert neither and "exactly one" in neither


# ---------------------------------------------------------------- adversarial: the wire gate


def test_wire_strict_schema_is_enforced_before_any_function_runs():
    """FastMCP's own default is validate_input=False (pydantic lax coercion: extra keys dropped,
    ``true`` -> 1.0). ``build_server`` turns the jsonschema gate on; prove it is the gate that
    fires — the error text is the low-level server's, and the registries stay empty."""

    async def go(client):
        ds = (
            await _call(
                client,
                "load_data",
                {"start": "2015-01-01", "end": "2016-12-31", "tickers": TICKERS},
            )
        )[0]["dataset_id"]
        bad_calls = [
            ("load_data", {"start": "2015-01-01", "end": "2016-12-31", "path": "/etc/passwd"}),
            ("load_data", {"start": "2015-1-1", "end": "2016-12-31"}),
            ("load_data", {"start": "2015-01-01", "end": "2016-12-31", "tickers": []}),
            ("load_data", {"start": "2015-01-01", "end": "2016-12-31", "tickers": ["EVIL"]}),
            ("run_backtest", {"dataset_id": ds, "strategy": "momentum", "cost_bps": True}),
            ("run_backtest", {"dataset_id": ds, "strategy": "momentum", "cost_bps": "10"}),
            ("run_backtest", {"dataset_id": ds, "strategy": "momentum", "cost_bps": 100.5}),
            ("run_backtest", {"dataset_id": ds, "strategy": "momentum", "source": "print(1)"}),
            ("run_backtest", {"dataset_id": ds, "strategy": "momentum", "engine": "shell"}),
            ("run_backtest", {"dataset_id": ds, "strategy": "__import__('os')"}),
            (
                "run_backtest",
                {"dataset_id": ds, "strategy": "momentum", "params": {"code": "import os"}},
            ),
            ("run_backtest", {"strategy": "momentum"}),
            ("optimize_portfolio", {"dataset_id": ds, "objective": "max_leverage"}),
            ("optimize_portfolio", {"result_ids": ["res_a"]}),
            ("get_metrics", {}),
        ]
        errors = []
        for name, args in bad_calls:
            payload, err = await _call(client, name, args)
            assert payload is None, (name, args, payload)
            errors.append((name, args, err))
        return errors

    for name, args, err in _run(go):
        assert err is not None and "Input validation error" in err, (name, args, err)
    assert mcp_server._RESULTS == {}
    assert len(mcp_server._DATASETS) == 1  # only the one good load registered


def test_schema_valid_but_whitelist_invalid_param_is_refused_by_the_function():
    """The union ``params`` schema admits momentum lookback=10 (mean-reversion's floor is 5);
    the second gate, ``validate_params``, must still refuse it — with its own message, so the
    two gates are visibly distinct and the loose union is not a hole."""

    async def go(client):
        ds = (
            await _call(
                client,
                "load_data",
                {"start": "2015-01-01", "end": "2016-12-31", "tickers": TICKERS},
            )
        )[0]["dataset_id"]
        return await _call(
            client,
            "run_backtest",
            {"dataset_id": ds, "strategy": "momentum", "params": {"lookback": 10}},
        )

    payload, err = _run(go)
    assert payload is None
    assert "Input validation error" not in err  # passed jsonschema ...
    assert "lookback" in err and "[20, 252]" in err  # ... refused by the whitelist
    assert mcp_server._RESULTS == {}


def test_wire_holdout_boundary_is_exact_even_though_data_exists():
    bounds = loader.get_split_bounds()
    lo, hi = bounds["train"][0], bounds["validation"][1]
    hi_plus = (pd.Timestamp(hi) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    lo_minus = (pd.Timestamp(lo) - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    # the refusal must be policy, not absence of rows: the synthetic panel covers both dates
    dates = set(_SYNTHETIC["date"].dt.strftime("%Y-%m-%d"))
    assert hi_plus in dates and lo_minus in dates

    async def go(client):
        ok, err = await _call(client, "load_data", {"start": lo, "end": hi, "tickers": TICKERS})
        assert err is None, err
        _, late = await _call(client, "load_data", {"start": "2022-06-01", "end": hi_plus})
        _, early = await _call(client, "load_data", {"start": lo_minus, "end": "2010-06-30"})
        _, holdout = await _call(client, "load_data", {"start": "2023-01-01", "end": "2023-12-31"})
        return ok, late, early, holdout

    ok, late, early, holdout = _run(go)
    assert ok["start"] == lo and ok["end"] == hi
    for err in (late, holdout):
        assert err is not None and "holdout" in err and hi in err
    assert early is not None and lo in early
    assert len(mcp_server._DATASETS) == 1  # the rejected windows registered nothing


def test_wire_unknown_tool_and_unknown_handle_are_errors_not_silence():
    async def go(client):
        _, unknown_tool = await _call(client, "load_prices", {"start": "2015-01-01"})
        _, unknown_res = await _call(client, "get_metrics", {"result_id": "res_0000000000000000"})
        _, unknown_ds = await _call(
            client, "run_backtest", {"dataset_id": "ds_0000000000000000", "strategy": "momentum"}
        )
        return unknown_tool, unknown_res, unknown_ds

    unknown_tool, unknown_res, unknown_ds = _run(go)
    assert unknown_tool and "load_prices" in unknown_tool
    assert unknown_res and "result_id" in unknown_res and "run_backtest" in unknown_res
    assert unknown_ds and "dataset_id" in unknown_ds and "load_data" in unknown_ds
