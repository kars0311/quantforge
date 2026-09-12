"""Tests for the MCP schema/discovery layer of ``quantforge.ai.mcp_server`` (component 11).

The milestone-3 suite (``test_mcp_tools.py``) proves the four *functions*; this file proves the
layer on top: ``TOOL_SCHEMAS`` (what a model reads) and ``build_server`` (what an MCP client
discovers and calls). Fully offline — the same synthetic price source pattern as the tool tests,
extended into the holdout years so the MCP-level rejection is a refusal, not a missing-data
accident. Nothing here opens stdio; the ``__main__`` block is only inspected as source.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re

import jsonschema
import numpy as np
import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from quantforge import interchange
from quantforge.ai import mcp_server
from quantforge.data import loader
from quantforge.strategies import PARAM_WHITELIST, STRATEGIES

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM"]
SYN_START, SYN_END = "2010-01-01", "2026-06-30"
HANDLE_RE = re.compile(r"^(ds|res)_[0-9a-f]{16}$")

#: Required fields per tool, exactly as the spec states them (None = nothing required).
REQUIRED = {
    "load_data": ["start", "end"],
    "run_backtest": ["dataset_id", "strategy"],
    "optimize_portfolio": None,
    "get_metrics": ["result_id"],
}


def _synthetic_long(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(SYN_START, SYN_END, tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), len(TICKERS)))
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


@pytest.fixture
def server() -> FastMCP:
    return mcp_server.build_server()


def _payload(out) -> dict:
    """Parse whatever shape ``FastMCP.call_tool`` returns into the tool's dict.

    In-process the SDK hands back a list of ``TextContent`` holding the JSON text; other SDK
    versions return the structured dict, or a ``(content, structured)`` tuple. Accepting all
    three keeps the test about the tools, not about one SDK release's return type.
    """
    if isinstance(out, tuple):
        return out[1]
    if isinstance(out, dict):
        return out
    return json.loads(out[0].text)


def _call(server: FastMCP, name: str, args: dict) -> tuple[dict | None, str | None]:
    """``(payload, None)`` on success, ``(None, error_text)`` when the call is refused.

    FastMCP raises ``ToolError`` in-process and returns an ``isError`` result on the wire; either
    way the point under test is that the failure is *surfaced with its message*, not swallowed.
    """
    try:
        out = asyncio.run(server.call_tool(name, args))
    except ToolError as exc:
        return None, str(exc)
    if getattr(out, "isError", False):
        return None, "".join(getattr(c, "text", "") for c in out.content)
    return _payload(out), None


def _listed(server: FastMCP) -> list:
    return asyncio.run(server.list_tools())


# ---------------------------------------------------------------- build_server


def test_list_tools_names_equal_tools_in_order(server):
    assert [t.name for t in _listed(server)] == mcp_server.TOOLS


def test_build_server_returns_fresh_instances():
    a, b = mcp_server.build_server(), mcp_server.build_server()
    assert isinstance(a, FastMCP) and isinstance(b, FastMCP)
    assert a is not b
    assert a.name == "quantforge"
    assert "read-only" in a.instructions and "holdout" in a.instructions


def test_listed_schemas_are_the_strict_tool_schemas(server):
    by_name = {t["name"]: t for t in mcp_server.TOOL_SCHEMAS}
    for tool in _listed(server):
        schema = tool.inputSchema
        assert schema["additionalProperties"] is False
        assert schema.get("required") == REQUIRED[tool.name]
        assert schema == by_name[tool.name]["input_schema"]
        assert tool.description == by_name[tool.name]["description"]


def test_pydantic_arg_models_match_schema_properties(server):
    """Drift guard: the SDK validates calls with a signature-derived model, the client sees our
    hand-built schema. If a function parameter is renamed without the schema (or vice versa),
    a schema-valid call would fail argument validation — this pins the two field sets together."""
    for schema in mcp_server.TOOL_SCHEMAS:
        tool = server._tool_manager.get_tool(schema["name"])
        assert set(tool.fn_metadata.arg_model.model_fields) == set(
            schema["input_schema"]["properties"]
        )


def test_main_block_runs_stdio_server():
    src = inspect.getsource(mcp_server)
    assert 'if __name__ == "__main__":' in src
    main_body = src.split('if __name__ == "__main__":', 1)[1]
    assert re.search(r"build_server\(\)\.run\(", main_body)


# ---------------------------------------------------------------- TOOL_SCHEMAS


def test_tool_schemas_shape_and_json_round_trip():
    schemas = mcp_server.TOOL_SCHEMAS
    assert len(schemas) == 4
    assert [s["name"] for s in schemas] == mcp_server.TOOLS
    for s in schemas:
        assert set(s) == {"name", "description", "input_schema"}
        assert s["input_schema"]["type"] == "object"
        assert s["input_schema"]["additionalProperties"] is False
        assert s["input_schema"].get("required") == REQUIRED[s["name"]]
    assert json.loads(json.dumps(schemas)) == schemas
    # deterministic: building twice yields byte-identical prompt prefix material (AR-6)
    assert json.dumps(mcp_server._build_tool_schemas(), sort_keys=True) == json.dumps(
        schemas, sort_keys=True
    )


def test_descriptions_state_holdout_and_read_only():
    for s in mcp_server.TOOL_SCHEMAS:
        assert "Read-only" in s["description"] or "read-only" in s["description"]
    assert "holdout" in mcp_server.TOOL_SCHEMAS[0]["description"]
    assert "holdout" in mcp_server.TOOL_SCHEMAS[1]["description"]


def test_load_data_schema_literals():
    props = mcp_server.TOOL_SCHEMAS[0]["input_schema"]["properties"]
    assert set(props) == {"start", "end", "tickers"}
    for field in ("start", "end"):
        assert props[field] == {**props[field], "type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}
        assert re.match(props[field]["pattern"], "2015-01-01")
        assert not re.match(props[field]["pattern"], "2015-1-1")
    assert props["tickers"]["type"] == "array"
    assert props["tickers"]["items"]["enum"] == list(loader.UNIVERSE)


def test_run_backtest_schema_enums_and_bounds():
    props = mcp_server.TOOL_SCHEMAS[1]["input_schema"]["properties"]
    assert set(props) == {"dataset_id", "strategy", "params", "engine", "cost_bps"}
    assert props["strategy"]["enum"] == sorted(STRATEGIES)
    assert props["engine"]["enum"] == sorted(mcp_server.ENGINES)
    assert props["engine"]["default"] == "python"
    assert props["cost_bps"] == {
        **props["cost_bps"],
        "type": "number",
        "minimum": 0.0,
        "maximum": 100.0,
        "default": 10.0,
    }
    obj = mcp_server.TOOL_SCHEMAS[2]["input_schema"]["properties"]
    assert set(obj) == {"result_ids", "dataset_id", "objective"}
    assert obj["objective"]["enum"] == sorted(mcp_server.OBJECTIVES)
    assert obj["result_ids"] == {**obj["result_ids"], "type": "array", "items": {"type": "string"}}
    assert set(mcp_server.TOOL_SCHEMAS[3]["input_schema"]["properties"]) == {"result_id"}


def test_params_schema_is_introspected_from_whitelist():
    params = mcp_server.TOOL_SCHEMAS[1]["input_schema"]["properties"]["params"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False

    expected: dict[str, list[dict]] = {}
    for strategy_specs in PARAM_WHITELIST.values():
        for name, spec in strategy_specs.items():
            expected.setdefault(name, []).append(spec)
    assert set(params["properties"]) == set(expected)

    for name, specs in expected.items():
        prop = params["properties"][name]
        if "choices" in specs[0]:
            assert prop["type"] == "string"
            assert prop["enum"] == sorted(set().union(*(s["choices"] for s in specs)))
        else:
            assert prop["type"] == {int: "integer", float: "number"}[specs[0]["type"]]
            # union bounds: loosest across strategies, and every per-strategy range fits inside
            assert prop["minimum"] == min(s["min"] for s in specs)
            assert prop["maximum"] == max(s["max"] for s in specs)
            for s in specs:
                assert prop["minimum"] <= s["min"] <= s["max"] <= prop["maximum"]


def test_wire_level_schema_accepts_valid_and_rejects_codegen_shapes():
    """The low-level MCP server validates arguments against ``inputSchema`` with jsonschema
    before a tool runs, so a strict schema is a real gate on the wire — prove it on our schema."""
    schema = mcp_server.TOOL_SCHEMAS[1]["input_schema"]
    jsonschema.validate(
        {"dataset_id": "ds_x", "strategy": "momentum", "params": {"lookback": 60}}, schema
    )
    for bad in (
        {"dataset_id": "ds_x", "strategy": "momentum", "params": {"code": "import os"}},
        {"dataset_id": "ds_x", "strategy": "pairs"},
        {"dataset_id": "ds_x", "strategy": "momentum", "source": "print(1)"},
        {"dataset_id": "ds_x", "strategy": "momentum", "cost_bps": 101},
        {"strategy": "momentum"},
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
    load_schema = mcp_server.TOOL_SCHEMAS[0]["input_schema"]
    jsonschema.validate({"start": "2015-01-01", "end": "2016-12-31"}, load_schema)
    for bad in (
        {"start": "2015-1-1", "end": "2016-12-31"},
        {"start": "2015-01-01", "end": "2016-12-31", "tickers": ["NOPE"]},
        {"start": "2015-01-01", "end": "2016-12-31", "path": "/etc/passwd"},
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, load_schema)


# ---------------------------------------------------------------- MCP-level end to end


def test_mcp_round_trip_matches_direct_function_call(server):
    loaded, err = _call(
        server,
        "load_data",
        {"start": "2015-01-01", "end": "2016-12-31", "tickers": ["AAPL", "MSFT"]},
    )
    assert err is None
    assert set(loaded) == {"dataset_id", "tickers", "start", "end", "n_days"}
    assert HANDLE_RE.match(loaded["dataset_id"])
    assert loaded["tickers"] == ["AAPL", "MSFT"]
    ds = loaded["dataset_id"]

    ran, err = _call(
        server,
        "run_backtest",
        {"dataset_id": ds, "strategy": "momentum", "params": {"lookback": 60}},
    )
    assert err is None
    assert HANDLE_RE.match(ran["result_id"])

    got, err = _call(server, "get_metrics", {"result_id": ran["result_id"]})
    assert err is None
    assert got["metrics"] == ran["metrics"]
    assert got["metrics"] == mcp_server.get_metrics(ran["result_id"])["metrics"]

    # the same run through the plain function on the same handle gives identical metrics
    direct = mcp_server.run_backtest(ds, "momentum", {"lookback": 60})
    assert direct["metrics"] == ran["metrics"]

    opt, err = _call(server, "optimize_portfolio", {"dataset_id": ds})
    assert err is None
    assert set(opt) == {"weights", "frontier"}
    assert set(opt["weights"]) == {"AAPL", "MSFT"}
    assert len(opt["frontier"]) == 20


def test_holdout_request_is_surfaced_as_mcp_error(server):
    payload, err = _call(server, "load_data", {"start": "2023-01-01", "end": "2023-12-31"})
    assert payload is None
    assert err is not None and "holdout" in err
    assert mcp_server._DATASETS == {}  # nothing registered on rejection


def test_codegen_param_is_surfaced_as_mcp_error(server):
    ds = mcp_server.load_data("2015-01-01", "2016-12-31", ["AAPL", "MSFT"])["dataset_id"]
    payload, err = _call(
        server,
        "run_backtest",
        {"dataset_id": ds, "strategy": "momentum", "params": {"code": "import os"}},
    )
    assert payload is None
    assert err is not None and "code" in err
    assert mcp_server._RESULTS == {}


def test_unknown_tool_is_surfaced_as_mcp_error(server):
    payload, err = _call(server, "not_a_tool", {})
    assert payload is None
    assert err is not None and "not_a_tool" in err
