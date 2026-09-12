"""Tests for ``quantforge.ai.agent`` (component 13), milestones 1 to 4.

Milestone 1 (foundation) pins: the model id and its price entry, the two proposal tools' exact
shapes (and that ``params`` is a deep copy of the MCP ``run_backtest`` schema), the frozen system
prompt (deterministic across reloads, names every vetted strategy / whitelisted param / metric
key / train+validation bound, and contains NO holdout-era date), ``_call``'s metering and cache
breakpoints, the pre-spend validators, and the source-level pins (one ``messages.create`` site,
no engine/portfolio imports, no direct strategy instantiation).

Milestone 2 (the experiment step) pins: ``_run_experiment`` goes through ``mcp_server.run_backtest``
exactly twice (train then validation) with the validated params, its metrics match both the tool
and a direct strategy+engine computation, and a bad proposal executes nothing; ``_select_best``
ranks by VALIDATION Sharpe only, treats NaN as ``-inf``, breaks ties by earliest iter, and returns
copies; ``_tool_result_text`` is small, sorted JSON with rounded metrics and no handles or dates.

Milestone 3 (``_research_loop``) pins: one iteration is one model call so ``max_iters`` bounds
``messages.create`` exactly; the pre-call budget gate, API-error and convergence stops; the exact
transcript shape (assistant content appended as-is, one tool_result per tool_use id, a turn
counter after every turn); rejected / early-done / multi-tool / prose-only / unknown-tool turns;
and that the loop source and conversation never touch the holdout.

Milestone 4 (``run_research``) pins: the exact result contract (JSON only, no handles or ids);
best chosen by VALIDATION Sharpe even when the train winner differs; the holdout scored exactly
once, by the runner, AFTER the last model call, for the best config, on the same ticker universe
the agent iterated on; no holdout call when nothing succeeded or the budget refused; every bad
argument rejected before any spend; the frozen keyword-only signature (no ``public_mode``); an
API error mid-run still scoring the best-so-far; and PUBLIC_MODE running identically to dev mode
for parameter-only proposals. The one live test, ``test_live_smoke``, is skipped unless
``QUANTFORGE_LIVE_AI=1``.

Fully offline: a ``FakeClient`` stands in for ``anthropic.Anthropic`` (records every
``messages.create`` request, returns canned responses with real-looking ``usage``), the MCP tools
read a synthetic price panel, and the budget ledger lives in ``tmp_path``. ``ANTHROPIC_API_KEY``
is never read — the module builds its client lazily and the fixture makes that path explode.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import re
from types import SimpleNamespace

import anthropic
import httpx

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange
from quantforge.ai import agent, budget, guardrails, mcp_server
from quantforge.ai.budget import PRICES_PER_MTOK
from quantforge.data import loader
from quantforge.engine.python_engine import PythonEngine
from quantforge.metrics.performance import _KEYS
from quantforge.strategies import PARAM_WHITELIST, STRATEGIES, validate_params

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM"]
MODEL = agent.MODEL


# ---------------------------------------------------------------- fakes


def _usage(tokens_in: int, tokens_out: int, cache_create: int = 0, cache_read: int = 0):
    return SimpleNamespace(
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cache_creation_input_tokens=cache_create,
        cache_read_input_tokens=cache_read,
    )


def tool_use_response(name: str, data: dict, usage=None, block_id: str = "toolu_01"):
    """A canned agent turn: one ``tool_use`` block."""
    block = SimpleNamespace(type="tool_use", id=block_id, name=name, input=data)
    return SimpleNamespace(
        content=[block], usage=usage or _usage(1000, 200), stop_reason="tool_use"
    )


class FakeClient:
    """Records every ``messages.create`` request and pops canned responses in order."""

    def __init__(self, *responses):
        self.calls: list[dict] = []
        self._queue = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._queue:
            raise AssertionError("FakeClient received more requests than canned responses")
        nxt = self._queue.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


# ---------------------------------------------------------------- fixtures


def _synthetic_long(seed: int = 0) -> pd.DataFrame:
    """Long interchange 'prices' frame for TICKERS on business days (extends into the holdout)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2026-06-30", tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), len(TICKERS)))
    wide = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=TICKERS)
    return interchange.to_long(wide, "prices")


_SYNTHETIC = _synthetic_long()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Synthetic prices, temp ledger, dev-mode env, no real client, no API key."""
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _SYNTHETIC)
    mcp_server.reset_registry()
    ledger = tmp_path / "cache" / "ai_ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(ledger))
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    monkeypatch.delenv("AI_DISABLED", raising=False)
    monkeypatch.delenv("AI_BUDGET_USD_DAILY", raising=False)
    monkeypatch.delenv("AI_BUDGET_USD_TOTAL", raising=False)
    monkeypatch.delenv("AI_RATE_LIMIT_PER_HOUR", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Any accidental path to a real client must blow up loudly, not silently dial out.
    monkeypatch.setattr(agent, "_client", None)
    monkeypatch.setattr(
        agent, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("real client"))
    )
    yield ledger
    mcp_server.reset_registry()


def today_bucket(ledger) -> dict | None:
    if not ledger.exists():
        return None
    return json.loads(ledger.read_text())["days"].get(budget._utc_today())


# ---------------------------------------------------------------- (1) model + constants


def test_model_is_sonnet_and_priced():
    assert agent.MODEL == "claude-sonnet-5"
    assert agent.MODEL in PRICES_PER_MTOK


def test_constants():
    assert agent._MAX_TOKENS == 1024
    assert agent._EST_TOKENS_IN_BASE == 4000
    assert agent._EST_TOKENS_IN_PER_TURN == 1200
    assert agent._EST_TOKENS_OUT == agent._MAX_TOKENS
    assert agent._MAX_GOAL_CHARS == 2000
    assert agent._DEFAULT_COST_BPS == 10.0
    bounds = loader.get_split_bounds()
    assert (agent._WINDOW_START, agent._WINDOW_END) == (
        bounds["train"][0],
        bounds["validation"][1],
    )


# ---------------------------------------------------------------- (2) tool shapes


def _run_backtest_params():
    schema = mcp_server.TOOL_SCHEMAS[1]
    assert schema["name"] == "run_backtest"
    return schema["input_schema"]["properties"]["params"]


def test_propose_tool_shape():
    tool = agent.PROPOSE_TOOL
    assert tool["name"] == "propose_experiment"
    assert set(tool) == {"name", "description", "input_schema"}
    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert list(schema["properties"]) == ["strategy", "params", "rationale"]
    assert schema["required"] == ["strategy", "params", "rationale"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["strategy"]["type"] == "string"
    assert schema["properties"]["strategy"]["enum"] == sorted(STRATEGIES)
    assert schema["properties"]["rationale"]["type"] == "string"


def test_propose_params_is_a_deepcopy_of_run_backtest_params():
    ours = agent.PROPOSE_TOOL["input_schema"]["properties"]["params"]
    theirs = _run_backtest_params()
    assert ours == theirs
    assert ours is not theirs
    assert ours["properties"] is not theirs["properties"]
    assert agent._run_backtest_params_schema() == theirs
    assert agent._run_backtest_params_schema() is not theirs


def test_done_tool_shape():
    tool = agent.DONE_TOOL
    assert tool["name"] == "declare_done"
    assert set(tool) == {"name", "description", "input_schema"}
    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert list(schema["properties"]) == ["reason"]
    assert schema["properties"]["reason"]["type"] == "string"
    assert schema["required"] == ["reason"]
    assert schema["additionalProperties"] is False


def test_tools_list_is_serialisable_and_not_strict():
    assert agent.TOOLS == [agent.PROPOSE_TOOL, agent.DONE_TOOL]
    assert agent.TOOLS[0] is agent.PROPOSE_TOOL and agent.TOOLS[1] is agent.DONE_TOOL
    text = json.dumps(agent.TOOLS)
    assert "strict" not in text
    for tool in agent.TOOLS:
        assert "strict" not in tool
        assert "cache_control" not in tool
        json.dumps(tool)


# ---------------------------------------------------------------- (3) frozen prompt


HOLDOUT_DATE_RE = re.compile(r"20(2[3-9]|[3-9]\d)-\d\d-\d\d")


def test_system_prompt_is_frozen_across_reload():
    before = agent.SYSTEM_RESEARCH
    tools_before = copy.deepcopy(agent.TOOLS)
    reloaded = importlib.reload(agent)
    try:
        assert reloaded.SYSTEM_RESEARCH == before
        assert reloaded.TOOLS == tools_before
    finally:
        importlib.reload(agent)
    assert agent.SYSTEM_RESEARCH == before
    assert agent.TOOLS == tools_before


def test_system_prompt_names_strategies_params_metrics_and_windows():
    text = agent.SYSTEM_RESEARCH
    assert isinstance(text, str) and text
    for strategy in STRATEGIES:
        assert strategy in text
    for strategy, params in PARAM_WHITELIST.items():
        for param, spec in params.items():
            assert param in text, (strategy, param)
            assert str(spec["default"]) in text, (strategy, param)
    for key in _KEYS:
        assert key in text
    bounds = loader.get_split_bounds()
    for date in (*bounds["train"], *bounds["validation"]):
        assert date in text
    assert agent._WINDOW_START in text and agent._WINDOW_END in text


def test_system_prompt_states_the_protocol():
    text = agent.SYSTEM_RESEARCH.lower()
    assert "propose_experiment" in text and "declare_done" in text
    assert "validation" in text and "train" in text
    assert "overfit" in text
    assert "held-out" in text or "holdout" in text
    assert "one tool" in text


def test_prompt_material_contains_no_holdout_date():
    bounds = loader.get_split_bounds()
    for material in (agent.SYSTEM_RESEARCH, json.dumps(agent.TOOLS)):
        assert not HOLDOUT_DATE_RE.search(material), material
        for date in bounds["holdout"]:
            assert date not in material


# ---------------------------------------------------------------- (4) _call metering + caching


def _call_with_fake(messages, tools=None):
    usage = _usage(1000, 150, cache_create=300, cache_read=200)
    resp = tool_use_response("declare_done", {"reason": "done"}, usage=usage)
    client = FakeClient(resp)
    tools = agent.TOOLS if tools is None else tools
    response, usd = agent._call(
        client,
        system=agent.SYSTEM_RESEARCH,
        messages=messages,
        tools=tools,
        tool_choice={"type": "any"},
        max_tokens=agent._MAX_TOKENS,
    )
    return client, response, usd


def test_call_charges_real_usage_with_cache_tokens_at_full_rate(isolated):
    messages = [{"role": "user", "content": "Goal: beat Sharpe 1 on validation."}]
    client, response, usd = _call_with_fake(messages)
    expected = (1500 / 1e6) * 2.0 + (150 / 1e6) * 10.0
    assert usd == pytest.approx(expected)
    assert usd == pytest.approx(budget.estimate(MODEL, 1500, 150))
    bucket = today_bucket(isolated)
    assert bucket is not None
    assert bucket["usd"] == pytest.approx(expected)
    assert bucket["calls"] == 1
    assert bucket["tokens_in"] == 1500 and bucket["tokens_out"] == 150
    assert response.content[0].name == "declare_done"


def test_call_request_shape_and_cache_breakpoints():
    messages = [
        {"role": "user", "content": "Goal: beat Sharpe 1 on validation."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "propose_experiment",
                    "input": {"strategy": "momentum", "params": {}, "rationale": "baseline"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": "{}"},
                {"type": "text", "text": "Continue."},
            ],
        },
    ]
    messages_before = copy.deepcopy(messages)
    tools_before = copy.deepcopy(agent.TOOLS)

    client, _, _ = _call_with_fake(messages)
    assert len(client.calls) == 1
    req = client.calls[0]

    assert req["model"] == "claude-sonnet-5"
    assert req["max_tokens"] == agent._MAX_TOKENS
    assert req["tool_choice"] == {"type": "any"}
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in req

    assert req["system"] == [
        {"type": "text", "text": agent.SYSTEM_RESEARCH, "cache_control": {"type": "ephemeral"}}
    ]

    assert [t["name"] for t in req["tools"]] == ["propose_experiment", "declare_done"]
    assert "cache_control" not in req["tools"][0]
    assert req["tools"][-1]["cache_control"] == {"type": "ephemeral"}

    sent = req["messages"]
    assert len(sent) == 3
    assert sent[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in sent[-1]["content"][0]
    assert "cache_control" not in sent[0]
    assert json.dumps(sent[1]) == json.dumps(messages_before[1])

    # Caller's objects untouched: module constants and the live transcript stay marker-free.
    assert messages == messages_before
    assert agent.TOOLS == tools_before
    assert "cache_control" not in json.dumps(messages)
    assert "cache_control" not in json.dumps(agent.TOOLS)


def test_call_promotes_str_content_to_a_cached_text_block():
    messages = [{"role": "user", "content": "Goal: beat Sharpe 1 on validation."}]
    client, _, _ = _call_with_fake(messages)
    sent = client.calls[0]["messages"]
    assert sent == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Goal: beat Sharpe 1 on validation.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    assert messages == [{"role": "user", "content": "Goal: beat Sharpe 1 on validation."}]


def test_call_refuses_malformed_messages_before_spend(isolated):
    client = FakeClient()
    for bad in ([], [{"role": "user", "content": []}], [{"role": "user", "content": 42}]):
        with pytest.raises(ValueError):
            agent._call(
                client,
                system=agent.SYSTEM_RESEARCH,
                messages=bad,
                tools=agent.TOOLS,
                tool_choice={"type": "any"},
                max_tokens=agent._MAX_TOKENS,
            )
    assert client.calls == []
    assert today_bucket(isolated) is None


def test_call_propagates_sdk_errors_without_charging(isolated):
    client = FakeClient(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        agent._call(
            client,
            system=agent.SYSTEM_RESEARCH,
            messages=[{"role": "user", "content": "go"}],
            tools=agent.TOOLS,
            tool_choice={"type": "any"},
            max_tokens=agent._MAX_TOKENS,
        )
    assert today_bucket(isolated) is None


def test_usage_int_treats_missing_and_none_as_zero():
    assert agent._usage_int(SimpleNamespace(input_tokens=None), "input_tokens") == 0
    assert agent._usage_int(SimpleNamespace(), "cache_read_input_tokens") == 0
    assert agent._usage_int(None, "output_tokens") == 0
    assert agent._usage_int(SimpleNamespace(output_tokens=7), "output_tokens") == 7


def test_find_tool_use_blocks_returns_all_in_order():
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="thinking aloud"),
            SimpleNamespace(type="tool_use", id="a", name="propose_experiment", input={}),
            SimpleNamespace(type="tool_use", id="b", name="declare_done", input={"reason": "x"}),
        ]
    )
    blocks = agent._find_tool_use_blocks(resp)
    assert [b.id for b in blocks] == ["a", "b"]
    assert [b.name for b in blocks] == ["propose_experiment", "declare_done"]
    assert agent._find_tool_use_blocks(SimpleNamespace(content=[])) == []
    assert agent._find_tool_use_blocks(SimpleNamespace()) == []


# ---------------------------------------------------------------- (5) validators


@pytest.mark.parametrize("bad", ["", "   ", None, 42, "x" * 2001])
def test_validate_goal_rejects(bad):
    with pytest.raises(ValueError):
        agent._validate_goal(bad)


def test_validate_goal_accepts_and_returns():
    goal = "find a momentum variant with Sharpe > 1 on validation"
    assert agent._validate_goal(goal) is goal
    assert agent._validate_goal("x" * 2000) == "x" * 2000


@pytest.mark.parametrize("bad", [0, -1, guardrails.MAX_AGENT_ITERS + 1, 2.5, True, "3"])
def test_validate_max_iters_rejects(bad):
    with pytest.raises(ValueError, match=str(guardrails.MAX_AGENT_ITERS)):
        agent._validate_max_iters(bad)


def test_validate_max_iters_accepts_bounds():
    assert guardrails.MAX_AGENT_ITERS == 10
    assert agent._validate_max_iters(1) == 1
    assert agent._validate_max_iters(10) == 10
    assert isinstance(agent._validate_max_iters(5), int)


# ---------------------------------------------------------------- (6) source pins


def test_every_api_call_is_metered_at_one_site():
    src = inspect.getsource(agent)
    assert src.count("messages.create") == 1
    call_src = inspect.getsource(agent._call)
    assert "messages.create" in call_src and "budget.charge" in call_src


def test_agent_reaches_the_pipeline_only_through_mcp_tools():
    src = inspect.getsource(agent)
    assert "quantforge.engine" not in src
    assert "quantforge.portfolio" not in src
    assert "STRATEGIES[" not in src


def test_lazy_client_needs_no_credentials():
    assert agent._client is None
    with pytest.raises(AssertionError, match="real client"):
        agent._get_client()


# ---------------------------------------------------------------- (7) public surface


def test_run_research_signature_is_frozen_and_keyword_only():
    """The stub's ``public_mode`` kwarg is gone: PUBLIC_MODE comes from the environment only."""
    sig = inspect.signature(agent.run_research)
    assert list(sig.parameters) == ["goal", "max_iters", "tickers", "engine", "cost_bps", "client"]
    assert "public_mode" not in sig.parameters
    for name, param in sig.parameters.items():
        if name != "goal":
            assert param.kind is inspect.Parameter.KEYWORD_ONLY, name
    assert sig.parameters["max_iters"].default == guardrails.MAX_AGENT_ITERS == 10
    assert sig.parameters["tickers"].default is None
    assert sig.parameters["engine"].default == "python"
    assert sig.parameters["cost_bps"].default == agent._DEFAULT_COST_BPS
    assert sig.parameters["client"].default is None
    assert "NotImplementedError" not in inspect.getsource(agent.run_research)


# ---------------------------------------------------------------- (8) _run_experiment


ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@pytest.fixture
def windows():
    """``(train_id, val_id)`` handles for the loader's train and validation bounds."""
    bounds = loader.get_split_bounds()
    train_id = mcp_server.load_data(*bounds["train"])["dataset_id"]
    val_id = mcp_server.load_data(*bounds["validation"])["dataset_id"]
    return train_id, val_id


def _direct_metrics(dataset_id: str, strategy: str, params: dict, cost_bps: float) -> dict:
    """The pipeline computed by hand (strategy + engine), bypassing the tool layer entirely."""
    wide = mcp_server._DATASETS[dataset_id]
    validated = validate_params(strategy, params)
    positions = STRATEGIES[strategy]().generate_signals(wide, validated)
    result = PythonEngine().run_backtest(
        wide, positions, {"cost_bps": cost_bps, "strategy": strategy, **validated}
    )
    return {k: float(v) for k, v in result.metrics.items()}


def _assert_metrics_equal(got: dict, expected: dict) -> None:
    assert list(got) == list(expected) == _KEYS
    for key in _KEYS:
        if np.isnan(expected[key]):
            assert np.isnan(got[key]), key
        else:
            assert got[key] == pytest.approx(expected[key], abs=1e-12, rel=0), key


def test_run_experiment_record_shape_and_defaults_merged(windows):
    train_id, val_id = windows
    record = agent._run_experiment(train_id, val_id, "momentum", {"lookback": 60}, "python", 10.0)
    assert list(record) == ["strategy", "params", "train_metrics", "val_metrics"]
    assert record["strategy"] == "momentum"
    assert record["params"] == {"lookback": 60, "top_n": 0}
    assert record["params"] == validate_params("momentum", {"lookback": 60})
    assert list(record["train_metrics"]) == _KEYS
    assert list(record["val_metrics"]) == _KEYS
    for metrics in (record["train_metrics"], record["val_metrics"]):
        assert all(isinstance(v, float) for v in metrics.values())
    assert "result_id" not in record and "dataset_id" not in record


def test_run_experiment_metrics_match_the_tool_and_a_direct_computation(windows):
    train_id, val_id = windows
    params = {"lookback": 60}
    record = agent._run_experiment(train_id, val_id, "momentum", params, "python", 10.0)

    via_tool_train = mcp_server.run_backtest(train_id, "momentum", params, "python", 10.0)
    via_tool_val = mcp_server.run_backtest(val_id, "momentum", params, "python", 10.0)
    _assert_metrics_equal(record["train_metrics"], via_tool_train["metrics"])
    _assert_metrics_equal(record["val_metrics"], via_tool_val["metrics"])

    _assert_metrics_equal(
        record["train_metrics"], _direct_metrics(train_id, "momentum", params, 10.0)
    )
    _assert_metrics_equal(record["val_metrics"], _direct_metrics(val_id, "momentum", params, 10.0))
    # Train and validation are different windows, so the two metric sets must differ.
    assert record["train_metrics"] != record["val_metrics"]


def test_run_experiment_none_params_means_defaults(windows):
    train_id, val_id = windows
    record = agent._run_experiment(train_id, val_id, "momentum", None, "python", 10.0)
    assert record["params"] == validate_params("momentum", {})
    explicit = agent._run_experiment(train_id, val_id, "momentum", {}, "python", 10.0)
    _assert_metrics_equal(record["train_metrics"], explicit["train_metrics"])
    _assert_metrics_equal(record["val_metrics"], explicit["val_metrics"])


def test_run_experiment_validates_before_executing_anything(windows):
    train_id, val_id = windows
    with pytest.raises(ValueError) as expected:
        validate_params("momentum", {"lookback": 999})
    before = dict(mcp_server._RESULTS)
    with pytest.raises(ValueError) as got:
        agent._run_experiment(train_id, val_id, "momentum", {"lookback": 999}, "python", 10.0)
    assert str(got.value) == str(expected.value)
    assert mcp_server._RESULTS == before  # no handle minted for a request that was never valid


@pytest.mark.parametrize(
    "strategy, params, train_key, cost, match",
    [
        ("no_such_strategy", {}, "train", 10.0, "unknown strategy"),
        ("momentum", {}, "bogus", 10.0, "unknown dataset_id"),
        ("momentum", {}, "train", 101, "cost_bps"),
        ("momentum", [1], "train", 10.0, "params must be a dict"),
        ("momentum", "lookback=60", "train", 10.0, "params must be a dict"),
        (None, {}, "train", 10.0, "strategy must be a string"),
    ],
)
def test_run_experiment_rejections_propagate_as_value_error(
    windows, strategy, params, train_key, cost, match
):
    train_id, val_id = windows
    tid = train_id if train_key == "train" else "ds_0000000000000000"
    before = dict(mcp_server._RESULTS)
    with pytest.raises(ValueError, match=match):
        agent._run_experiment(tid, val_id, strategy, params, "python", cost)
    assert mcp_server._RESULTS == before


def test_run_experiment_names_the_type_of_a_non_dict_params(windows):
    train_id, val_id = windows
    with pytest.raises(ValueError, match="list"):
        agent._run_experiment(train_id, val_id, "momentum", [1], "python", 10.0)


def test_run_experiment_unknown_engine_rejected(windows):
    train_id, val_id = windows
    with pytest.raises(ValueError, match="unknown engine"):
        agent._run_experiment(train_id, val_id, "momentum", {}, "matlab", 10.0)
    assert mcp_server._RESULTS == {}


def test_run_experiment_public_mode_codegen_is_a_public_mode_violation(windows, monkeypatch):
    train_id, val_id = windows
    monkeypatch.setenv("PUBLIC_MODE", "on")
    with pytest.raises(guardrails.PublicModeViolation):
        agent._run_experiment(train_id, val_id, "momentum", {"code": "x"}, "python", 10.0)
    assert mcp_server._RESULTS == {}
    # The same proposal outside public mode is still refused — by the whitelist — and still
    # executes nothing: public mode adds a layer, it does not replace one.
    monkeypatch.delenv("PUBLIC_MODE")
    with pytest.raises(ValueError, match="unknown param"):
        agent._run_experiment(train_id, val_id, "momentum", {"code": "x"}, "python", 10.0)
    assert mcp_server._RESULTS == {}


def test_run_experiment_public_mode_allows_parameter_only_proposals(windows, monkeypatch):
    train_id, val_id = windows
    monkeypatch.setenv("PUBLIC_MODE", "on")
    record = agent._run_experiment(train_id, val_id, "momentum", {"lookback": 60}, "python", 10.0)
    assert record["params"] == {"lookback": 60, "top_n": 0}
    assert len(mcp_server._RESULTS) == 2


def _spy_run_backtest(monkeypatch):
    calls: list[tuple] = []
    original = mcp_server.run_backtest

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(mcp_server, "run_backtest", spy)
    return calls


def test_run_experiment_calls_the_tool_twice_train_then_validation(windows, monkeypatch):
    train_id, val_id = windows
    calls = _spy_run_backtest(monkeypatch)
    record = agent._run_experiment(train_id, val_id, "momentum", {"lookback": 60}, "python", 10.0)
    assert len(calls) == 2
    (train_args, train_kw), (val_args, val_kw) = calls
    assert train_kw == {} and val_kw == {}
    assert train_args[0] == train_id and val_args[0] == val_id
    assert train_args[1:] == val_args[1:]
    assert train_args[1:] == ("momentum", record["params"], "python", 10.0)
    # Two result handles were minted, one per window; the record carries neither.
    assert len(mcp_server._RESULTS) == 2
    assert not any(rid in json.dumps(record) for rid in mcp_server._RESULTS)


def test_run_experiment_validation_failure_after_train_leaves_no_partial_record(
    windows, monkeypatch
):
    train_id, val_id = windows
    original = mcp_server.run_backtest
    seen: list[str] = []

    def flaky(dataset_id, *args, **kwargs):
        seen.append(dataset_id)
        if dataset_id == val_id:
            raise ValueError("validation window exploded")
        return original(dataset_id, *args, **kwargs)

    monkeypatch.setattr(mcp_server, "run_backtest", flaky)
    with pytest.raises(ValueError, match="validation window exploded"):
        agent._run_experiment(train_id, val_id, "momentum", {"lookback": 60}, "python", 10.0)
    assert seen == [train_id, val_id]


def test_run_experiment_never_touches_the_pipeline_directly():
    src = inspect.getsource(agent._run_experiment)
    assert "mcp_server.run_backtest" in src
    assert "generate_signals" not in src
    assert "ENGINES" not in src
    assert "_DATASETS" not in src


# ---------------------------------------------------------------- (9) _select_best


def _metrics(sharpe: float) -> dict:
    metrics = {k: 0.1 for k in _KEYS}
    metrics["sharpe"] = sharpe
    return metrics


def _record(
    iter_: int, val_sharpe: float, train_sharpe: float = 0.5, error: str | None = None
) -> dict:
    record = {
        "iter": iter_,
        "strategy": "momentum",
        "params": {"lookback": 20 + iter_, "top_n": 0},
        "train_metrics": None if error else _metrics(train_sharpe),
        "val_metrics": None if error else _metrics(val_sharpe),
        "rationale": f"try {iter_}",
        "error": error,
    }
    assert tuple(record) == agent._RECORD_KEYS
    return record


def _four_records() -> list[dict]:
    return [
        _record(1, 5.0, error="momentum: unknown param(s) ['code']"),
        _record(2, 1.4, train_sharpe=0.2),
        _record(3, 0.9, train_sharpe=9.0),
        _record(4, float("nan"), train_sharpe=3.0),
    ]


def test_select_best_ranks_by_validation_sharpe_only():
    history = _four_records()
    best = agent._select_best(history)
    assert best is not None
    assert list(best) == ["strategy", "params", "train_metrics", "val_metrics"]
    assert best["params"] == {"lookback": 22, "top_n": 0}
    assert best["val_metrics"]["sharpe"] == 1.4
    assert best["train_metrics"]["sharpe"] == 0.2  # the train winner (iter 3, 9.0) lost


def test_select_best_ties_go_to_the_earliest_iter():
    history = _four_records()
    history[3]["val_metrics"]["sharpe"] = 1.4
    assert agent._select_best(history)["params"]["lookback"] == 22
    # Order in the list does not matter, iter does.
    assert agent._select_best(list(reversed(history)))["params"]["lookback"] == 22


def test_select_best_skips_errors_and_treats_non_finite_as_worst():
    assert agent._select_best([]) is None
    assert agent._select_best([_record(1, 5.0, error="boom"), _record(2, 9.0, error="x")]) is None
    inf_first = [_record(1, float("inf")), _record(2, float("-inf")), _record(3, 0.1)]
    assert agent._select_best(inf_first)["params"]["lookback"] == 23
    only_nan = [_record(1, float("nan"))]
    assert agent._select_best(only_nan)["params"]["lookback"] == 21


def test_select_best_returns_copies_not_aliases():
    history = _four_records()
    best = agent._select_best(history)
    best["params"]["lookback"] = 999
    best["val_metrics"]["sharpe"] = 99.0
    assert history[1]["params"] == {"lookback": 22, "top_n": 0}
    assert history[1]["val_metrics"]["sharpe"] == 1.4
    assert best["params"] is not history[1]["params"]
    assert best["train_metrics"] is not history[1]["train_metrics"]


def test_record_keys_are_the_documented_six_plus_error():
    assert agent._RECORD_KEYS == (
        "iter",
        "strategy",
        "params",
        "train_metrics",
        "val_metrics",
        "rationale",
        "error",
    )
    assert agent._CONFIG_KEYS == ("strategy", "params", "train_metrics", "val_metrics")


# ---------------------------------------------------------------- (10) _tool_result_text


def test_tool_result_text_is_small_sorted_json_with_rounded_metrics(windows):
    train_id, val_id = windows
    record = agent._run_experiment(train_id, val_id, "momentum", {"lookback": 60}, "python", 10.0)
    text = agent._tool_result_text(record)
    payload = json.loads(text)
    assert list(payload) == ["params", "strategy", "train_metrics", "val_metrics"]
    assert payload["strategy"] == "momentum"
    assert payload["params"] == {"lookback": 60, "top_n": 0}
    for which in ("train_metrics", "val_metrics"):
        assert list(payload[which]) == sorted(_KEYS)  # sort_keys sorts nested dicts too
        for key, value in payload[which].items():
            assert value is None or value == round(value, 4), (which, key, value)
            if value is not None:
                assert value == pytest.approx(record[which][key], abs=5e-5)
    assert "res_" not in text and "ds_" not in text
    assert not ISO_DATE_RE.search(text)
    assert not any(rid in text for rid in mcp_server._RESULTS)
    assert train_id not in text and val_id not in text
    assert text == json.dumps(payload, sort_keys=True)  # deterministic bytes


def test_tool_result_text_maps_non_finite_to_null_and_is_valid_json():
    record = _record(1, float("nan"))
    record["train_metrics"]["max_drawdown"] = float("inf")
    text = agent._tool_result_text(record)
    assert "NaN" not in text and "Infinity" not in text
    payload = json.loads(text)
    assert payload["val_metrics"]["sharpe"] is None
    assert payload["train_metrics"]["max_drawdown"] is None
    assert payload["train_metrics"]["sharpe"] == 0.5
    assert "iter" not in payload and "rationale" not in payload and "error" not in payload


def test_tool_result_text_rounds_and_does_not_mutate_the_record():
    record = _record(1, 1.23456789)
    before = copy.deepcopy(record)
    payload = json.loads(agent._tool_result_text(record))
    assert payload["val_metrics"]["sharpe"] == 1.2346
    assert record == before


# ---------------------------------------------------------------- (11) _research_loop


GOAL = "find a momentum variant with Sharpe > 1 on validation"
TOOL_CHOICE = {"type": "any", "disable_parallel_tool_use": True}


def propose_response(strategy, params, rationale="try it", usage=None, tool_id="toolu_1"):
    """A canned turn calling ``propose_experiment``."""
    data = {"strategy": strategy, "params": params, "rationale": rationale}
    return tool_use_response("propose_experiment", data, usage=usage, block_id=tool_id)


def done_response(reason="converged", usage=None, tool_id="toolu_done"):
    """A canned turn calling ``declare_done``."""
    return tool_use_response("declare_done", {"reason": reason}, usage=usage, block_id=tool_id)


def text_only_response(text, usage=None):
    """A canned turn with prose and no tool call (a protocol violation the loop must survive)."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=usage or _usage(1000, 200),
        stop_reason="end_turn",
    )


def multi_tool_response(*calls, usage=None):
    """A canned turn with several ``tool_use`` blocks: ``calls`` are ``(name, input, id)``."""
    blocks = [SimpleNamespace(type="tool_use", id=i, name=n, input=d) for n, d, i in calls]
    return SimpleNamespace(content=blocks, usage=usage or _usage(1000, 200), stop_reason="tool_use")


def api_connection_error():
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))


def run_loop(client, windows, max_iters=5, **overrides):
    train_id, val_id = windows
    kwargs = dict(
        train_id=train_id,
        val_id=val_id,
        engine="python",
        cost_bps=10.0,
        max_iters=max_iters,
        client=client,
    )
    kwargs.update(overrides)
    return agent._research_loop(GOAL, **kwargs)


def usd_of(usage) -> float:
    """What ``_call`` charges for one fake usage: every input flavour at the full input rate."""
    tokens_in = (
        usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
    )
    return budget.estimate(MODEL, tokens_in, usage.output_tokens)


def user_text(message) -> str:
    """The text of a user message, whether it is a str or a block list."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return " ".join(b["text"] for b in content if b.get("type") == "text")


def tool_results(message) -> list[dict]:
    content = message["content"]
    assert isinstance(content, list)
    return [b for b in content if b.get("type") == "tool_result"]


def test_stop_reasons_are_the_documented_three_plus_api_error():
    assert agent.STOP_REASONS == ("converged", "max_iters", "budget", "api_error")


def test_golden_run_converges_with_exact_metering(windows, isolated):
    usages = [
        _usage(1000, 200, cache_create=3000, cache_read=0),
        _usage(400, 180, cache_create=500, cache_read=3000),
        _usage(300, 60, cache_create=400, cache_read=3500),
    ]
    r1 = propose_response(
        "momentum", {"lookback": 60}, "baseline", usage=usages[0], tool_id="toolu_1"
    )
    r2 = propose_response(
        "momentum", {"lookback": 120}, "slower", usage=usages[1], tool_id="toolu_2"
    )
    r3 = done_response("no further gains", usage=usages[2])
    client = FakeClient(r1, r2, r3)

    out = run_loop(client, windows, max_iters=5)

    assert set(out) == {"history", "stopped_because", "spend_usd", "error", "n_model_calls"}
    assert out["stopped_because"] == "converged"
    assert out["error"] is None
    assert out["n_model_calls"] == 3
    assert len(client.calls) == 3
    assert client._queue == []

    history = out["history"]
    assert [r["iter"] for r in history] == [1, 2]
    assert all(r["error"] is None for r in history)
    assert all(tuple(r) == agent._RECORD_KEYS for r in history)
    assert history[0]["params"] == {"lookback": 60, "top_n": 0}
    assert history[1]["params"] == {"lookback": 120, "top_n": 0}
    assert [r["rationale"] for r in history] == ["baseline", "slower"]
    train_id, val_id = windows
    for record in history:
        params = {"lookback": record["params"]["lookback"]}
        _assert_metrics_equal(
            record["train_metrics"],
            mcp_server.run_backtest(train_id, "momentum", params, "python", 10.0)["metrics"],
        )
        _assert_metrics_equal(
            record["val_metrics"],
            mcp_server.run_backtest(val_id, "momentum", params, "python", 10.0)["metrics"],
        )

    expected_spend = sum(usd_of(u) for u in usages)
    assert expected_spend == pytest.approx(
        (4000 + 3900 + 4200) / 1e6 * 2.0 + (200 + 180 + 60) / 1e6 * 10.0
    )
    assert out["spend_usd"] == pytest.approx(expected_spend)
    bucket = today_bucket(isolated)
    assert bucket["usd"] == pytest.approx(expected_spend)
    assert bucket["calls"] == 3
    assert bucket["tokens_in"] == 4000 + 3900 + 4200
    assert bucket["tokens_out"] == 200 + 180 + 60


def test_golden_run_transcript_shape(windows):
    r1 = propose_response("momentum", {"lookback": 60}, "baseline", tool_id="toolu_1")
    r2 = propose_response("momentum", {"lookback": 120}, "slower", tool_id="toolu_2")
    client = FakeClient(r1, r2, done_response())
    out = run_loop(client, windows, max_iters=5)
    record = out["history"][0]

    first = client.calls[0]
    assert len(first["messages"]) == 1
    assert first["messages"][0]["role"] == "user"
    goal_text = user_text(first["messages"][0])
    assert GOAL in goal_text and "turn 1" in goal_text and "5 turns" in goal_text

    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert user_text(second[0]) == goal_text
    assert second[1]["content"] is r1.content  # identity: thinking blocks round-trip as-is
    results = tool_results(second[2])
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "toolu_1"
    assert results[0].get("is_error", False) is False
    assert results[0]["content"] == agent._tool_result_text(record)
    text = user_text(second[2])
    assert "Turn 1 of 5 used; 4 remaining" in text
    assert "Best validation Sharpe so far" in text
    assert f"{record['val_metrics']['sharpe']:.4f}" in text and "(turn 1)" in text
    assert second[2]["content"][-1]["type"] == "text"

    third = client.calls[2]["messages"]
    assert [m["role"] for m in third] == ["user", "assistant", "user", "assistant", "user"]
    assert third[3]["content"] is r2.content
    assert "Turn 2 of 5 used; 3 remaining" in user_text(third[4])

    for req in client.calls:
        assert req["tool_choice"] == TOOL_CHOICE
        assert req["model"] == MODEL and req["max_tokens"] == agent._MAX_TOKENS
        sent_tools = copy.deepcopy(req["tools"])
        assert sent_tools[-1].pop("cache_control") == {"type": "ephemeral"}
        assert sent_tools == agent.TOOLS
        assert req["system"][0]["text"] == agent.SYSTEM_RESEARCH


def test_max_iters_bounds_model_calls_exactly(windows):
    responses = [
        propose_response("momentum", {"lookback": 20 * k}, tool_id=f"toolu_{k}")
        for k in (1, 2, 3, 4)
    ]
    client = FakeClient(*responses)
    out = run_loop(client, windows, max_iters=3)
    assert out["stopped_because"] == "max_iters"
    assert out["n_model_calls"] == 3 and len(client.calls) == 3
    assert [r["iter"] for r in out["history"]] == [1, 2, 3]
    assert len(client._queue) == 1  # the 4th canned response was never consumed
    assert out["error"] is None
    # The 3rd request carries turn 2's closing counter; turn 3's own closing message is appended
    # after the last call (harmless) and never sent, so it is not observable here.
    assert "Turn 2 of 3 used; 1 remaining" in user_text(client.calls[-1]["messages"][-1])


def test_budget_stop_mid_loop(windows, monkeypatch, isolated):
    usage = _usage(1000, 200)
    per_call = usd_of(usage)
    est = [budget.estimate(MODEL, 4000 + 1200 * (t - 1), 1024) for t in (1, 2, 3)]
    # allow passes on turn 2 iff spent(1) + est2 <= cap and fails on turn 3 iff
    # spent(1) + spent(2) + est3 > cap; pick the midpoint of that open interval.
    lo, hi = per_call + est[1], 2 * per_call + est[2]
    assert lo < hi
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", repr((lo + hi) / 2))
    client = FakeClient(
        propose_response("momentum", {"lookback": 60}, usage=usage, tool_id="toolu_1"),
        propose_response("momentum", {"lookback": 120}, usage=usage, tool_id="toolu_2"),
        propose_response("momentum", {"lookback": 90}, usage=usage, tool_id="toolu_3"),
    )
    out = run_loop(client, windows, max_iters=5)
    assert out["stopped_because"] == "budget"
    assert out["n_model_calls"] == 2 and len(client.calls) == 2
    assert [r["iter"] for r in out["history"]] == [1, 2]
    assert out["spend_usd"] == pytest.approx(2 * per_call)
    assert today_bucket(isolated)["usd"] == pytest.approx(2 * per_call)
    assert out["error"] is None


@pytest.mark.parametrize("env", [("AI_BUDGET_USD_DAILY", "0"), ("AI_DISABLED", "on")])
def test_budget_denied_before_the_first_call(windows, monkeypatch, isolated, env):
    monkeypatch.setenv(*env)
    client = FakeClient(propose_response("momentum", {"lookback": 60}))
    out = run_loop(client, windows, max_iters=5)
    assert out == {
        "history": [],
        "stopped_because": "budget",
        "spend_usd": 0.0,
        "error": None,
        "n_model_calls": 0,
    }
    assert client.calls == []
    assert today_bucket(isolated) is None
    assert mcp_server._RESULTS == {}


def test_api_error_stops_the_loop_and_keeps_history(windows, isolated, caplog):
    usage = _usage(1000, 200, cache_create=100)
    err = api_connection_error()
    client = FakeClient(
        propose_response("momentum", {"lookback": 60}, usage=usage, tool_id="toolu_1"),
        err,
        propose_response("momentum", {"lookback": 120}, tool_id="toolu_3"),
    )
    with caplog.at_level("WARNING", logger="quantforge.ai.agent"):
        out = run_loop(client, windows, max_iters=5)
    assert out["stopped_because"] == "api_error"
    assert out["error"] == str(err)
    assert len(out["history"]) == 1 and out["history"][0]["error"] is None
    assert out["n_model_calls"] == 2 and len(client.calls) == 2
    assert out["spend_usd"] == pytest.approx(usd_of(usage))
    assert today_bucket(isolated)["calls"] == 1
    assert any("API error" in rec.message for rec in caplog.records)
    assert len(client._queue) == 1


def test_non_api_exception_propagates(windows):
    client = FakeClient(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        run_loop(client, windows, max_iters=2)


def test_rejected_proposal_is_recorded_verbatim_and_the_loop_continues(windows):
    with pytest.raises(ValueError) as expected:
        validate_params("momentum", {"lookback": 999})
    client = FakeClient(
        propose_response("momentum", {"lookback": 999}, "too long", tool_id="toolu_bad"),
        propose_response("momentum", {"lookback": 60}, "fixed", tool_id="toolu_ok"),
        done_response(),
    )
    before = dict(mcp_server._RESULTS)
    out = run_loop(client, windows, max_iters=5)
    assert out["stopped_because"] == "converged"
    bad, good = out["history"]
    assert bad["iter"] == 1 and bad["error"] == str(expected.value)
    assert bad["train_metrics"] is None and bad["val_metrics"] is None
    assert bad["params"] == {"lookback": 999}  # raw, not validated
    assert bad["strategy"] == "momentum" and bad["rationale"] == "too long"
    assert good["iter"] == 2 and good["error"] is None
    assert good["params"] == {"lookback": 60, "top_n": 0}

    second = client.calls[1]["messages"][-1]
    results = tool_results(second)
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "toolu_bad"
    assert results[0]["is_error"] is True
    assert results[0]["content"] == str(expected.value)
    assert "No successful experiment yet" in user_text(second)
    # After the good turn, the progress line names it as the best.
    assert "(turn 2)" in user_text(client.calls[2]["messages"][-1])
    # The rejected turn minted no result handles; the good one minted two.
    assert len(mcp_server._RESULTS) == len(before) + 2
    assert agent._select_best(out["history"])["params"] == {"lookback": 60, "top_n": 0}


def test_declare_done_before_any_success_is_refused(windows):
    client = FakeClient(
        done_response("nothing to do", tool_id="toolu_early"),
        propose_response("momentum", {"lookback": 60}, tool_id="toolu_1"),
        done_response("now done", tool_id="toolu_late"),
    )
    out = run_loop(client, windows, max_iters=5)
    assert out["stopped_because"] == "converged"
    assert out["n_model_calls"] == 3
    assert [r["iter"] for r in out["history"]] == [2]
    second = client.calls[1]["messages"]
    assert second[1]["content"] is client.calls[1]["messages"][1]["content"]
    results = tool_results(second[-1])
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "toolu_early"
    assert results[0]["is_error"] is True
    assert "at least one successful experiment" in results[0]["content"]
    assert "Turn 1 of 5 used; 4 remaining" in user_text(second[-1])
    # A converged run ends on the assistant's declare_done: no trailing user message.
    assert len(client.calls[2]["messages"]) == 5


def test_declare_done_after_only_failed_experiments_is_refused(windows):
    client = FakeClient(
        propose_response("momentum", {"lookback": 999}, tool_id="toolu_bad"),
        done_response(tool_id="toolu_early"),
        propose_response("momentum", {"lookback": 60}, tool_id="toolu_ok"),
        done_response(tool_id="toolu_late"),
    )
    out = run_loop(client, windows, max_iters=5)
    assert out["stopped_because"] == "converged"
    assert out["n_model_calls"] == 4
    results = tool_results(client.calls[2]["messages"][-1])
    assert results[0]["tool_use_id"] == "toolu_early" and results[0]["is_error"] is True


def test_multiple_tool_use_blocks_only_the_first_runs(windows, monkeypatch):
    calls = _spy_run_backtest(monkeypatch)
    multi = multi_tool_response(
        (
            "propose_experiment",
            {"strategy": "momentum", "params": {"lookback": 60}, "rationale": "a"},
            "toolu_a",
        ),
        (
            "propose_experiment",
            {"strategy": "momentum", "params": {"lookback": 120}, "rationale": "b"},
            "toolu_b",
        ),
    )
    client = FakeClient(multi, done_response())
    out = run_loop(client, windows, max_iters=5)
    assert out["stopped_because"] == "converged"
    assert len(out["history"]) == 1
    assert out["history"][0]["params"] == {"lookback": 60, "top_n": 0}
    assert len(calls) == 2  # one experiment: train + validation
    closing = client.calls[1]["messages"][-1]
    results = tool_results(closing)
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]
    assert results[0].get("is_error", False) is False
    assert results[1]["is_error"] is True and "ignored" in results[1]["content"]
    assert closing["content"][-1]["type"] == "text"
    assert len(closing["content"]) == 3


def test_text_only_turn_is_consumed_without_a_record(windows):
    client = FakeClient(
        text_only_response("Let me think about this first."),
        propose_response("momentum", {"lookback": 60}, tool_id="toolu_1"),
        propose_response("momentum", {"lookback": 120}, tool_id="toolu_2"),
    )
    out = run_loop(client, windows, max_iters=3)
    assert out["stopped_because"] == "max_iters"
    assert out["n_model_calls"] == 3 and len(client.calls) == 3
    assert [r["iter"] for r in out["history"]] == [2, 3]
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[1]["content"] is client.calls[1]["messages"][1]["content"]
    assert "must call exactly one tool" in user_text(second[2])
    assert "Turn 1 of 3 used; 2 remaining" in user_text(second[2])
    # No tool_use means no tool_result: the corrective message is text only (promoted to one
    # cached text block by ``_call``).
    assert [b["type"] for b in second[2]["content"]] == ["text"]


def test_text_only_turns_alone_exhaust_the_cap(windows):
    client = FakeClient(*[text_only_response("hmm") for _ in range(2)])
    out = run_loop(client, windows, max_iters=2)
    assert out["stopped_because"] == "max_iters"
    assert out["history"] == [] and out["n_model_calls"] == 2


def test_unknown_tool_name_gets_an_error_result_and_no_record(windows):
    unknown = tool_use_response("run_backtest", {"dataset_id": "ds_x"}, block_id="toolu_unk")
    client = FakeClient(unknown, propose_response("momentum", {"lookback": 60}), done_response())
    out = run_loop(client, windows, max_iters=5)
    assert out["stopped_because"] == "converged"
    assert [r["iter"] for r in out["history"]] == [2]
    results = tool_results(client.calls[1]["messages"][-1])
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "toolu_unk"
    assert results[0]["is_error"] is True and results[0]["content"] == "unknown tool"
    assert mcp_server._RESULTS != {} and len(mcp_server._RESULTS) == 2


def test_non_dict_proposal_input_is_a_rejected_record(windows, monkeypatch):
    calls = _spy_run_backtest(monkeypatch)
    weird = tool_use_response("propose_experiment", "momentum lookback=60", block_id="toolu_str")
    client = FakeClient(weird, propose_response("momentum", {"lookback": 60}), done_response())
    out = run_loop(client, windows, max_iters=5)
    bad = out["history"][0]
    assert bad["iter"] == 1 and bad["strategy"] is None
    assert bad["train_metrics"] is None and "object" in bad["error"]
    assert tool_results(client.calls[1]["messages"][-1])[0]["is_error"] is True
    assert len(calls) == 2  # only the good proposal reached the tool
    assert out["stopped_because"] == "converged"


def test_loop_source_never_names_the_holdout_machinery():
    src = inspect.getsource(agent._research_loop)
    for forbidden in ("split_data", "score_holdout", "HoldoutHandle", "_score"):
        assert forbidden not in src, forbidden
    for helper in (agent._run_proposal, agent._progress_text):
        for forbidden in ("split_data", "score_holdout", "HoldoutHandle"):
            assert forbidden not in inspect.getsource(helper), (helper.__name__, forbidden)


def test_conversation_carries_no_holdout_date_or_handle(windows):
    client = FakeClient(
        propose_response("momentum", {"lookback": 60}, tool_id="toolu_1"),
        propose_response("momentum", {"lookback": 120}, tool_id="toolu_2"),
        done_response(),
    )
    run_loop(client, windows, max_iters=5)
    bounds = loader.get_split_bounds()
    for req in client.calls:
        transcript = json.dumps([m for m in req["messages"] if m["role"] == "user"], default=str)
        assert not HOLDOUT_DATE_RE.search(transcript)
        for date in bounds["holdout"]:
            assert date not in transcript
        assert "res_" not in transcript and "ds_" not in transcript


def test_public_mode_code_proposal_runs_nothing_and_the_loop_continues(windows, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1.0")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "5.0")
    calls = _spy_run_backtest(monkeypatch)
    client = FakeClient(
        propose_response("momentum", {"code": "import os"}, "sneaky", tool_id="toolu_code"),
        propose_response("momentum", {"lookback": 60}, "honest", tool_id="toolu_ok"),
        done_response(),
    )
    before = dict(mcp_server._RESULTS)
    out = run_loop(client, windows, max_iters=5)
    assert out["stopped_because"] == "converged"
    bad, good = out["history"]
    assert bad["error"] is not None and "public mode" in bad["error"].lower()
    assert "code" in bad["error"]
    assert bad["train_metrics"] is None and bad["val_metrics"] is None
    assert good["error"] is None and good["params"] == {"lookback": 60, "top_n": 0}
    # The gate fired before the tool was even reached: the spy saw only the honest proposal.
    assert [args[0] for args, _ in calls] == list(windows)
    assert len(mcp_server._RESULTS) == len(before) + 2
    results = tool_results(client.calls[1]["messages"][-1])
    assert results[0]["is_error"] is True and results[0]["content"] == bad["error"]


def test_loop_never_builds_its_own_client(windows):
    # ``client`` is injected; the loop must not fall back to ``_get_client`` (which explodes here).
    client = FakeClient(propose_response("momentum", {"lookback": 60}), done_response())
    out = run_loop(client, windows, max_iters=2)
    assert out["stopped_because"] == "converged"


# ---------------------------------------------------------------- (12) run_research


RESULT_KEYS = {"best", "holdout_metrics", "history", "stopped_because", "spend_usd", "error"}
# Bound to the ORIGINALS at import so a test that spies on the module attributes can still
# compute the reference number through the unpatched functions.
_real_split_data = guardrails.split_data
_real_score_holdout = guardrails.score_holdout


def direct_holdout(best: dict, prices: pd.DataFrame = _SYNTHETIC, engine: str = "python") -> dict:
    """The reference holdout score: a fresh handle from ``prices``, scored once, no agent."""
    handle = _real_split_data(prices)[2]
    return _real_score_holdout(handle, best["strategy"], best["params"], engine)


def sequenced_client(seq: list[str], *responses) -> FakeClient:
    """A ``FakeClient`` that also appends ``"create"`` to ``seq`` on every model call."""
    client = FakeClient(*responses)
    original = client.messages.create

    def create(**kwargs):
        seq.append("create")
        return original(**kwargs)

    client.messages = SimpleNamespace(create=create)
    return client


def spy_holdout(monkeypatch, seq: list[str]) -> dict[str, list]:
    """Wrap ``guardrails.split_data``/``score_holdout`` (originals still run); record the order.

    ``calls["split"]`` holds ``(prices, handle)`` per call and ``calls["score"]`` the positional
    arguments per call, so a test can check the exact frame, handle and config that reached the
    holdout machinery — and that nothing reached it at all when there was no winner.
    """
    calls: dict[str, list] = {"split": [], "score": []}

    def split(prices=None):
        seq.append("split_data")
        out = _real_split_data(prices)
        calls["split"].append((prices, out[2]))
        return out

    def score(handle, strategy, params, engine="python"):
        seq.append("score_holdout")
        calls["score"].append((handle, strategy, params, engine))
        return _real_score_holdout(handle, strategy, params, engine)

    monkeypatch.setattr(guardrails, "split_data", split)
    monkeypatch.setattr(guardrails, "score_holdout", score)
    return calls


def config_of(record: dict) -> dict:
    return {k: record[k] for k in agent._CONFIG_KEYS}


def assert_no_handles_or_ids(result: dict) -> None:
    """The contract carries plain JSON only: no handle objects, frames, or registry ids."""
    text = json.dumps(result)  # raises on a HoldoutHandle or DataFrame
    assert "ds_" not in text and "res_" not in text and "HoldoutHandle" not in text

    def walk(value):
        assert not isinstance(value, (guardrails.HoldoutHandle, pd.DataFrame, pd.Series))
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(result)


def test_run_research_golden_end_to_end(isolated):
    usages = [
        _usage(1000, 200, cache_create=3000),
        _usage(400, 180, cache_read=3000),
        _usage(300, 60),
    ]
    client = FakeClient(
        propose_response("momentum", {"lookback": 60}, "baseline", usage=usages[0], tool_id="t1"),
        propose_response("momentum", {"lookback": 120}, "slower", usage=usages[1], tool_id="t2"),
        done_response("no further gains", usage=usages[2]),
    )

    out = agent.run_research(GOAL, max_iters=5, client=client)

    assert set(out) == RESULT_KEYS
    assert tuple(out) == agent._RESULT_KEYS
    assert out["stopped_because"] == "converged"
    assert out["error"] is None
    assert len(client.calls) == 3

    history = out["history"]
    assert [r["iter"] for r in history] == [1, 2]
    assert [r["params"]["lookback"] for r in history] == [60, 120]
    assert all(r["error"] is None for r in history)

    # Best is the record with the higher VALIDATION Sharpe — asserted against both records.
    first, second = history
    winner, loser = (
        (first, second)
        if first["val_metrics"]["sharpe"] >= second["val_metrics"]["sharpe"]
        else (second, first)
    )
    assert out["best"] == config_of(winner)
    assert out["best"] != config_of(loser)
    assert out["best"] is not winner and out["best"]["params"] is not winner["params"]

    # The one-shot holdout number: the documented keys, plain floats, equal to a direct score.
    holdout = out["holdout_metrics"]
    assert list(holdout) == _KEYS
    assert all(type(v) is float for v in holdout.values())
    reference = direct_holdout(out["best"])
    for key in _KEYS:
        assert holdout[key] == pytest.approx(reference[key], abs=1e-12), key

    assert out["spend_usd"] == pytest.approx(sum(usd_of(u) for u in usages))
    assert today_bucket(isolated)["usd"] == pytest.approx(out["spend_usd"])
    assert_no_handles_or_ids(out)


def test_run_research_best_is_chosen_by_validation_sharpe_not_train(monkeypatch):
    """Train and validation winners differ by construction; best must follow validation."""
    original = mcp_server.run_backtest
    sharpe_by_window = {  # (is_train, lookback) -> sharpe
        (True, 60): 2.0,
        (True, 120): 1.0,
        (False, 60): 0.5,
        (False, 120): 1.5,
    }

    def rigged(dataset_id, strategy, params=None, engine="python", cost_bps=10.0):
        out = original(dataset_id, strategy, params, engine, cost_bps)
        is_train = mcp_server._DATASETS[dataset_id].index[0].year < 2020
        out["metrics"]["sharpe"] = sharpe_by_window[(is_train, params["lookback"])]
        return out

    monkeypatch.setattr(mcp_server, "run_backtest", rigged)
    client = FakeClient(
        propose_response("momentum", {"lookback": 60}, tool_id="t1"),
        propose_response("momentum", {"lookback": 120}, tool_id="t2"),
        done_response(),
    )
    out = agent.run_research(GOAL, max_iters=5, client=client)

    train_winner, val_winner = out["history"]
    assert train_winner["train_metrics"]["sharpe"] > val_winner["train_metrics"]["sharpe"]
    assert val_winner["val_metrics"]["sharpe"] > train_winner["val_metrics"]["sharpe"]
    assert out["best"] == config_of(val_winner)
    assert out["best"]["params"]["lookback"] == 120
    # The holdout is scored for the validation winner, by the real engine (never rigged).
    reference = direct_holdout(out["best"])
    for key in _KEYS:
        assert out["holdout_metrics"][key] == pytest.approx(reference[key], abs=1e-12), key


def test_run_research_scores_holdout_exactly_once_after_the_last_model_call(monkeypatch):
    seq: list[str] = []
    calls = spy_holdout(monkeypatch, seq)
    client = sequenced_client(
        seq,
        propose_response("momentum", {"lookback": 60}, tool_id="t1"),
        propose_response("momentum", {"lookback": 120}, tool_id="t2"),
        done_response(),
    )
    out = agent.run_research(GOAL, max_iters=5, client=client)

    assert seq == ["create", "create", "create", "split_data", "score_holdout"]
    assert len(calls["split"]) == 1 and len(calls["score"]) == 1
    prices, handle = calls["split"][0]
    assert set(prices["ticker"]) == set(TICKERS)
    assert calls["score"][0] == (handle, out["best"]["strategy"], out["best"]["params"], "python")
    assert handle.consumed is True
    with pytest.raises(guardrails.HoldoutAlreadyScored):
        _real_score_holdout(handle, "momentum", {"lookback": 60}, "python")
    assert_no_handles_or_ids(out)


def test_run_research_all_rejected_means_no_best_and_no_holdout_call(monkeypatch):
    seq: list[str] = []
    calls = spy_holdout(monkeypatch, seq)
    client = sequenced_client(
        seq,
        propose_response("momentum", {"lookback": 999}, tool_id="t1"),
        propose_response("momentum", {"lookback": 999}, tool_id="t2"),
    )
    out = agent.run_research(GOAL, max_iters=2, client=client)

    assert out["best"] is None
    assert out["holdout_metrics"] is None
    assert out["stopped_because"] == "max_iters"
    assert out["error"] is None
    assert len(out["history"]) == 2 and all(r["error"] for r in out["history"])
    assert seq == ["create", "create"]
    assert calls == {"split": [], "score": []}
    assert_no_handles_or_ids(out)


def test_run_research_budget_cap_zero_stops_before_any_call(monkeypatch, isolated):
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "0")
    seq: list[str] = []
    calls = spy_holdout(monkeypatch, seq)
    client = sequenced_client(seq, propose_response("momentum", {"lookback": 60}))
    out = agent.run_research(GOAL, max_iters=5, client=client)

    assert out == {
        "best": None,
        "holdout_metrics": None,
        "history": [],
        "stopped_because": "budget",
        "spend_usd": 0.0,
        "error": None,
    }
    assert client.calls == [] and seq == []
    assert calls == {"split": [], "score": []}
    assert today_bucket(isolated) is None


def test_run_research_ticker_subset_bounds_both_windows_and_the_holdout(monkeypatch):
    seq: list[str] = []
    calls = spy_holdout(monkeypatch, seq)
    client = FakeClient(propose_response("momentum", {"lookback": 60}), done_response())
    out = agent.run_research(GOAL, max_iters=3, tickers=["AAPL", "MSFT"], client=client)

    # The agent's only data: two load_data handles on the fixed windows, this universe only.
    assert len(mcp_server._DATASETS) == 2
    lo, hi = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")
    starts = []
    for wide in mcp_server._DATASETS.values():
        assert list(wide.columns) == ["AAPL", "MSFT"]
        assert wide.index.min() >= lo and wide.index.max() < hi
        starts.append(wide.index.min())
    assert pd.Timestamp("2020-01-01", tz="UTC") in starts
    assert min(starts) == lo

    # The holdout was cut from the same universe the agent iterated on.
    prices, _ = calls["split"][0]
    assert set(prices["ticker"]) == {"AAPL", "MSFT"}
    subset = _SYNTHETIC[_SYNTHETIC["ticker"].isin(["AAPL", "MSFT"])].reset_index(drop=True)
    reference = direct_holdout(out["best"], subset)
    for key in _KEYS:
        assert out["holdout_metrics"][key] == pytest.approx(reference[key], abs=1e-12), key


def test_run_research_unknown_ticker_raises_before_any_model_call(isolated):
    client = FakeClient(propose_response("momentum", {"lookback": 60}))
    with pytest.raises(ValueError, match="not in the fixed universe"):
        agent.run_research(GOAL, max_iters=3, tickers=["AAPL", "ZZZZ"], client=client)
    assert client.calls == []
    assert today_bucket(isolated) is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"engine": "r"}, "registered engines"),
        ({"cost_bps": 101}, "cost_bps"),
        ({"max_iters": 11}, "MAX_AGENT_ITERS"),
        ({"goal": ""}, "goal"),
    ],
)
def test_run_research_rejects_bad_arguments_before_any_spend(isolated, kwargs, match):
    client = FakeClient(propose_response("momentum", {"lookback": 60}))
    call = {"goal": GOAL, "max_iters": 3, "client": client, **kwargs}
    with pytest.raises(ValueError, match=match):
        agent.run_research(call.pop("goal"), **call)
    assert client.calls == []
    assert today_bucket(isolated) is None
    assert mcp_server._DATASETS == {}


def test_run_research_engine_error_names_the_registered_engines():
    with pytest.raises(ValueError, match=re.escape(str(sorted(mcp_server.ENGINES)))):
        agent.run_research(GOAL, engine="r", client=FakeClient())


def test_run_research_api_error_still_scores_the_best_so_far_once(monkeypatch, isolated):
    seq: list[str] = []
    calls = spy_holdout(monkeypatch, seq)
    err = api_connection_error()
    client = sequenced_client(
        seq, propose_response("momentum", {"lookback": 60}, tool_id="t1"), err
    )
    out = agent.run_research(GOAL, max_iters=5, client=client)

    assert out["stopped_because"] == "api_error"
    assert out["error"] == str(err)
    assert len(out["history"]) == 1
    assert out["best"] == config_of(out["history"][0])
    assert seq == ["create", "create", "split_data", "score_holdout"]
    assert len(calls["score"]) == 1
    reference = direct_holdout(out["best"])
    for key in _KEYS:
        assert out["holdout_metrics"][key] == pytest.approx(reference[key], abs=1e-12), key
    assert today_bucket(isolated)["calls"] == 1
    assert_no_handles_or_ids(out)


def test_run_research_public_mode_parameter_only_run_matches_dev_mode(monkeypatch):
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1.0")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "5.0")

    def responses():
        return (
            propose_response("momentum", {"lookback": 60}, "a", tool_id="t1"),
            propose_response("mean_reversion", {}, "b", tool_id="t2"),
            done_response(),
        )

    dev = agent.run_research(GOAL, max_iters=5, client=FakeClient(*responses()))
    mcp_server.reset_registry()
    monkeypatch.setenv("PUBLIC_MODE", "on")
    public = agent.run_research(GOAL, max_iters=5, client=FakeClient(*responses()))

    assert public["stopped_because"] == "converged"
    assert all(r["error"] is None for r in public["history"])
    assert json.dumps(public, sort_keys=True) == json.dumps(dev, sort_keys=True)


def test_run_research_builds_the_sdk_client_only_when_none_is_given():
    # The fixture makes ``_get_client`` explode; that must be reached only AFTER validation.
    with pytest.raises(AssertionError, match="real client"):
        agent.run_research(GOAL, max_iters=2)
    assert mcp_server._DATASETS == {}
    with pytest.raises(ValueError):
        agent.run_research("", max_iters=2)  # invalid goal: never gets as far as the client


def test_run_research_source_scores_the_holdout_outside_the_loop():
    src = inspect.getsource(agent.run_research)
    assert src.index("_research_loop(") < src.index("_select_best(")
    assert src.index("_select_best(") < src.index("guardrails.split_data(")
    assert src.index("guardrails.split_data(") < src.index("guardrails.score_holdout(")
    assert src.count("guardrails.score_holdout(") == 1
    assert "handle" not in src[src.index("result = {") :]


# ---------------------------------------------------------------- (13) live smoke (opt-in)

# QUANTFORGE_LIVE_AI is a TEST-ONLY opt-in flag (see tests/test_nl_interface.py): a developer
# willing to spend real money on one end-to-end run sets it; it is never set in CI.


@pytest.mark.skipif(
    os.environ.get("QUANTFORGE_LIVE_AI") != "1",
    reason="live API smoke; set QUANTFORGE_LIVE_AI=1 to run (spends real money, never in CI)",
)
def test_live_smoke():
    """Real Sonnet loop, two turns, on the synthetic panel. Ledger still isolated to tmp_path."""
    out = agent.run_research(GOAL, max_iters=2, client=anthropic.Anthropic())
    assert set(out) == RESULT_KEYS
    assert out["stopped_because"] in agent.STOP_REASONS
    assert out["spend_usd"] > 0
    assert 1 <= len(out["history"]) <= 2
    if out["best"] is not None:
        assert list(out["holdout_metrics"]) == _KEYS
    assert_no_handles_or_ids(out)


def test_live_smoke_is_skipped_unless_opted_in():
    marks = [m for m in test_live_smoke.pytestmark if m.name == "skipif"]
    assert len(marks) == 1
    assert marks[0].args[0] is (os.environ.get("QUANTFORGE_LIVE_AI") != "1")
    assert "QUANTFORGE_LIVE_AI=1" in marks[0].kwargs["reason"]
