"""Tests for ``quantforge.ai.nl_interface`` (component 12): parse / explain / handle.

Fully offline by construction: a ``FakeClient`` stands in for ``anthropic.Anthropic`` (records
every ``messages.create`` request, returns canned responses with real-looking ``usage``), the
MCP tools read a synthetic price panel, and the budget ledger / rate-limit state live in
``tmp_path``. ``ANTHROPIC_API_KEY`` is never read — the module builds its client lazily and every
test passes ``client=`` explicitly — so the verifier can run this file with the key unset.

The one live test, ``test_live_smoke``, is skipped unless ``QUANTFORGE_LIVE_AI=1``: it spends
real money and is for a developer's manual check, never CI.
"""

from __future__ import annotations

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
from quantforge.ai import budget, guardrails, mcp_server, nl_interface
from quantforge.ai.budget import PRICES_PER_MTOK
from quantforge.data import loader
from quantforge.metrics.performance import _KEYS
from quantforge.strategies import STRATEGIES

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM"]
THREE = ["AAPL", "MSFT", "NVDA"]
MODEL = nl_interface.MODEL


# ---------------------------------------------------------------- fakes


def _usage(tokens_in: int, tokens_out: int, cache_create: int = 0, cache_read: int = 0):
    return SimpleNamespace(
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cache_creation_input_tokens=cache_create,
        cache_read_input_tokens=cache_read,
    )


def tool_use_response(plan: dict, usage=None, name: str = "plan_backtest"):
    """A canned parse response: one ``plan_backtest`` tool_use block."""
    block = SimpleNamespace(type="tool_use", id="toolu_01", name=name, input=plan)
    return SimpleNamespace(
        content=[block], usage=usage or _usage(1000, 200), stop_reason="tool_use"
    )


def text_response(text: str, usage=None):
    """A canned explain response: one text block (empty string -> block with empty text)."""
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block], usage=usage or _usage(800, 150), stop_reason="end_turn")


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


GOLDEN_PLAN = {"strategy": "momentum", "start": "2015-01-01", "end": "2019-12-31", "tickers": THREE}


def golden_client(explanation: str = "Momentum did fine.") -> FakeClient:
    return FakeClient(tool_use_response(GOLDEN_PLAN), text_response(explanation))


def expected_usd(*usages) -> float:
    """Hand-computed cost of the fake usages at the module's prices (cache tokens at full rate)."""
    in_rate, out_rate = PRICES_PER_MTOK[MODEL]
    total = 0.0
    for u in usages:
        tokens_in = u.input_tokens + u.cache_creation_input_tokens + u.cache_read_input_tokens
        total += tokens_in / 1e6 * in_rate + u.output_tokens / 1e6 * out_rate
    return total


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
    """Synthetic prices, temp ledger + rate state, dev-mode env, no real client, no API key."""
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
    monkeypatch.setattr(nl_interface, "_client", None)
    monkeypatch.setattr(
        nl_interface, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("real client"))
    )
    yield ledger
    mcp_server.reset_registry()


def today_bucket(ledger) -> dict | None:
    if not ledger.exists():
        return None
    return json.loads(ledger.read_text())["days"].get(budget._utc_today())


# ---------------------------------------------------------------- module surface


def test_constants_and_signature():
    assert nl_interface.MODEL == "claude-haiku-4-5"
    assert nl_interface._EST_TOKENS_IN == 2500 and nl_interface._EST_TOKENS_OUT == 600
    sig = inspect.signature(nl_interface.handle)
    assert list(sig.parameters) == ["query", "session_key", "client"]
    assert sig.parameters["session_key"].default == "anon"
    assert sig.parameters["session_key"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "public_mode" not in sig.parameters
    src = inspect.getsource(nl_interface)
    assert re.findall(r"claude-[a-z0-9-]+", src) == ["claude-haiku-4-5"] * src.count("claude-")


def test_plan_tool_schema_mirrors_pipeline_inputs():
    tool = nl_interface.PLAN_TOOL
    assert tool["name"] == "plan_backtest"
    props = tool["input_schema"]["properties"]
    assert set(props) == {"strategy", "params", "tickers", "start", "end", "cost_bps", "clarify"}
    assert props["strategy"]["enum"] == sorted(STRATEGIES)
    assert props["tickers"]["items"]["enum"] == list(loader.UNIVERSE)
    run_bt = next(s for s in mcp_server.TOOL_SCHEMAS if s["name"] == "run_backtest")
    assert props["params"] == run_bt["input_schema"]["properties"]["params"]
    assert props["params"]["additionalProperties"] is False
    assert tool["input_schema"]["additionalProperties"] is False
    assert (props["cost_bps"]["minimum"], props["cost_bps"]["maximum"]) == (0, 100)
    assert "ONLY" in props["clarify"]["description"]
    json.dumps(tool)  # wire-serializable


def test_system_prompts_are_frozen_and_name_the_facts():
    bounds = loader.get_split_bounds()
    for text in (nl_interface.SYSTEM_PARSE, nl_interface.SYSTEM_EXPLAIN):
        assert isinstance(text, str) and text
        for name in STRATEGIES:
            assert name in text
        assert bounds["train"][0] in text and bounds["validation"][1] in text
        assert not re.search(r"20\d\d-\d\d-\d\dT", text)  # no timestamps
    assert "never invent a date range or strategy" in nl_interface.SYSTEM_PARSE
    assert "clarify" in nl_interface.SYSTEM_PARSE
    for ticker in loader.UNIVERSE:
        assert ticker in nl_interface.SYSTEM_PARSE


# ---------------------------------------------------------------- golden path


def test_handle_golden_path(isolated):
    client = golden_client("Momentum on three names returned a bit.")
    out = nl_interface.handle("backtest momentum on AAPL MSFT NVDA 2015-2019", client=client)

    assert set(out) == {"plan", "metrics", "explanation", "spend_usd", "fallback"}
    assert out["fallback"] is None
    assert list(out["metrics"]) == _KEYS
    assert out["explanation"] == "Momentum on three names returned a bit."
    assert out["plan"] == {
        "strategy": "momentum",
        "params": {"lookback": 126, "top_n": 0},
        "tickers": THREE,
        "start": "2015-01-01",
        "end": "2019-12-31",
        "cost_bps": 10.0,
    }
    want = expected_usd(_usage(1000, 200), _usage(800, 150))
    assert out["spend_usd"] == pytest.approx(want)
    assert want == pytest.approx(1000 / 1e6 * 1 + 200 / 1e6 * 5 + 800 / 1e6 * 1 + 150 / 1e6 * 5)

    bucket = today_bucket(isolated)
    assert bucket["calls"] == 2
    assert bucket["usd"] == pytest.approx(want)
    assert bucket["tokens_in"] == 1800 and bucket["tokens_out"] == 350
    assert len(client.calls) == 2


def test_golden_metrics_equal_direct_tool_pipeline():
    out = nl_interface.handle("go", client=golden_client())
    ds = mcp_server.load_data("2015-01-01", "2019-12-31", THREE)["dataset_id"]
    direct = mcp_server.run_backtest(ds, "momentum", {}, "python", 10.0)["metrics"]
    assert out["metrics"] == direct


def test_parse_request_shape_uses_model_forced_tool_and_cache_breakpoints():
    client = golden_client()
    nl_interface.handle("backtest momentum 2015-2019", client=client)
    parse_kw, explain_kw = client.calls

    assert parse_kw["model"] == MODEL
    assert parse_kw["max_tokens"] == 512
    assert parse_kw["tool_choice"] == {"type": "tool", "name": "plan_backtest"}
    assert parse_kw["system"][0]["text"] == nl_interface.SYSTEM_PARSE
    assert parse_kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert parse_kw["tools"][0]["name"] == "plan_backtest"
    assert parse_kw["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert parse_kw["messages"] == [{"role": "user", "content": "backtest momentum 2015-2019"}]
    assert "cache_control" not in nl_interface.PLAN_TOOL  # module constant not mutated

    assert explain_kw["model"] == MODEL
    assert explain_kw["max_tokens"] == 400
    assert "tools" not in explain_kw and "tool_choice" not in explain_kw
    assert explain_kw["system"][0]["text"] == nl_interface.SYSTEM_EXPLAIN
    assert explain_kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    payload = json.loads(explain_kw["messages"][0]["content"])
    assert set(payload) == {"plan", "metrics"}
    assert list(payload["metrics"]) == sorted(_KEYS)  # sort_keys=True -> stable bytes


def test_charge_receives_actual_usage_including_cache_tokens(monkeypatch):
    seen: list[dict] = []
    real = budget.charge

    def spy(usd, *, model, tokens_in, tokens_out):
        seen.append({"usd": usd, "model": model, "in": tokens_in, "out": tokens_out})
        return real(usd, model=model, tokens_in=tokens_in, tokens_out=tokens_out)

    monkeypatch.setattr(budget, "charge", spy)
    u1 = _usage(100, 20, cache_create=1500, cache_read=0)
    u2 = _usage(50, 10, cache_create=0, cache_read=1500)
    client = FakeClient(tool_use_response(GOLDEN_PLAN, u1), text_response("ok", u2))
    out = nl_interface.handle("q", client=client)
    assert [s["model"] for s in seen] == [MODEL, MODEL]
    assert (seen[0]["in"], seen[0]["out"]) == (1600, 20)
    assert (seen[1]["in"], seen[1]["out"]) == (1550, 10)
    assert seen[0]["usd"] == pytest.approx(budget.estimate(MODEL, 1600, 20))
    assert out["spend_usd"] == pytest.approx(expected_usd(u1, u2))


def test_usage_fields_missing_or_none_count_as_zero(isolated):
    usage = SimpleNamespace(input_tokens=300, output_tokens=None)  # no cache fields at all
    client = FakeClient(tool_use_response(GOLDEN_PLAN, usage), text_response("ok", usage))
    out = nl_interface.handle("q", client=client)
    assert out["fallback"] is None
    assert out["spend_usd"] == pytest.approx(2 * 300 / 1e6 * 1)
    assert today_bucket(isolated)["tokens_out"] == 0


# ---------------------------------------------------------------- gates


def test_gate_order_rate_budget_parse_codegen_tools_explain(monkeypatch):
    order: list[str] = []

    monkeypatch.setattr(guardrails, "rate_limit", lambda key: order.append("rate") or True)
    monkeypatch.setattr(budget, "allow", lambda est: order.append("budget") or True)
    real_codegen = guardrails.assert_no_codegen
    monkeypatch.setattr(
        guardrails, "assert_no_codegen", lambda a: order.append("codegen") or real_codegen(a)
    )
    real_load = mcp_server.load_data
    monkeypatch.setattr(
        mcp_server, "load_data", lambda *a, **k: order.append("tools") or real_load(*a, **k)
    )
    real_call = nl_interface._call

    def call_spy(client, **kw):
        order.append("parse" if "tools" in kw else "explain")
        return real_call(client, **kw)

    monkeypatch.setattr(nl_interface, "_call", call_spy)

    out = nl_interface.handle("q", client=golden_client())
    assert out["fallback"] is None
    # run_backtest also calls the codegen gate internally (it is a tool), hence the second entry.
    assert order == ["rate", "budget", "parse", "codegen", "tools", "codegen", "explain"]


def test_rate_gate_closed_means_no_client_call_and_no_ledger(isolated, monkeypatch):
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "0")
    client = golden_client()
    out = nl_interface.handle("q", session_key="s1", client=client)
    assert out["fallback"] == "rate"
    assert out["plan"] is None and out["metrics"] is None
    assert out["spend_usd"] == 0.0
    assert isinstance(out["explanation"], str) and out["explanation"]
    assert client.calls == []
    assert today_bucket(isolated) is None


def test_rate_gate_consulted_once_per_query(isolated, monkeypatch):
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "2")
    assert nl_interface.handle("q", session_key="k", client=golden_client())["fallback"] is None
    assert nl_interface.handle("q", session_key="k", client=golden_client())["fallback"] is None
    assert nl_interface.handle("q", session_key="k", client=golden_client())["fallback"] == "rate"
    assert nl_interface.handle("q", session_key="other", client=golden_client())["fallback"] is None


@pytest.mark.parametrize("env", [("AI_BUDGET_USD_DAILY", "0"), ("AI_DISABLED", "on")])
def test_budget_gate_closed_means_no_call(isolated, monkeypatch, env):
    monkeypatch.setenv(*env)
    client = golden_client()
    out = nl_interface.handle("q", client=client)
    assert out["fallback"] == "budget"
    assert out["plan"] is None and out["metrics"] is None and out["spend_usd"] == 0.0
    assert client.calls == []
    assert today_bucket(isolated) is None


def test_budget_gate_asks_for_both_calls_up_front(monkeypatch):
    seen = []
    monkeypatch.setattr(budget, "allow", lambda est: seen.append(est) or False)
    nl_interface.handle("q", client=golden_client())
    per_call = budget.estimate(MODEL, 2500, 600)
    assert seen == [pytest.approx(2 * per_call)]


# ---------------------------------------------------------------- parse outcomes


def test_clarify_path_runs_no_tools_and_charges_once(isolated, monkeypatch):
    monkeypatch.setattr(mcp_server, "load_data", lambda *a, **k: pytest.fail("tools ran"))
    client = FakeClient(tool_use_response({"clarify": "Which strategy and which years?"}))
    out = nl_interface.handle("tell me a joke", client=client)
    assert out["fallback"] is None
    assert out["plan"] == {"clarify": "Which strategy and which years?"}
    assert out["metrics"] is None
    assert out["explanation"] == "Which strategy and which years?"
    assert len(client.calls) == 1
    assert today_bucket(isolated)["calls"] == 1
    assert out["spend_usd"] == pytest.approx(expected_usd(_usage(1000, 200)))


@pytest.mark.parametrize(
    "plan",
    [
        {"strategy": "momentum", "start": "2015-01-01"},  # no end
        {"strategy": "momentum", "end": "2019-12-31"},  # no start
        {"start": "2015-01-01", "end": "2019-12-31"},  # no strategy
        {},
    ],
)
def test_parse_never_guesses_essentials(plan):
    out = nl_interface.parse("q", client=FakeClient(tool_use_response(plan)))
    assert set(out) == {"clarify"}
    assert "strategy" in out["clarify"] and "date range" in out["clarify"]


def test_parse_without_tool_use_block_clarifies():
    out = nl_interface.parse("q", client=FakeClient(text_response("I am prose.")))
    assert out == {
        "clarify": "I couldn't turn that into a backtest — which strategy and date range?"
    }


def test_parse_applies_defaults_and_validates_params():
    plan = {
        "strategy": "mean_reversion",
        "start": "2016-01-01",
        "end": "2018-01-01",
        "params": {"lookback": 10},
        "cost_bps": 5,
    }
    out = nl_interface.parse("q", client=FakeClient(tool_use_response(plan)))
    assert out["tickers"] == list(loader.UNIVERSE)
    assert out["cost_bps"] == 5
    assert out["params"] == {"lookback": 10, "entry_z": 2.0, "exit_z": 0.5, "mode": "long_flat"}
    assert set(out) == {"strategy", "params", "tickers", "start", "end", "cost_bps"}


def test_parse_whitelist_violation_becomes_clarify_in_dev_mode():
    plan = {**GOLDEN_PLAN, "params": {"lookback": 999}}
    out = nl_interface.parse("q", client=FakeClient(tool_use_response(plan)))
    assert set(out) == {"clarify"}
    assert "lookback" in out["clarify"] and "252" in out["clarify"]


@pytest.mark.parametrize("bad", ["", "   ", "x" * 2001, None, 42])
def test_parse_rejects_bad_query_before_any_spend(isolated, bad):
    client = golden_client()
    with pytest.raises(ValueError):
        nl_interface.parse(bad, client=client)
    assert client.calls == [] and today_bucket(isolated) is None


def test_parse_with_spend_returns_charge():
    plan, usd = nl_interface.parse_with_spend(
        "q", client=FakeClient(tool_use_response(GOLDEN_PLAN))
    )
    assert plan["strategy"] == "momentum"
    assert usd == pytest.approx(expected_usd(_usage(1000, 200)))


# ---------------------------------------------------------------- public mode / invalid / api


def test_public_mode_rejects_unwhitelisted_param_after_one_charge(isolated, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "5")
    monkeypatch.setattr(mcp_server, "load_data", lambda *a, **k: pytest.fail("tools ran"))
    plan = {**GOLDEN_PLAN, "params": {"lookback": 60, "code": "x"}}
    client = FakeClient(tool_use_response(plan), text_response("never"))
    out = nl_interface.handle("q", client=client)
    assert out["fallback"] == "public_mode"
    assert out["metrics"] is None and out["plan"] is None
    assert "code" in out["explanation"]
    assert len(client.calls) == 1
    assert today_bucket(isolated)["calls"] == 1
    assert out["spend_usd"] == pytest.approx(expected_usd(_usage(1000, 200)))


def test_public_mode_parameter_only_plan_still_runs(monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "5")
    plan = {**GOLDEN_PLAN, "params": {"lookback": 60}}
    out = nl_interface.handle("q", client=FakeClient(tool_use_response(plan), text_response("ok")))
    assert out["fallback"] is None and out["plan"]["params"]["lookback"] == 60


def test_holdout_dates_are_invalid_with_exactly_one_charge(isolated):
    plan = {**GOLDEN_PLAN, "end": "2024-01-01"}
    client = FakeClient(tool_use_response(plan), text_response("never"))
    out = nl_interface.handle("q", client=client)
    assert out["fallback"] == "invalid"
    assert "holdout" in out["explanation"]
    assert out["metrics"] is None
    assert out["plan"]["end"] == "2024-01-01"
    assert len(client.calls) == 1
    assert today_bucket(isolated)["calls"] == 1
    assert mcp_server._DATASETS == {}  # rejected loads register nothing


def test_unknown_ticker_is_invalid_not_a_crash():
    plan = {**GOLDEN_PLAN, "tickers": ["AAPL", "ZZZZ"]}
    out = nl_interface.handle("q", client=FakeClient(tool_use_response(plan)))
    assert out["fallback"] == "invalid" and "ZZZZ" in out["explanation"]


def test_api_error_on_parse_is_a_fallback_not_a_traceback(isolated):
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.example"))
    out = nl_interface.handle("q", client=FakeClient(err))
    assert out["fallback"] == "api_error"
    assert out["explanation"] == "Connection error."
    assert out["plan"] is None and out["metrics"] is None and out["spend_usd"] == 0.0
    assert today_bucket(isolated) is None


def test_api_error_on_explain_keeps_metrics_and_parse_spend(isolated):
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.example"))
    out = nl_interface.handle("q", client=FakeClient(tool_use_response(GOLDEN_PLAN), err))
    assert out["fallback"] == "api_error"
    assert list(out["metrics"]) == _KEYS
    assert out["spend_usd"] == pytest.approx(expected_usd(_usage(1000, 200)))
    assert today_bucket(isolated)["calls"] == 1


# ---------------------------------------------------------------- explain


def test_explain_empty_text_falls_back_to_template():
    plan = {**GOLDEN_PLAN, "params": {}, "cost_bps": 10.0}
    metrics = {"total_return": 0.25, "sharpe": 1.234, "max_drawdown": -0.1}
    text = nl_interface.explain(plan, metrics, client=FakeClient(text_response("")))
    assert (
        text
        == "Ran momentum on 3 tickers 2015-01-01–2019-12-31: total return 25.0%, Sharpe 1.23, max drawdown -10.0%."
    )


def test_explain_concatenates_text_blocks_and_ignores_others():
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="One. "),
            SimpleNamespace(type="tool_use", name="x", input={}),
            SimpleNamespace(type="text", text="Two."),
        ],
        usage=_usage(10, 5),
        stop_reason="end_turn",
    )
    assert nl_interface.explain(GOLDEN_PLAN, {}, client=FakeClient(resp)) == "One. Two."


def test_every_api_call_is_metered_at_one_site():
    src = inspect.getsource(nl_interface)
    assert src.count("messages.create") == 1
    call_src = inspect.getsource(nl_interface._call)
    assert "messages.create" in call_src and "budget.charge" in call_src


# ---------------------------------------------------------------- live smoke (manual only)

# QUANTFORGE_LIVE_AI is a TEST-ONLY opt-in flag, deliberately absent from .env.example and from
# the docs/components/18-runtime-config.md inventory: it is not a runtime setting the app reads,
# it only tells pytest that a developer is willing to spend real money on this one smoke test.


@pytest.mark.skipif(
    os.environ.get("QUANTFORGE_LIVE_AI") != "1",
    reason="live API smoke; set QUANTFORGE_LIVE_AI=1 to run (spends real money, never in CI)",
)
def test_live_smoke():
    """Real Haiku call end to end. The ledger fixture still isolates spend to tmp_path."""
    out = nl_interface.handle(
        "backtest momentum on AAPL and MSFT from 2015 to 2019 with 10 bps costs",
        client=anthropic.Anthropic(),
    )
    assert out["fallback"] is None, out
    assert list(out["metrics"]) == _KEYS
    assert out["explanation"]
    assert 0 < out["spend_usd"] < 0.01
