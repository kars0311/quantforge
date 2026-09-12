"""Independent verification of ``quantforge.ai.nl_interface`` (component 12) — adversarial cases.

Written separately from ``tests/test_nl_interface.py`` with its OWN fake client, synthetic panel,
and ledger fixture so the two files cannot share a blind spot. Everything is offline:
``ANTHROPIC_API_KEY`` is deleted, ``_get_client`` is patched to explode, and every test passes a
recording fake as ``client=``.

What this file proves beyond the builder's suite:

* the rigor rules survive the NL path — a deliberately *prescient* strategy injected into the
  registry and requested through ``handle`` earns exactly the hand-computed one-day-lagged
  return, and the plan's ``cost_bps`` produces exactly ``turnover * cost`` of daily drag;
* the ledger's today bucket is charged with the model's REAL usage (cache-write and cache-read
  tokens included, at the full input rate), hand-computed from ``PRICES_PER_MTOK``;
* closed gates never reach the client, the budget, or the ledger file — and the gate that is
  closed is the only one consulted;
* the model's output is never trusted: a tool_use block for a different tool, a code-shaped
  strategy name, a missing essential, and non-dict params all end in a clarification (dev) or
  a ``public_mode`` fallback (public), with nothing executed;
* the SF-1 grep: every ``messages.create`` in ``src/quantforge/ai`` is inside ``_call``, read from
  the files on disk (not via ``inspect``, which a module-level alias could fool).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import numpy as np
import pandas as pd
import pytest

from quantforge import interchange, strategies
from quantforge.ai import budget, guardrails, mcp_server, nl_interface
from quantforge.ai.budget import PRICES_PER_MTOK
from quantforge.data import loader
from quantforge.engine.base import Strategy
from quantforge.metrics.performance import _KEYS

AI_PKG = Path(nl_interface.__file__).resolve().parent
TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
THREE = ["AAPL", "MSFT", "NVDA"]
PLAN = {"strategy": "momentum", "start": "2015-01-01", "end": "2019-12-31", "tickers": THREE}
RESULT_KEYS = {"plan", "metrics", "explanation", "spend_usd", "fallback"}


# ---------------------------------------------------------------- fakes


def usage(tokens_in, tokens_out, cache_create=0, cache_read=0):
    return SimpleNamespace(
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cache_creation_input_tokens=cache_create,
        cache_read_input_tokens=cache_read,
    )


def usd_of(*usages) -> float:
    """Hand-computed: every input-side counter at the full input rate, outputs at the output rate."""
    in_rate, out_rate = PRICES_PER_MTOK[nl_interface.MODEL]
    return sum(
        (u.input_tokens + u.cache_creation_input_tokens + u.cache_read_input_tokens) / 1e6 * in_rate
        + u.output_tokens / 1e6 * out_rate
        for u in usages
    )


def tool_resp(plan: dict, *, name="plan_backtest", use=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="toolu_x", name=name, input=plan)],
        usage=use or usage(1000, 200),
        stop_reason="tool_use",
    )


def text_resp(*texts: str, use=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=t) for t in texts],
        usage=use or usage(800, 150),
        stop_reason="end_turn",
    )


class Recorder:
    """A stand-in for ``anthropic.Anthropic``: records requests, replays canned responses/errors."""

    def __init__(self, *responses):
        self.requests: list[dict] = []
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request):
        self.requests.append(request)
        assert self._responses, "more model calls than canned responses"
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def client_for(plan: dict = PLAN, explanation: str = "Fine.") -> Recorder:
    return Recorder(tool_resp(plan), text_resp(explanation))


# ---------------------------------------------------------------- fixtures


def _make_wide(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2026-06-30", tz="UTC")
    rets = rng.normal(0.0003, 0.012, size=(len(dates), len(TICKERS)))
    return pd.DataFrame(50.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=TICKERS)


_WIDE = _make_wide()
_LONG = interchange.to_long(_WIDE, "prices")


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    """Offline, dev-mode, temp ledger; any path to a real SDK client explodes."""
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _LONG)
    mcp_server.reset_registry()
    path = tmp_path / "ai_ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(path))
    for var in (
        "PUBLIC_MODE",
        "AI_DISABLED",
        "AI_BUDGET_USD_DAILY",
        "AI_BUDGET_USD_TOTAL",
        "AI_RATE_LIMIT_PER_HOUR",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(nl_interface, "_client", None)

    def boom():
        raise AssertionError("a real anthropic client was requested")

    monkeypatch.setattr(nl_interface, "_get_client", boom)
    yield path
    mcp_server.reset_registry()


def today(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())["days"].get(budget._utc_today())


# ---------------------------------------------------------------- verifier checks 1, 2, 8


def test_model_id_is_haiku_and_the_only_claude_string_on_disk():
    assert nl_interface.MODEL == "claude-haiku-4-5"
    src = Path(nl_interface.__file__).read_text()
    assert src.count("claude-") == src.count("claude-haiku-4-5") >= 1
    assert nl_interface.MODEL in PRICES_PER_MTOK  # the budget can price it


def test_handle_signature_has_no_public_mode_kwarg():
    sig = inspect.signature(nl_interface.handle)
    assert list(sig.parameters) == ["query", "session_key", "client"]
    assert sig.parameters["session_key"].default == "anon"
    assert sig.parameters["client"].default is None
    for name in ("session_key", "client"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_every_messages_create_in_the_ai_package_is_inside_call():
    hits = {
        p.name: [ln for ln in p.read_text().splitlines() if "messages.create" in ln]
        for p in AI_PKG.glob("*.py")
    }
    hits = {k: v for k, v in hits.items() if v}
    assert list(hits) == ["nl_interface.py"], hits
    assert len(hits["nl_interface.py"]) == 1
    call_src = inspect.getsource(nl_interface._call)
    assert "messages.create" in call_src
    assert call_src.index("messages.create") < call_src.index("budget.charge(")


# ---------------------------------------------------------------- check 3: golden path + ledger


def test_golden_path_shape_metrics_and_hand_computed_ledger(ledger):
    parse_use = usage(1234, 321, cache_create=400, cache_read=300)
    explain_use = usage(777, 99)
    client = Recorder(
        tool_resp(PLAN, use=parse_use), text_resp("Momentum ", "did fine.", use=explain_use)
    )

    out = nl_interface.handle("backtest momentum 2015-2019", session_key="s1", client=client)

    assert set(out) == RESULT_KEYS
    assert out["fallback"] is None
    assert list(out["metrics"]) == _KEYS
    assert out["explanation"] == "Momentum did fine."
    assert out["plan"] == {
        "strategy": "momentum",
        "params": {"lookback": 126, "top_n": 0},
        "tickers": THREE,
        "start": "2015-01-01",
        "end": "2019-12-31",
        "cost_bps": 10.0,
    }
    expected = usd_of(parse_use, explain_use)
    assert out["spend_usd"] == pytest.approx(expected)
    bucket = today(ledger)
    assert bucket["calls"] == 2
    assert bucket["usd"] == pytest.approx(expected)
    assert bucket["tokens_in"] == 1234 + 400 + 300 + 777
    assert bucket["tokens_out"] == 321 + 99
    assert len(client.requests) == 2


def test_golden_metrics_equal_a_direct_engine_run(ledger):
    """The NL path is the SAME pipeline as the tools: same window, same params, same costs."""
    out = nl_interface.handle("q", session_key="s2", client=client_for())
    ds = mcp_server.load_data("2015-01-01", "2019-12-31", THREE)["dataset_id"]
    direct = mcp_server.run_backtest(ds, "momentum", {}, "python", 10.0)["metrics"]
    assert out["metrics"] == direct


# ---------------------------------------------------------------- rigor through the NL path


class _Prescient(Strategy):
    """Cheats on purpose: at close t, 100% long the asset with the largest return ON day t."""

    name = "prescient"

    def generate_signals(self, prices: pd.DataFrame, params: dict) -> pd.DataFrame:
        rets = prices.pct_change().fillna(0.0)
        positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for date, ticker in rets.idxmax(axis=1).items():
            positions.loc[date, ticker] = 1.0
        return positions


def test_prescient_strategy_via_handle_earns_only_the_lagged_return(monkeypatch):
    monkeypatch.setitem(strategies.STRATEGIES, "prescient", _Prescient)
    monkeypatch.setitem(strategies.PARAM_WHITELIST, "prescient", {})
    plan = {**PLAN, "strategy": "prescient", "cost_bps": 0}

    out = nl_interface.handle("q", session_key="s3", client=client_for(plan))
    assert out["fallback"] is None, out

    wide = _WIDE.loc["2015-01-01":"2019-12-31", THREE]
    asset_rets = wide.pct_change().fillna(0.0)
    positions = _Prescient().generate_signals(wide, {})
    lagged_total = float(
        (1.0 + (positions.shift(1).fillna(0.0) * asset_rets).sum(axis=1)).prod() - 1.0
    )
    cheat_total = float((1.0 + (positions * asset_rets).sum(axis=1)).prod() - 1.0)

    assert out["metrics"]["total_return"] == pytest.approx(lagged_total, rel=1e-9)
    assert cheat_total > 1e3  # what a same-day peek would have booked
    assert out["metrics"]["total_return"] < 10.0
    assert abs(out["metrics"]["sharpe"]) < 2.0


def test_plan_cost_bps_reaches_the_engine_as_turnover_drag():
    free = nl_interface.handle("q", session_key="s4a", client=client_for({**PLAN, "cost_bps": 0}))
    dear = nl_interface.handle("q", session_key="s4b", client=client_for({**PLAN, "cost_bps": 100}))
    assert free["plan"]["cost_bps"] == 0.0 and dear["plan"]["cost_bps"] == 100.0
    assert free["metrics"]["total_return"] > dear["metrics"]["total_return"]

    wide = _WIDE.loc["2015-01-01":"2019-12-31", THREE]
    held = (
        strategies.STRATEGIES["momentum"]().generate_signals(wide, free["plan"]["params"]).shift(1)
    )
    turnover = held.fillna(0.0).diff().abs().sum(axis=1).fillna(0.0)
    asset_rets = wide.pct_change().fillna(0.0)
    gross = (held.fillna(0.0) * asset_rets).sum(axis=1)
    assert turnover.sum() > 0
    expected_free = float((1.0 + gross).prod() - 1.0)
    expected_dear = float((1.0 + gross - turnover * 0.01).prod() - 1.0)
    assert free["metrics"]["total_return"] == pytest.approx(expected_free, rel=1e-9)
    assert dear["metrics"]["total_return"] == pytest.approx(expected_dear, rel=1e-9)


def test_omitted_cost_bps_defaults_to_ten_and_matches_explicit_ten():
    implicit = nl_interface.handle("q", session_key="s5a", client=client_for(PLAN))
    explicit = nl_interface.handle(
        "q", session_key="s5b", client=client_for({**PLAN, "cost_bps": 10})
    )
    assert implicit["plan"]["cost_bps"] == 10.0
    assert implicit["metrics"] == explicit["metrics"]


# ---------------------------------------------------------------- checks 4-6: gates


def test_rate_gate_closed_touches_nothing(ledger, monkeypatch):
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "0")
    allow_calls: list[float] = []
    monkeypatch.setattr(budget, "allow", lambda est: allow_calls.append(est) or True)
    client = client_for()

    out = nl_interface.handle("q", session_key="s6", client=client)

    assert out == {
        "plan": None,
        "metrics": None,
        "explanation": out["explanation"],
        "spend_usd": 0.0,
        "fallback": "rate",
    }
    assert out["explanation"].strip() and "\n" not in out["explanation"].strip()
    assert client.requests == []
    assert allow_calls == []  # the later gate is not even consulted
    assert not ledger.exists()


def test_kill_switch_closes_budget_gate_with_no_call(ledger, monkeypatch):
    monkeypatch.setenv("AI_DISABLED", "on")
    client = client_for()
    out = nl_interface.handle("q", session_key="s7", client=client)
    assert out["fallback"] == "budget"
    assert out["plan"] is None and out["metrics"] is None and out["spend_usd"] == 0.0
    assert client.requests == []
    assert today(ledger) is None


def test_budget_gate_is_asked_for_two_calls_and_a_one_call_budget_is_refused(ledger, monkeypatch):
    per_call = budget.estimate(nl_interface.MODEL, 2500, 600)
    # A cap that fits one call but not two must close the gate BEFORE the first call.
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", f"{per_call * 1.5:.8f}")
    seen: list[float] = []
    real_allow = budget.allow
    monkeypatch.setattr(budget, "allow", lambda est: seen.append(est) or real_allow(est))
    client = client_for()

    out = nl_interface.handle("q", session_key="s8", client=client)

    assert seen == [pytest.approx(2 * per_call)]
    assert out["fallback"] == "budget" and client.requests == []


def test_public_mode_code_param_is_rejected_after_exactly_one_charge(ledger, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "5")
    monkeypatch.setattr(mcp_server, "load_data", lambda *a, **k: pytest.fail("tools ran"))
    client = client_for({**PLAN, "params": {"lookback": 60, "code": "x"}})

    out = nl_interface.handle("q", session_key="s9", client=client)

    assert out["fallback"] == "public_mode"
    assert out["plan"] is None and out["metrics"] is None
    assert len(client.requests) == 1
    assert today(ledger)["calls"] == 1
    assert out["spend_usd"] == pytest.approx(usd_of(usage(1000, 200)))
    assert "Traceback" not in out["explanation"]


@pytest.mark.parametrize(
    "bad_plan",
    [
        {**PLAN, "strategy": "__import__('os').system('id')"},
        {**PLAN, "strategy": "momentum.py"},
        {**PLAN, "params": {"lookback": [60]}},
        {**PLAN, "params": [{"lookback": 60}]},
    ],
)
def test_public_mode_rejects_code_shaped_plans_without_running_tools(monkeypatch, bad_plan):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "5")
    monkeypatch.setattr(mcp_server, "load_data", lambda *a, **k: pytest.fail("tools ran"))
    out = nl_interface.handle("q", session_key="s10", client=client_for(bad_plan))
    assert out["fallback"] == "public_mode"


def test_holdout_end_date_is_invalid_with_one_charge_and_no_registry_entry(ledger):
    client = client_for({**PLAN, "end": "2024-01-01"})
    out = nl_interface.handle("q", session_key="s11", client=client)
    assert out["fallback"] == "invalid"
    assert "holdout" in out["explanation"]
    assert out["metrics"] is None
    assert out["plan"]["end"] == "2024-01-01"  # the rejected plan is reported, not hidden
    assert len(client.requests) == 1  # no explain call on a failed backtest
    assert today(ledger)["calls"] == 1
    assert out["spend_usd"] == pytest.approx(usd_of(usage(1000, 200)))
    assert mcp_server._DATASETS == {} and mcp_server._RESULTS == {}


def test_holdout_start_date_and_pre_data_start_are_both_invalid():
    hi = loader.get_split_bounds()["validation"][1]
    lo = loader.get_split_bounds()["train"][0]
    late = nl_interface.handle(
        "q",
        session_key="s12a",
        client=client_for({**PLAN, "start": "2023-06-01", "end": "2023-12-31"}),
    )
    early = nl_interface.handle(
        "q", session_key="s12b", client=client_for({**PLAN, "start": "2009-12-31"})
    )
    assert late["fallback"] == "invalid" and hi in late["explanation"]
    assert early["fallback"] == "invalid" and lo in early["explanation"]


# ---------------------------------------------------------------- check 7: request shape


def test_parse_request_is_forced_cached_and_carries_the_query_verbatim():
    client = client_for()
    query = "run mean reversion on JPM in 2018"
    nl_interface.handle(query, session_key="s13", client=client)

    parse_req, explain_req = client.requests
    assert parse_req["model"] == nl_interface.MODEL
    assert parse_req["max_tokens"] == 512
    assert parse_req["tool_choice"] == {"type": "tool", "name": "plan_backtest"}
    assert parse_req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert parse_req["system"][0]["text"] == nl_interface.SYSTEM_PARSE
    assert parse_req["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert parse_req["tools"][0]["name"] == "plan_backtest"
    assert parse_req["messages"] == [{"role": "user", "content": query}]

    assert explain_req["model"] == nl_interface.MODEL
    assert explain_req["max_tokens"] == 400
    assert "tools" not in explain_req and "tool_choice" not in explain_req
    assert explain_req["system"][0]["cache_control"] == {"type": "ephemeral"}
    payload = json.loads(explain_req["messages"][0]["content"])
    assert set(payload) == {"plan", "metrics"}
    assert list(payload["metrics"]) == sorted(_KEYS)  # sort_keys=True: byte-stable payload


def test_call_never_mutates_the_module_constants():
    before = json.dumps(nl_interface.PLAN_TOOL, sort_keys=True)
    system_before = nl_interface.SYSTEM_PARSE
    nl_interface.handle("q", session_key="s14", client=client_for())
    assert json.dumps(nl_interface.PLAN_TOOL, sort_keys=True) == before
    assert "cache_control" not in nl_interface.PLAN_TOOL
    assert nl_interface.SYSTEM_PARSE is system_before


def test_plan_tool_schema_is_closed_and_mirrors_the_whitelist():
    schema = nl_interface.PLAN_TOOL["input_schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["strategy"]["enum"] == sorted(strategies.STRATEGIES)
    assert schema["properties"]["tickers"]["items"]["enum"] == list(loader.UNIVERSE)
    params = schema["properties"]["params"]
    assert params["additionalProperties"] is False
    every_param = set().union(*(set(v) for v in strategies.PARAM_WHITELIST.values()))
    assert set(params["properties"]) == every_param
    assert schema["properties"]["cost_bps"]["minimum"] == 0
    assert schema["properties"]["cost_bps"]["maximum"] == 100
    assert "ONLY" in schema["properties"]["clarify"]["description"]
    for prompt in (nl_interface.SYSTEM_PARSE, nl_interface.SYSTEM_EXPLAIN):
        for s in strategies.STRATEGIES:
            assert s in prompt
        assert loader.get_split_bounds()["train"][0] in prompt
        assert loader.get_split_bounds()["validation"][1] in prompt
        assert (
            "never invent" in nl_interface.SYSTEM_PARSE and "clarify" in nl_interface.SYSTEM_PARSE
        )


# ---------------------------------------------------------------- metering


def test_charge_gets_real_usage_and_none_fields_count_as_zero(monkeypatch):
    charged: list[dict] = []
    real_charge = budget.charge

    def spy(usd, **kw):
        charged.append({"usd": usd, **kw})
        return real_charge(usd, **kw)

    monkeypatch.setattr(budget, "charge", spy)
    weird = SimpleNamespace(
        input_tokens=500, output_tokens=40, cache_creation_input_tokens=None
    )  # cache_read attr missing entirely
    client = Recorder(tool_resp(PLAN, use=usage(10, 20, 30, 40)), text_resp("x", use=weird))

    out = nl_interface.handle("q", session_key="s15", client=client)

    assert out["fallback"] is None
    assert charged == [
        {
            "usd": usd_of(usage(10, 20, 30, 40)),
            "model": nl_interface.MODEL,
            "tokens_in": 80,
            "tokens_out": 20,
        },
        {
            "usd": usd_of(usage(500, 40)),
            "model": nl_interface.MODEL,
            "tokens_in": 500,
            "tokens_out": 40,
        },
    ]
    assert out["spend_usd"] == pytest.approx(sum(c["usd"] for c in charged))


def test_api_error_on_parse_charges_nothing_and_is_not_a_traceback(ledger):
    err = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    out = nl_interface.handle("q", session_key="s16", client=Recorder(err))
    assert out["fallback"] == "api_error"
    assert out["plan"] is None and out["metrics"] is None and out["spend_usd"] == 0.0
    assert "Traceback" not in out["explanation"] and out["explanation"]
    assert today(ledger) is None


def test_api_error_on_explain_keeps_parse_spend_only(ledger):
    err = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    out = nl_interface.handle("q", session_key="s17", client=Recorder(tool_resp(PLAN), err))
    assert out["fallback"] == "api_error"
    assert list(out["metrics"]) == _KEYS
    assert out["spend_usd"] == pytest.approx(usd_of(usage(1000, 200)))
    assert today(ledger)["calls"] == 1


# ---------------------------------------------------------------- the model's output is untrusted


def test_tool_use_for_a_different_tool_is_a_clarification_not_an_execution(monkeypatch):
    monkeypatch.setattr(mcp_server, "load_data", lambda *a, **k: pytest.fail("tools ran"))
    client = Recorder(tool_resp(PLAN, name="run_backtest"))
    out = nl_interface.handle("q", session_key="s18", client=client)
    assert out["fallback"] is None
    assert set(out["plan"]) == {"clarify"} and out["explanation"] == out["plan"]["clarify"]
    assert out["metrics"] is None
    assert len(client.requests) == 1


@pytest.mark.parametrize("missing", ["strategy", "start", "end"])
def test_missing_essential_is_asked_about_never_guessed(monkeypatch, missing):
    monkeypatch.setattr(mcp_server, "load_data", lambda *a, **k: pytest.fail("tools ran"))
    plan = {k: v for k, v in PLAN.items() if k != missing}
    out = nl_interface.handle("q", session_key="s19", client=Recorder(tool_resp(plan)))
    assert set(out["plan"]) == {"clarify"}
    assert out["fallback"] is None and out["metrics"] is None


def test_clarify_from_the_model_wins_even_when_the_plan_looks_complete(monkeypatch):
    monkeypatch.setattr(mcp_server, "load_data", lambda *a, **k: pytest.fail("tools ran"))
    plan = {**PLAN, "clarify": "Which years?"}
    out = nl_interface.handle("q", session_key="s20", client=Recorder(tool_resp(plan)))
    assert out["plan"] == {"clarify": "Which years?"} and out["explanation"] == "Which years?"


@pytest.mark.parametrize(
    "plan",
    [
        {**PLAN, "strategy": "__import__('os').system('id')"},
        {**PLAN, "strategy": "buy_and_hold"},
        {**PLAN, "params": {"lookback": 60, "code": "x"}},
        {**PLAN, "params": {"lookback": 10}},  # momentum's floor is 20
        {**PLAN, "params": [{"lookback": 60}]},  # non-dict params: TypeError trap (fixed)
    ],
)
def test_dev_mode_bad_plans_become_clarifications_and_never_run(monkeypatch, plan):
    monkeypatch.setattr(mcp_server, "load_data", lambda *a, **k: pytest.fail("tools ran"))
    out = nl_interface.handle("q", session_key="s21", client=Recorder(tool_resp(plan)))
    assert out["fallback"] is None
    assert set(out["plan"]) == {"clarify"}
    assert out["metrics"] is None


@pytest.mark.parametrize(
    "plan, needle",
    [
        ({**PLAN, "tickers": ["AAPL", "SPY"]}, "SPY"),
        ({**PLAN, "tickers": ["../etc/passwd"]}, "not in the fixed universe"),
        ({**PLAN, "start": "2016-01-01", "end": "2015-01-01"}, "must not be after"),
        ({**PLAN, "start": "2015-02-30"}, "not a valid calendar date"),
        ({**PLAN, "cost_bps": 500}, "cost_bps"),
        ({**PLAN, "cost_bps": "10"}, "cost_bps"),
    ],
)
def test_tool_rejections_are_the_invalid_fallback_with_the_tool_message(plan, needle):
    client = Recorder(tool_resp(plan), text_resp("never"))
    out = nl_interface.handle("q", session_key="s22", client=client)
    assert out["fallback"] == "invalid"
    assert needle in out["explanation"]
    assert len(client.requests) == 1


def test_parse_returns_exactly_the_pipeline_shape_with_defaults():
    plan = nl_interface.parse(
        "q",
        client=Recorder(
            tool_resp({"strategy": "mean_reversion", "start": "2018-01-01", "end": "2018-12-31"})
        ),
    )
    assert set(plan) == {"strategy", "params", "tickers", "start", "end", "cost_bps"}
    assert plan["tickers"] == list(loader.UNIVERSE)
    assert plan["cost_bps"] == 10.0
    assert plan["params"] == {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "mode": "long_flat"}


@pytest.mark.parametrize("bad", ["", "   \n", "x" * 2001, None, 7, b"bytes"])
def test_parse_rejects_bad_queries_before_any_call(ledger, bad):
    client = Recorder()
    with pytest.raises(ValueError):
        nl_interface.parse(bad, client=client)
    with pytest.raises(ValueError):
        nl_interface.parse_with_spend(bad, client=client)
    assert client.requests == [] and not ledger.exists()


def test_parse_accepts_exactly_2000_chars():
    plan = nl_interface.parse("x" * 2000, client=Recorder(tool_resp(PLAN)))
    assert plan["strategy"] == "momentum"


# ---------------------------------------------------------------- explain


def test_explain_empty_or_non_text_reply_uses_the_exact_template():
    plan = {**PLAN, "params": {}, "cost_bps": 10.0}
    metrics = {"total_return": 0.4567, "sharpe": 1.234, "max_drawdown": -0.2101}
    expected = "Ran momentum on 3 tickers 2015-01-01–2019-12-31: total return 45.7%, Sharpe 1.23, max drawdown -21.0%."
    assert nl_interface.explain(plan, metrics, client=Recorder(text_resp(""))) == expected
    assert nl_interface.explain(plan, metrics, client=Recorder(text_resp("  ", "\n"))) == expected
    only_tool = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="x", input={})],
        usage=usage(1, 1),
        stop_reason="end_turn",
    )
    assert nl_interface.explain(plan, metrics, client=Recorder(only_tool)) == expected


def test_explain_with_spend_returns_text_and_its_charge(ledger):
    text, usd = nl_interface.explain_with_spend(
        PLAN, {"sharpe": 1.0}, client=Recorder(text_resp("a", "b", use=usage(50, 5)))
    )
    assert text == "ab"
    assert usd == pytest.approx(usd_of(usage(50, 5)))
    assert today(ledger)["calls"] == 1


# ---------------------------------------------------------------- gate order


def test_gate_order_and_each_gate_consulted_once(monkeypatch):
    trace: list[str] = []
    monkeypatch.setattr(guardrails, "rate_limit", lambda key: trace.append(f"rate:{key}") or True)
    monkeypatch.setattr(budget, "allow", lambda est: trace.append("budget") or True)
    real_gate = guardrails.assert_no_codegen
    monkeypatch.setattr(
        guardrails, "assert_no_codegen", lambda a: trace.append("codegen") or real_gate(a)
    )
    real_load = mcp_server.load_data
    monkeypatch.setattr(
        mcp_server, "load_data", lambda *a, **k: trace.append("load") or real_load(*a, **k)
    )
    real_call = nl_interface._call

    def call_spy(client, **kw):
        trace.append("model:parse" if kw.get("tools") else "model:explain")
        return real_call(client, **kw)

    monkeypatch.setattr(nl_interface, "_call", call_spy)

    out = nl_interface.handle("q", session_key="sess-42", client=client_for())
    assert out["fallback"] is None
    assert trace[:5] == ["rate:sess-42", "budget", "model:parse", "codegen", "load"]
    assert trace[-1] == "model:explain"
    assert trace.count("rate:sess-42") == 1 and trace.count("budget") == 1
