"""Safety: with PUBLIC_MODE=on, no LLM-generated code can execute; only vetted strategies + params.

Closes the worst public-MCP-hook risk (arbitrary code execution on a public server) — SF-3. The
gate under test is ``guardrails.assert_no_codegen``: in public mode the ONLY accepted action is
``{"strategy": <vetted>, "params": <whitelisted scalars>}``; every other shape raises
``PublicModeViolation``. The second clause proves the money gate (``budget.allow``) still closes
past the cap in public mode, so "parameter-only" and "capped spend" hold together.

Third clause (week 8, the agent): with ``PUBLIC_MODE=on`` a canned ``propose_experiment`` whose
params smuggle ``code`` or ``source`` is recorded in ``history`` with an error naming public
mode, mints no result handle, and never reaches the engine — the spy engine registered under
``mcp_server.ENGINES["python"]`` sees zero calls for those turns — while a following
parameter-only proposal runs normally. And the agent module's source contains no code-execution
token at all (``exec(``, ``eval(``, ``compile(``, ``importlib``, ``subprocess``, ``open(``): there
is no path by which generated text could become running code (SF-3).
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange

# ``PublicModeViolation`` is always referenced as ``guardrails.PublicModeViolation`` (looked up
# at call time), never imported by name: a name bound at collection time would go stale if any
# test ever ``importlib.reload``s the module, and ``pytest.raises`` could no longer match it.
from quantforge.ai import agent, budget, guardrails, mcp_server
from quantforge.data import loader
from quantforge.engine.python_engine import PythonEngine


@pytest.fixture
def public(monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")


GOOD = {"strategy": "momentum", "params": {"lookback": 126}}

REJECTED = [
    pytest.param({"code": "import os; os.system('rm -rf /')"}, id="code-only"),
    pytest.param({"strategy": "momentum", "params": {}, "source": "print(1)"}, id="extra-source"),
    pytest.param({"strategy": "momentum", "params": {}, "python": "1+1"}, id="extra-python"),
    pytest.param({"strategy": "momentum", "params": {}, "eval": "1+1"}, id="extra-eval"),
    pytest.param({"strategy": "momentum", "params": {}, "exec": "1+1"}, id="extra-exec"),
    pytest.param({"strategy": "momentum", "params": {}, "file": "/etc/passwd"}, id="extra-file"),
    pytest.param({"strategy": "momentum"}, id="missing-params"),
    pytest.param({"params": {}}, id="missing-strategy"),
    pytest.param({}, id="empty"),
    pytest.param({"strategy": "pairs", "params": {}}, id="unvetted-strategy"),
    pytest.param({"strategy": "momentum", "params": {"lookback": 999}}, id="out-of-range"),
    pytest.param({"strategy": "momentum", "params": {"lookbak": 100}}, id="unknown-param"),
    pytest.param({"strategy": "momentum", "params": {"code": "x"}}, id="code-as-param"),
    pytest.param({"strategy": "momentum", "params": {"lookback": {"n": 126}}}, id="nested-dict"),
    pytest.param({"strategy": "momentum", "params": {"lookback": [126]}}, id="nested-list"),
    pytest.param({"strategy": "momentum", "params": {"lookback": lambda: 126}}, id="callable"),
    pytest.param({"strategy": "momentum", "params": {"lookback": b"126"}}, id="bytes"),
    pytest.param({"strategy": "momentum", "params": "lookback=126"}, id="params-not-dict"),
    pytest.param({"strategy": ["momentum"], "params": {}}, id="strategy-not-str"),
    pytest.param("run momentum", id="not-a-dict"),
]


def test_public_mode_accepts_parameter_only_actions(public):
    assert guardrails.assert_no_codegen(GOOD) is None
    assert guardrails.assert_no_codegen({"strategy": "momentum", "params": {}}) is None
    assert (
        guardrails.assert_no_codegen({"strategy": "momentum", "params": {"lookback": 50}}) is None
    )
    assert (
        guardrails.assert_no_codegen(
            {
                "strategy": "mean_reversion",
                "params": {"lookback": 10, "entry_z": 2.5, "mode": "long_short"},
            }
        )
        is None
    )


@pytest.mark.parametrize("action", REJECTED)
def test_public_mode_rejects_codegen_and_unvetted_params(public, action):
    with pytest.raises(guardrails.PublicModeViolation):
        guardrails.assert_no_codegen(action)


def test_public_mode_violation_is_a_value_error(public):
    # Callers that already catch ValueError from validate_params keep working.
    with pytest.raises(ValueError):
        guardrails.assert_no_codegen({"code": "x"})


def test_rejection_message_names_the_reason(public):
    with pytest.raises(guardrails.PublicModeViolation, match="exec"):
        guardrails.assert_no_codegen({"strategy": "momentum", "params": {}, "exec": "x"})
    with pytest.raises(guardrails.PublicModeViolation, match="pairs"):
        guardrails.assert_no_codegen({"strategy": "pairs", "params": {}})
    with pytest.raises(guardrails.PublicModeViolation, match="lookback"):
        guardrails.assert_no_codegen({"strategy": "momentum", "params": {"lookback": 999}})


def test_same_bad_inputs_do_not_raise_outside_public_mode(monkeypatch):
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    for action in [
        {"strategy": "momentum", "params": {"code": "x"}},
        {"strategy": "momentum", "params": {}, "exec": "x"},
        {"code": "x"},
    ]:
        assert guardrails.assert_no_codegen(action) is None
    assert guardrails.public_mode() is False


# ---------------------------------------------------------------- second clause: budget gate


def test_budget_blocks_calls_past_the_cap_in_public_mode(public, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_LEDGER_PATH", str(tmp_path / "ai_ledger.json"))
    monkeypatch.delenv("AI_DISABLED", raising=False)
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1.0")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "10.0")

    assert budget.allow(0.5) is True
    budget.charge(0.9, model="claude-haiku-4-5", tokens_in=100, tokens_out=100)
    assert budget.allow(0.1) is True  # exactly on the cap is allowed
    assert budget.allow(0.2) is False  # past the cap is blocked
    assert budget.remaining()["daily"] == pytest.approx(0.1)


def test_budget_denies_everything_when_caps_unset_in_public_mode(public, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_LEDGER_PATH", str(tmp_path / "ai_ledger.json"))
    monkeypatch.delenv("AI_DISABLED", raising=False)
    monkeypatch.delenv("AI_BUDGET_USD_DAILY", raising=False)
    monkeypatch.delenv("AI_BUDGET_USD_TOTAL", raising=False)
    assert budget.allow(0.0) is False


# ---------------------------------------------------------------- third clause: the agent (week 8)
#
# A minimal offline harness (duplicated from ``tests/test_agent.py`` so this proof stands alone):
# a ``FakeClient`` pops canned Sonnet turns, the MCP tools read a synthetic panel, the ledger
# lives in ``tmp_path``, and any path to a real SDK client explodes.

AGENT_TICKERS = list(loader.UNIVERSE[:4])
GOAL = "Find a robust momentum lookback."


def _agent_panel(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2026-06-30", tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), len(AGENT_TICKERS)))
    wide = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=AGENT_TICKERS)
    return interchange.to_long(wide, "prices")


_SYNTHETIC = _agent_panel()


def _tool_turn(name: str, data: dict, block_id: str):
    block = SimpleNamespace(type="tool_use", id=block_id, name=name, input=data)
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return SimpleNamespace(content=[block], usage=usage, stop_reason="tool_use")


def propose(strategy: str, params, tool_id: str, rationale: str = "x"):
    data = {"strategy": strategy, "params": params, "rationale": rationale}
    return _tool_turn("propose_experiment", data, tool_id)


def done(tool_id: str = "toolu_done"):
    return _tool_turn("declare_done", {"reason": "converged"}, tool_id)


class FakeClient:
    def __init__(self, *responses):
        self.calls: list[dict] = []
        self._queue = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._queue:
            raise AssertionError("FakeClient received more requests than canned responses")
        return self._queue.pop(0)


class SpyEngine(PythonEngine):
    """The real engine, recording the ``params`` of every ``run_backtest`` it is asked to run."""

    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def run_backtest(self, prices, positions, params=None):
        self.calls.append(dict(params or {}))
        return super().run_backtest(prices, positions, params)


@pytest.fixture
def public_agent(public, tmp_path, monkeypatch):
    """PUBLIC_MODE=on plus the offline agent harness; yields the spy engine.

    Caps are set explicitly: in public mode ``budget.allow`` denies everything when they are
    unset (proved above), and this clause is about the code gate, not the money gate.
    """
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _SYNTHETIC)
    mcp_server.reset_registry()
    monkeypatch.setenv("AI_LEDGER_PATH", str(tmp_path / "ai_ledger.json"))
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1.0")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "5.0")
    for var in ("AI_DISABLED", "AI_RATE_LIMIT_PER_HOUR", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(agent, "_client", None)
    monkeypatch.setattr(
        agent, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("real client"))
    )
    engine = SpyEngine()
    monkeypatch.setitem(mcp_server.ENGINES, "python", engine)
    yield engine
    mcp_server.reset_registry()


def _spy_run_backtest(monkeypatch) -> list[tuple]:
    """Record every call that reaches the MCP ``run_backtest`` tool (the original still runs)."""
    calls: list[tuple] = []
    original = mcp_server.run_backtest

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(mcp_server, "run_backtest", spy)
    return calls


POISONED = [
    pytest.param({"code": '__import__("os")'}, id="code-as-param"),
    pytest.param({"lookback": 60, "source": "print(1)"}, id="source-as-param"),
]


@pytest.mark.parametrize("params", POISONED)
def test_agent_public_mode_poisoned_proposal_runs_nothing(public_agent, monkeypatch, params):
    engine = public_agent
    tool_calls = _spy_run_backtest(monkeypatch)
    client = FakeClient(propose("momentum", params, "toolu_bad"), done())
    results_before = dict(mcp_server._RESULTS)

    # Two turns: the poisoned proposal, then a ``declare_done`` the loop must refuse (nothing
    # has succeeded), which exhausts the cap.
    result = agent.run_research(GOAL, max_iters=2, client=client)

    # The turn is real, paid-for history: recorded, with the refusal naming public mode.
    (record,) = result["history"]
    assert record["error"] is not None
    assert "public mode" in record["error"].lower()
    assert record["params"] == params  # the raw proposal, so the timeline shows what was tried
    assert record["train_metrics"] is None and record["val_metrics"] is None
    # ...and it executed nothing: no result handle, no tool call, no engine call.
    assert mcp_server._RESULTS == results_before == {}
    assert tool_calls == []
    assert engine.calls == []
    # With nothing successful there is no winner and no holdout look either.
    assert result["best"] is None and result["holdout_metrics"] is None
    # ``declare_done`` is refused until something succeeded, so the run hit the cap.
    assert result["stopped_because"] == "max_iters"
    assert len(client.calls) == 2


def test_agent_public_mode_poisoned_turns_then_a_parameter_only_proposal_runs(
    public_agent, monkeypatch
):
    engine = public_agent
    tool_calls = _spy_run_backtest(monkeypatch)
    client = FakeClient(
        propose("momentum", {"code": '__import__("os")'}, "toolu_code"),
        propose("momentum", {"lookback": 60, "source": "print(1)"}, "toolu_source"),
        propose("momentum", {"lookback": 60}, "toolu_ok"),
        done(),
    )

    result = agent.run_research(GOAL, max_iters=5, client=client)

    assert result["stopped_because"] == "converged"
    code_rec, source_rec, ok_rec = result["history"]
    for rec in (code_rec, source_rec):
        assert rec["error"] is not None and "public mode" in rec["error"].lower()
        assert rec["train_metrics"] is None and rec["val_metrics"] is None
    assert "code" in code_rec["error"] and "source" in source_rec["error"]

    # The poisoned turns reached neither the tool nor the engine; only the honest one did,
    # twice (train, then validation) — and the runner's one holdout score is a third engine
    # call with the same validated params and no smuggled key.
    assert ok_rec["error"] is None and ok_rec["params"] == {"lookback": 60, "top_n": 0}
    assert [kw.get("params") or a[2] for a, kw in tool_calls] == [ok_rec["params"]] * 2
    assert len(engine.calls) == 3
    for params in engine.calls:
        assert "code" not in params and "source" not in params
        assert params["lookback"] == 60 and params["strategy"] == "momentum"
    # Exactly the honest proposal's two backtests were registered (the holdout registers none).
    assert len(mcp_server._RESULTS) == 2
    assert result["best"]["params"] == ok_rec["params"]
    assert result["holdout_metrics"] is not None

    # The model was told, in the tool_result for each poisoned turn, exactly why it was refused.
    for i, rec in ((1, code_rec), (2, source_rec)):
        blocks = client.calls[i]["messages"][-1]["content"]
        tool_result = next(b for b in blocks if b.get("type") == "tool_result")
        assert tool_result["is_error"] is True and tool_result["content"] == rec["error"]
    json.dumps(result)


def test_agent_source_has_no_code_execution_tokens():
    """SF-3 at the source level: the agent module cannot turn generated text into code."""
    src = inspect.getsource(agent)
    for token in ("exec(", "eval(", "compile(", "importlib", "subprocess", "open("):
        assert token not in src, token
