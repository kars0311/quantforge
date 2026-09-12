"""Safety: with PUBLIC_MODE=on, no LLM-generated code can execute; only vetted strategies + params.

Closes the worst public-MCP-hook risk (arbitrary code execution on a public server) — SF-3. The
gate under test is ``guardrails.assert_no_codegen``: in public mode the ONLY accepted action is
``{"strategy": <vetted>, "params": <whitelisted scalars>}``; every other shape raises
``PublicModeViolation``. The second clause proves the money gate (``budget.allow``) still closes
past the cap in public mode, so "parameter-only" and "capped spend" hold together.
"""

from __future__ import annotations

import pytest

# ``PublicModeViolation`` is always referenced as ``guardrails.PublicModeViolation`` (looked up
# at call time), never imported by name: a name bound at collection time would go stale if any
# test ever ``importlib.reload``s the module, and ``pytest.raises`` could no longer match it.
from quantforge.ai import budget, guardrails


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
