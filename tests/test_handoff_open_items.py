"""Proving tests for the handoff.md open-items fix run (2026-09-12; offline, no network).

Every builder test for that run lives in this ONE file so no existing per-suite collection-count
pin (e.g. test_week8_closeout_verifier's `test_agent.py`=98 / holdout=15 / public-mode=30) moves.

Milestone 1 — config + hygiene alignment:
- .env.example's two budget-cap example values equal budget.py's dev defaults. Before the fix the
  file said 5 / 25 while its own comment block, docs/components/18-runtime-config.md and the code
  all said 2 / 10 — a reader copying the example got caps the docs never mentioned.
- requirements.txt lists `jsonschema` explicitly: tests/test_mcp_server.py imports it directly,
  and relying on it arriving as a transitive dependency of `mcp` is one upstream refactor away
  from a broken suite.
- .gitignore covers the AI state files AND every sidecar the budget/rate-limit code can leave
  beside them (`.lock`, `.corrupt`, the mkstemp `.<rand>.tmp`), so a `git add -A` after a dev
  session never ships spend history or a stray lock file.

Milestone 2 — budget.py accepts numbers.Integral token counts:
- `estimate`/`charge` refused numpy integer usage counters (`np.int64` summed off a DataFrame is
  a perfectly good token count) because the check was `isinstance(value, int)`. The fix accepts
  any `numbers.Integral` AND coerces to a plain `int`, since `np.int64` is not JSON-serialisable
  and the ledger write would otherwise fail *after* the money was spent.

Milestone 3 — agent.py pre-spend validation, None-Sharpe guard, tool_choice/thinking pin:
- `_call(tools=[])` raised a bare IndexError from the cache-breakpoint placement; it now raises a
  clear ValueError before the request is built (nothing sent, nothing charged).
- `_val_sharpe` tolerates `sharpe is None` (a NaN that went through a JSON round-trip) as `-inf`
  instead of TypeError, so ranking stored history cannot crash.
- `run_research` rejects an unhashable engine with the documented ValueError (not TypeError) and
  validates tickers BEFORE constructing the SDK client, so an unknown ticker fails identically
  with or without credentials.
- The request shape (forced `tool_choice: any` + no `thinking` key on `claude-sonnet-5`) was
  verified against the claude-api skill as a supported Claude-API combination and is pinned
  here so a future "fix" cannot silently add or remove a key.
- The holdout is always scored at `guardrails._HOLDOUT_COST_BPS`, whatever `cost_bps` the loop
  used — pinned with a spy on the engine registry entry.

Milestone 4 — nl_interface.py pre-gate query validation; non-list tickers -> clarify:
- `handle("")` / whitespace / >2000 chars used to reach `_validate_query`'s ValueError only
  AFTER `guardrails.rate_limit` had recorded a slot and `budget.allow` had read the ledger, so a
  stranger hammering "submit" on an empty box burned their hourly allowance on nothing. Now the
  query shape is checked FIRST and returns a clarification (fallback None, spend 0.0) with no
  slot recorded and no ledger touched; a non-str query is still a caller bug -> ValueError.
  `parse`/`parse_with_spend` keep raising ValueError (pinned by the existing suites).
- `_raw_plan` used to `list()` whatever the model put in `tickers`: a str exploded into single
  characters and a dict into its keys. A non-list is now a clarification; `None`/`[]`/missing
  still mean the whole universe and a list passes through untouched.

Milestone 6 — `ruff format` applied to the six files that were not `--check` clean:
- src/quantforge/data/loader.py, src/quantforge/metrics/performance.py and four older test files
  (interchange round-trip/verifier, loader, loader constants) were the only files the formatter
  would still touch. They were reformatted with NO manual edits; the AST of every one was proven
  equal to its HEAD version by a throwaway scratchpad script. What is pinned here is the part a
  future edit could silently undo: `loader.UNIVERSE` keeps the exact order/length the constants
  suite restates from the design doc (the formatter exploded the list one ticker per line, and
  its section comments must survive in place), and the six paths stay `ruff format --check`
  clean so the repo-wide check keeps reporting zero `Would reformat`.

Milestone 7 — closeout self-check:
- The newest handoff.md entry is the dated 2026-09-12 fix-run entry and names what a fresh
  session needs to pick up from it (the `74af8d9` commit correction, this proving file, the
  still-placeholder `ANTHROPIC_API_KEY`, Week 9 as next, and a real `N passed / 2 skipped`
  count); every earlier entry is byte-identical to `HEAD:handoff.md` (same split logic as the
  week-8 verifier; skipped when git is unavailable), so prepending never rewrote history.
- docs/components/16-tests.md gained a row for this file with its status header untouched, and
  docs/TEN_WEEK_PLAN.md is byte-identical to HEAD (this run ticked nothing).

The `FakeClient`/response/`isolated`/`windows` helpers are copied (minimally) from
tests/test_agent.py because tests/ is not a package; `isolated` here is NOT autouse so the
milestone 1/2 tests keep their own environment. Milestone 4 reuses the same `FakeClient` (it
matches tests/test_nl_interface.py's) with its own `nl_isolated` fixture.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange
from quantforge.ai import agent, budget, guardrails, mcp_server, nl_interface
from quantforge.data import loader
from quantforge.data.loader import UNIVERSE

_ROOT = Path(__file__).resolve().parents[1]
_ENV_EXAMPLE = (_ROOT / ".env.example").read_text()
_REQUIREMENTS = (_ROOT / "requirements.txt").read_text()


def _env_example_vars() -> dict[str, str]:
    """{VAR: value} for every 'VAR=value' assignment line in .env.example."""
    return dict(re.findall(r"^([A-Z_]+)=(.*)$", _ENV_EXAMPLE, flags=re.M))


# ---------------------------------------------------------------------------
# Milestone 1: .env.example / requirements.txt / .gitignore
# ---------------------------------------------------------------------------


def test_env_example_budget_caps_equal_module_dev_defaults():
    values = _env_example_vars()
    # Compare through float so the pin follows budget.py if a default is ever re-tuned, rather
    # than hard-coding "2" / "10" a second time.
    assert float(values["AI_BUDGET_USD_DAILY"]) == budget._DEV_DEFAULT_DAILY_USD
    assert float(values["AI_BUDGET_USD_TOTAL"]) == budget._DEV_DEFAULT_TOTAL_USD
    # And the prose comment still quotes the same pair, so the file cannot contradict itself.
    daily, total = int(budget._DEV_DEFAULT_DAILY_USD), int(budget._DEV_DEFAULT_TOTAL_USD)
    assert re.search(rf"{daily} \(daily\) / {total} \(total\)", _ENV_EXAMPLE)


def test_requirements_lists_jsonschema_explicitly():
    lines = [ln.strip() for ln in _REQUIREMENTS.splitlines()]
    assert any(ln.startswith("jsonschema") for ln in lines), "jsonschema missing from requirements"
    # Under the Dev / test header, since only tests import it.
    dev_idx = next(i for i, ln in enumerate(lines) if ln.startswith("# Dev / test"))
    js_idx = next(i for i, ln in enumerate(lines) if ln.startswith("jsonschema"))
    assert js_idx > dev_idx


def test_ai_state_files_and_all_sidecars_are_gitignored():
    if shutil.which("git") is None or not (_ROOT / ".git").exists():
        pytest.skip("git or the repo metadata is unavailable")
    # The mkstemp temp is `<ledger>.<rand>.tmp` (budget._write_ledger uses prefix=path.name + "."),
    # so a prefix glob on the ledger name covers it; a crashed write must not leak into a commit.
    ledger = Path(budget._DEFAULT_LEDGER_PATH)
    paths = [
        ".env",
        str(ledger),
        str(ledger) + ".lock",
        str(ledger) + ".corrupt",
        str(ledger) + ".Ab12xy.tmp",
        "data_cache/ai_rate_limits.json",
        "data_cache/ai_rate_limits.json.lock",
    ]
    for rel in paths:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", rel],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"{rel} is not gitignored"


# ---------------------------------------------------------------------------
# Milestone 2: budget.py token counts accept any numbers.Integral
# ---------------------------------------------------------------------------

_HAIKU = "claude-haiku-4-5"


@pytest.fixture
def tmp_ledger(tmp_path, monkeypatch):
    """Redirect the ledger into tmp_path (same shape as tests/test_budget.py's fixture)."""
    path = tmp_path / "ledger" / "ai_ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(path))
    monkeypatch.setenv("PUBLIC_MODE", "off")
    monkeypatch.delenv("AI_DISABLED", raising=False)
    monkeypatch.delenv("AI_BUDGET_USD_DAILY", raising=False)
    monkeypatch.delenv("AI_BUDGET_USD_TOTAL", raising=False)
    return path


def test_estimate_accepts_numpy_ints_and_returns_python_float():
    # 1M in at $1/MTok + 200k out at $5/MTok = $2 exactly, whichever integer type carries it.
    got = budget.estimate(_HAIKU, np.int64(1_000_000), np.int32(200_000))
    assert got == budget.estimate(_HAIKU, 1_000_000, 200_000) == 2.0
    assert type(got) is float, "a numpy scalar must not leak out of estimate()"


def test_charge_with_numpy_ints_writes_json_parsable_ledger(tmp_ledger):
    budget.charge(0.5, model=_HAIKU, tokens_in=np.int64(7), tokens_out=np.uint8(3))
    # json.load (not the module's own reader) proves nothing numpy-typed reached json.dump.
    with open(tmp_ledger, encoding="utf-8") as fh:
        data = json.load(fh)
    (bucket,) = data["days"].values()
    assert bucket["tokens_in"] == 7 and type(bucket["tokens_in"]) is int
    assert bucket["tokens_out"] == 3 and type(bucket["tokens_out"]) is int
    assert bucket["calls"] == 1 and type(bucket["calls"]) is int
    assert bucket["usd"] == 0.5


@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), 1.0, np.float64(1.0), np.int64(-1), "3"],
    ids=["True", "False", "np.bool_", "float", "np.float64", "np.int64(-1)", "str"],
)
def test_token_count_rejections_still_raise_for_non_integral_or_negative(bad, tmp_ledger):
    with pytest.raises(ValueError, match="must be a non-negative int"):
        budget.estimate(_HAIKU, bad, 0)
    with pytest.raises(ValueError, match="must be a non-negative int"):
        budget.estimate(_HAIKU, 0, bad)
    with pytest.raises(ValueError, match="must be a non-negative int"):
        budget.charge(0.1, model=_HAIKU, tokens_in=bad, tokens_out=0)
    assert not tmp_ledger.exists(), "a rejected charge must not touch the ledger"


def test_check_token_count_uses_numbers_integral_and_returns_int():
    src = Path(budget.__file__).read_text()
    assert "import numbers" in src
    assert "isinstance(value, numbers.Integral)" in src
    body = src.split("def _check_token_count")[1].split("\ndef ")[0]
    assert "isinstance(value, int)" not in body
    assert budget._check_token_count("x", np.int64(5)) == 5
    assert type(budget._check_token_count("x", np.int64(5))) is int


# ---------------------------------------------------------------------------
# Milestone 3: agent.py — helpers copied from tests/test_agent.py (tests/ is not a package)
# ---------------------------------------------------------------------------

_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM"]
_GOAL = "find a momentum variant with Sharpe > 1 on validation"


def _usage(tokens_in: int, tokens_out: int):
    return SimpleNamespace(
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def _tool_use_response(name: str, data: dict, block_id: str):
    block = SimpleNamespace(type="tool_use", id=block_id, name=name, input=data)
    return SimpleNamespace(content=[block], usage=_usage(1000, 200), stop_reason="tool_use")


def propose_response(strategy, params, rationale="try it", tool_id="toolu_1"):
    data = {"strategy": strategy, "params": params, "rationale": rationale}
    return _tool_use_response("propose_experiment", data, tool_id)


def done_response(reason="converged", tool_id="toolu_done"):
    return _tool_use_response("declare_done", {"reason": reason}, tool_id)


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
        return self._queue.pop(0)


def _synthetic_long(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2026-06-30", tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), len(_TICKERS)))
    wide = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=_TICKERS)
    return interchange.to_long(wide, "prices")


_SYNTHETIC = _synthetic_long()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Synthetic prices, temp ledger, dev-mode env, no real client, no API key (not autouse)."""
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _SYNTHETIC)
    mcp_server.reset_registry()
    ledger = tmp_path / "cache" / "ai_ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(ledger))
    for var in (
        "PUBLIC_MODE",
        "AI_DISABLED",
        "AI_BUDGET_USD_DAILY",
        "AI_BUDGET_USD_TOTAL",
        "AI_RATE_LIMIT_PER_HOUR",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(agent, "_client", None)
    # Any path that would build a real client must fail loudly, with a message no other layer
    # produces, so a test can tell "client never built" from "client built, then rejected".
    monkeypatch.setattr(
        agent, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("real client built"))
    )
    yield ledger
    mcp_server.reset_registry()


def _windows() -> tuple[str, str]:
    bounds = loader.get_split_bounds()
    train_id = mcp_server.load_data(*bounds["train"])["dataset_id"]
    val_id = mcp_server.load_data(*bounds["validation"])["dataset_id"]
    return train_id, val_id


# ---------------------------------------------------------------- (5) _call rejects tools=[]


def test_call_with_empty_tools_raises_value_error_before_any_request(isolated):
    client = FakeClient(propose_response("momentum", {"lookback": 60}))
    with pytest.raises(ValueError, match="at least one tool"):
        agent._call(
            client,
            system="s",
            messages=[{"role": "user", "content": "x"}],
            tools=[],
            tool_choice={"type": "any"},
            max_tokens=10,
        )
    assert client.calls == [], "the request must be refused before it is sent"
    assert not isolated.exists(), "nothing was sent, so nothing may be charged"


# ---------------------------------------------------------------- (6) None Sharpe


def test_val_sharpe_none_ranks_as_minus_inf():
    assert agent._val_sharpe({"val_metrics": {"sharpe": None}}) == -math.inf
    # The existing NaN / inf mapping is unchanged.
    assert agent._val_sharpe({"val_metrics": {"sharpe": float("nan")}}) == -math.inf
    assert agent._val_sharpe({"val_metrics": {"sharpe": 1.25}}) == 1.25


def _record(iter_, sharpe, error=None):
    metrics = None if error is not None else {"sharpe": sharpe}
    return {
        "iter": iter_,
        "strategy": "momentum",
        "params": {"lookback": 60},
        "train_metrics": metrics,
        "val_metrics": metrics,
        "rationale": "",
        "error": error,
    }


def test_select_best_skips_none_sharpe_and_picks_the_finite_record():
    history = [_record(1, None), _record(2, 0.8), _record(3, None), _record(4, 0.5)]
    best = agent._select_best(history)
    assert best is not None and best["val_metrics"]["sharpe"] == 0.8
    # A None Sharpe is a last-ranked candidate (like NaN), not an error: alone it still wins,
    # while an errored record (val_metrics is None) is skipped before ranking, as before.
    only = agent._select_best([_record(1, None), _record(2, 0.1, error="bad")])
    assert only is not None and only["val_metrics"] == {"sharpe": None}
    assert agent._select_best([_record(1, 0.1, error="bad")]) is None


# ---------------------------------------------------------------- (7) engine / tickers pre-spend


def test_run_research_unhashable_engine_raises_the_documented_value_error(isolated):
    client = FakeClient(propose_response("momentum", {"lookback": 60}))
    with pytest.raises(ValueError, match="registered engines"):
        agent.run_research(_GOAL, engine=["python"], client=client)
    assert client.calls == []
    assert mcp_server._DATASETS == {}


def test_unknown_ticker_fails_before_the_sdk_client_is_built(isolated):
    # client=None is the production path: the SDK client would be constructed lazily. With the
    # API key unset that construction must never be reached — the ticker error comes first.
    with pytest.raises(ValueError, match="not in the fixed universe"):
        agent.run_research(_GOAL, tickers=["AAPL", "ZZZZ"], client=None)
    assert agent._client is None
    assert mcp_server._DATASETS == {}
    assert not isolated.exists()


@pytest.mark.parametrize("tickers", [[], "AAPL", ["AAPL", "AAPL"], [1]])
def test_bad_ticker_shapes_also_fail_before_the_client(isolated, tickers):
    with pytest.raises(ValueError, match="tickers"):
        agent.run_research(_GOAL, tickers=tickers, client=None)
    assert agent._client is None


def test_ticker_validation_happens_after_goal_engine_and_cost_checks(isolated):
    # Order pin: the cheap scalar checks still win over the ticker check, so their messages are
    # unchanged for callers that get several things wrong at once.
    with pytest.raises(ValueError, match="goal"):
        agent.run_research("", tickers=["ZZZZ"], client=None)
    with pytest.raises(ValueError, match="registered engines"):
        agent.run_research(_GOAL, engine="r", tickers=["ZZZZ"], client=None)
    with pytest.raises(ValueError, match="cost_bps"):
        agent.run_research(_GOAL, cost_bps=101, tickers=["ZZZZ"], client=None)


# ---------------------------------------------------------------- (8) tool_choice / thinking pin


def test_request_shape_forced_tool_choice_without_thinking_on_sonnet(isolated):
    client = FakeClient(propose_response("momentum", {"lookback": 60}), done_response())
    out = agent.run_research(_GOAL, max_iters=3, client=client)
    assert out["stopped_because"] == "converged"
    assert len(client.calls) == 2
    for req in client.calls:
        assert set(req) == {"model", "max_tokens", "system", "tools", "tool_choice", "messages"}
        assert "thinking" not in req
        assert req["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
        assert req["model"] == "claude-sonnet-5"


def test_model_is_sonnet_5_not_a_model_that_rejects_forced_tool_choice():
    # Forced tool_choice (any/tool) returns 400 only on Claude Fable 5.1 / Mythos 5.1; the
    # agent's model must never silently drift onto one of those with this request shape.
    assert agent.MODEL == "claude-sonnet-5"
    assert not agent.MODEL.startswith(("claude-fable-5-1", "claude-mythos-5-1"))
    assert agent._TOOL_CHOICE == {"type": "any", "disable_parallel_tool_use": True}


def test_call_source_sets_no_thinking_literal():
    src = inspect.getsource(agent._call)
    assert '"thinking"' not in src
    assert "'thinking'" not in src
    # And the decision is written down where the next reader will look.
    assert "claude-api" in src and "Bedrock" in src


def test_agent_doc_records_the_tool_choice_and_holdout_cost_decisions():
    doc = (_ROOT / "docs" / "components" / "13-ai-agent.md").read_text()
    decisions = doc.split("## Decisions made in build")[1].split("\n## ")[0]
    assert "tool_choice" in decisions and "thinking" in decisions
    assert "_HOLDOUT_COST_BPS" in decisions


# ---------------------------------------------------------------- (12) holdout cost is fixed


class _SpyEngine:
    """Wraps the registered engine and records every params dict it is handed."""

    def __init__(self, inner):
        self._inner = inner
        self.params: list[dict] = []

    def run_backtest(self, prices, positions, params):
        self.params.append(dict(params))
        return self._inner.run_backtest(prices, positions, params)


def test_holdout_is_scored_at_guardrails_cost_regardless_of_loop_cost(isolated, monkeypatch):
    spy = _SpyEngine(mcp_server.ENGINES["python"])
    monkeypatch.setitem(mcp_server.ENGINES, "python", spy)
    client = FakeClient(propose_response("momentum", {"lookback": 60}), done_response())

    out = agent.run_research(_GOAL, max_iters=3, cost_bps=50.0, client=client)

    assert out["holdout_metrics"] is not None
    # train, validation, then exactly one holdout run.
    assert len(spy.params) == 3
    assert [p["cost_bps"] for p in spy.params[:2]] == [50.0, 50.0]
    assert spy.params[-1]["cost_bps"] == guardrails._HOLDOUT_COST_BPS
    assert guardrails._HOLDOUT_COST_BPS != 50.0  # the pin is only meaningful if they differ
    assert "_HOLDOUT_COST_BPS" in (agent.run_research.__doc__ or "")


# ---------------------------------------------------------------------------
# Milestone 4: nl_interface — query shape gate before the rate limit; non-list tickers
# ---------------------------------------------------------------------------


def _plan_response(plan: dict):
    """A canned parse response: one ``plan_backtest`` tool_use block (as in test_nl_interface)."""
    return _tool_use_response(nl_interface.PLAN_TOOL["name"], plan, "toolu_01")


@pytest.fixture
def nl_isolated(tmp_path, monkeypatch):
    """Empty temp dir for the ledger + rate state, dev env, no API key, no real client.

    Yields the ledger path; the rate-limit state file is derived from it by guardrails. The
    directory starts EMPTY so "no file exists afterwards" is the strongest possible proof that a
    gate was never consulted (a read-only ``budget.allow`` would still not create the ledger, but
    ``rate_limit`` writes on every admitted call — so its absence is the real signal).
    """
    ledger = tmp_path / "state" / "ai_ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(ledger))
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
    monkeypatch.setattr(
        nl_interface,
        "_get_client",
        lambda: (_ for _ in ()).throw(AssertionError("real client built")),
    )
    return ledger


def _rate_state(ledger: Path) -> Path:
    path = guardrails._rate_state_path()
    assert path == ledger.with_name("ai_rate_limits.json")
    return path


_BAD_QUERIES = ["", "   \n", "x" * 2001]


@pytest.mark.parametrize("bad", _BAD_QUERIES, ids=["empty", "whitespace", "over-long"])
def test_handle_bad_query_is_a_free_clarification_that_records_nothing(nl_isolated, bad):
    client = FakeClient()
    out = nl_interface.handle(bad, session_key="k", client=client)

    assert set(out) == {"plan", "metrics", "explanation", "spend_usd", "fallback"}
    assert set(out["plan"]) == {"clarify"}
    assert out["plan"]["clarify"] == nl_interface._CLARIFY_EMPTY_QUERY
    assert out["explanation"] == out["plan"]["clarify"]
    assert out["metrics"] is None
    assert out["fallback"] is None
    assert out["spend_usd"] == 0.0 and type(out["spend_usd"]) is float
    assert client.calls == []
    # Nothing was consulted: no rate slot written, no ledger created.
    assert not _rate_state(nl_isolated).exists()
    assert not nl_isolated.exists()


def test_clarification_message_names_the_real_character_limit():
    # Built from the constant, not a retyped number, so the two cannot drift apart.
    assert str(nl_interface._MAX_QUERY_CHARS) in nl_interface._CLARIFY_EMPTY_QUERY
    assert nl_interface._MAX_QUERY_CHARS == 2000


@pytest.mark.parametrize("bad", _BAD_QUERIES, ids=["empty", "whitespace", "over-long"])
def test_bad_query_consumes_no_rate_slot_so_the_next_valid_query_is_still_admitted(
    nl_isolated, monkeypatch, bad
):
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "1")

    first = nl_interface.handle(bad, session_key="k", client=FakeClient())
    assert first["fallback"] is None and set(first["plan"]) == {"clarify"}
    state = _rate_state(nl_isolated)
    assert not state.exists()

    # The ONE slot is still free: the valid query is admitted and the model is actually called.
    client = FakeClient(_plan_response({"clarify": "Which strategy?"}))
    second = nl_interface.handle("q", session_key="k", client=client)
    assert second["fallback"] is None
    assert second["plan"] == {"clarify": "Which strategy?"}
    assert len(client.calls) == 1
    assert len(json.loads(state.read_text())["k"]) == 1

    # ...and the limit of 1 is real: a third valid query is now rate-limited with no call.
    third_client = FakeClient()
    third = nl_interface.handle("q", session_key="k", client=third_client)
    assert third["fallback"] == "rate" and third_client.calls == []


def test_bad_query_short_circuits_before_rate_limit_and_budget_are_even_called(
    nl_isolated, monkeypatch
):
    # Stronger than "no file exists": make both gates explode, and prove they are never reached.
    monkeypatch.setattr(
        guardrails, "rate_limit", lambda key: (_ for _ in ()).throw(AssertionError("rate"))
    )
    monkeypatch.setattr(
        budget, "allow", lambda est: (_ for _ in ()).throw(AssertionError("budget"))
    )
    for bad in _BAD_QUERIES:
        out = nl_interface.handle(bad, session_key="k", client=FakeClient())
        assert out["plan"] == {"clarify": nl_interface._CLARIFY_EMPTY_QUERY}
        assert out["fallback"] is None and out["spend_usd"] == 0.0


@pytest.mark.parametrize("bad", [None, 42, b"bytes", ["q"]])
def test_handle_non_string_query_is_a_caller_bug_raised_before_any_gate(nl_isolated, bad):
    client = FakeClient()
    with pytest.raises(ValueError, match="must be a string"):
        nl_interface.handle(bad, session_key="k", client=client)
    assert client.calls == []
    assert not _rate_state(nl_isolated).exists()
    assert not nl_isolated.exists()


def test_parse_still_raises_for_the_same_bad_queries(nl_isolated):
    # ``handle`` softens a bad query into a clarification; the lower-level API keeps its contract.
    for bad in _BAD_QUERIES:
        with pytest.raises(ValueError):
            nl_interface.parse(bad, client=FakeClient())
    assert not nl_isolated.exists()


def test_handle_docstring_and_source_put_query_validation_before_the_rate_limit():
    doc = nl_interface.handle.__doc__
    assert "query shape -> rate limit -> budget -> parse" in doc
    src = inspect.getsource(nl_interface.handle)
    body = src[src.index('"""', src.index('"""') + 3) + 3 :]  # after the docstring
    assert body.index("isinstance(query, str)") < body.index("rate_limit(")
    assert body.index("_MAX_QUERY_CHARS") < body.index("rate_limit(")


def test_nl_doc_gate_order_lists_query_validation_first():
    text = (_ROOT / "docs" / "components" / "12-nl-interface.md").read_text()
    assert text.index("query shape") < text.index("rate_limit(session_key)")
    assert "query validation" in text


# ---------------------------------------------------------------- _raw_plan tickers shape

_ESSENTIALS = {"strategy": "momentum", "start": "2015-01-01", "end": "2019-12-31"}


@pytest.mark.parametrize("tickers", ["AAPL", {"a": 1}, 7, 1.5, True], ids=repr)
def test_raw_plan_non_list_tickers_is_a_clarification_not_a_mangled_list(tickers):
    out = nl_interface._raw_plan(_plan_response({**_ESSENTIALS, "tickers": tickers}))
    assert set(out) == {"clarify"}
    assert "tickers" in out["clarify"]


def test_raw_plan_missing_none_or_empty_tickers_means_the_whole_universe():
    for data in (_ESSENTIALS, {**_ESSENTIALS, "tickers": None}, {**_ESSENTIALS, "tickers": []}):
        out = nl_interface._raw_plan(_plan_response(data))
        assert "clarify" not in out
        assert out["tickers"] == list(UNIVERSE)


def test_raw_plan_list_tickers_pass_through_unchanged():
    out = nl_interface._raw_plan(_plan_response({**_ESSENTIALS, "tickers": ["AAPL", "MSFT"]}))
    assert out["tickers"] == ["AAPL", "MSFT"]
    assert out["strategy"] == "momentum" and out["cost_bps"] == 10.0
    # A list of non-strings is NOT the parser's problem: it passes through so
    # mcp_server._validate_tickers rejects it in _run_tools (fallback "invalid"), as documented.
    out = nl_interface._raw_plan(_plan_response({**_ESSENTIALS, "tickers": [1]}))
    assert out["tickers"] == [1]
    with pytest.raises(ValueError):
        mcp_server._validate_tickers(out["tickers"])


# ---------------------------------------------------------------------------
# Milestone 5 — mcp_server + guardrails doc/docstring corrections (text-only code changes)
# ---------------------------------------------------------------------------
#
# Nothing executable changed in this milestone; these tests pin that the DOCUMENTED behaviour
# now matches the real one, from both sides: the code is exercised (validation order, schema
# shape) and the prose is checked for the claims it now makes.

_MCP_SERVER_SOURCE = (_ROOT / "src" / "quantforge" / "ai" / "mcp_server.py").read_text()
_GUARDRAILS_SOURCE = (_ROOT / "src" / "quantforge" / "ai" / "guardrails.py").read_text()
_DOC_MCP = (_ROOT / "docs" / "components" / "11-mcp-server.md").read_text()
_DOC_GUARDRAILS = (_ROOT / "docs" / "components" / "10-ai-guardrails.md").read_text()

_M5_TICKERS = ["AAPL", "MSFT", "NVDA"]


def _m5_synthetic_long(seed: int = 5) -> pd.DataFrame:
    """Long interchange 'prices' frame on business days, tz-aware UTC (test_mcp_tools pattern)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2026-06-30", tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), len(_M5_TICKERS)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return interchange.to_long(pd.DataFrame(prices, index=dates, columns=_M5_TICKERS), "prices")


_M5_SYNTHETIC = _m5_synthetic_long()


@pytest.fixture
def public_dataset(monkeypatch) -> str:
    """A loaded synthetic dataset_id with PUBLIC_MODE=on and the loader never consulted."""
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _M5_SYNTHETIC)
    monkeypatch.setenv("PUBLIC_MODE", "on")
    mcp_server.reset_registry()
    ds = mcp_server.load_data("2015-01-01", "2019-12-31", _M5_TICKERS)["dataset_id"]
    yield ds
    mcp_server.reset_registry()


# ---------------------------------------------------------------- (9) validation order proof


def test_non_dict_params_is_a_plain_value_error_before_the_public_mode_gate(public_dataset):
    # A list is the shape a "code" smuggle attempt might take, but it is rejected as a malformed
    # call, not as a PublicModeViolation — the gate never sees it.
    with pytest.raises(ValueError, match="params must be a dict") as info:
        mcp_server.run_backtest(public_dataset, strategy="momentum", params=["code"])
    assert not isinstance(info.value, guardrails.PublicModeViolation)
    assert mcp_server._RESULTS == {}


def test_non_str_strategy_is_a_plain_value_error_before_the_public_mode_gate(public_dataset):
    with pytest.raises(ValueError, match="strategy must be a string") as info:
        mcp_server.run_backtest(public_dataset, strategy=["momentum"], params={})
    assert not isinstance(info.value, guardrails.PublicModeViolation)
    assert mcp_server._RESULTS == {}


def test_unknown_dataset_is_checked_before_anything_else(public_dataset):
    # Both the strategy and the params here would trip the gate; the dataset check wins.
    with pytest.raises(ValueError, match="unknown dataset_id") as info:
        mcp_server.run_backtest("ds_nope", strategy="code", params={"code": "x"})
    assert not isinstance(info.value, guardrails.PublicModeViolation)


def test_gate_runs_before_validate_params_in_public_mode(public_dataset):
    # `code` is not a whitelisted momentum param, so validate_params would also reject it — but
    # in PUBLIC_MODE the gate fires first and the failure is typed as a PublicModeViolation.
    with pytest.raises(guardrails.PublicModeViolation):
        mcp_server.run_backtest(public_dataset, strategy="momentum", params={"code": "x"})
    assert mcp_server._RESULTS == {}


def test_same_bad_params_outside_public_mode_is_a_plain_whitelist_error(
    public_dataset, monkeypatch
):
    # Sanity check on the previous test: with the gate off, the same call fails on the whitelist
    # with a plain ValueError — so it really was the gate that typed the public-mode failure.
    monkeypatch.delenv("PUBLIC_MODE")
    with pytest.raises(ValueError) as info:
        mcp_server.run_backtest(public_dataset, strategy="momentum", params={"code": "x"})
    assert not isinstance(info.value, guardrails.PublicModeViolation)


def test_run_backtest_docstring_states_the_real_validation_order():
    doc = inspect.getdoc(mcp_server.run_backtest)
    assert doc is not None
    i_dataset = doc.index("dataset")
    i_params = doc.index("params")
    i_gate = doc.index("assert_no_codegen")
    i_validate = doc.index("validate_params")
    assert i_dataset < i_params < i_gate < i_validate
    assert "PublicModeViolation" in doc  # the "never surfaces as" sentence
    # The prose order must be the source order: dataset lookup, params check, strategy check,
    # gate, whitelist, engine, cost.
    src = inspect.getsource(mcp_server.run_backtest)
    body = src[src.index('"""', src.index('"""') + 3) + 3 :]
    order = [
        body.index("_get_dataset("),
        body.index("isinstance(params, dict)"),
        body.index("isinstance(strategy, str)"),
        body.index("assert_no_codegen("),
        body.index("validate_params("),
        body.index("not in ENGINES"),
        body.index("_validate_cost_bps("),
    ]
    assert order == sorted(order)


def test_mcp_doc_records_that_only_run_backtest_carries_the_gate():
    decisions = _DOC_MCP.split("## Decisions made in build")[1]
    flat = " ".join(decisions.split())
    assert "runs in `run_backtest` only" in flat
    assert "test_public_mode_no_codegen.py" in flat
    assert "`{strategy, params}`" in flat
    # The gate really is CALLED from run_backtest and nowhere else in the module (counted on the
    # AST so the docstring's own mention of the call does not inflate the count).
    calls = [
        node
        for node in ast.walk(ast.parse(_MCP_SERVER_SOURCE))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "assert_no_codegen"
    ]
    assert len(calls) == 1
    assert "assert_no_codegen" in inspect.getsource(mcp_server.run_backtest)
    for name in ("load_data", "optimize_portfolio", "get_metrics"):
        assert "assert_no_codegen" not in inspect.getsource(getattr(mcp_server, name))
    # The table row lists the same order as the docstring, in short form.
    row = next(line for line in _DOC_MCP.splitlines() if line.startswith("| `run_backtest`"))
    assert row.index("dataset_id") < row.index("params") < row.index("assert_no_codegen")
    assert row.index("assert_no_codegen") < row.index("validate_params")
    assert "before the gate" in row


# ---------------------------------------------------------------- (13) selector rule in schema


def _optimize_schema() -> dict:
    return next(s for s in mcp_server.TOOL_SCHEMAS if s["name"] == "optimize_portfolio")


def test_optimize_portfolio_schema_states_exactly_one_selector_without_oneof():
    entry = _optimize_schema()
    dumped = json.dumps(entry)
    assert "oneOf" not in dumped and "anyOf" not in dumped
    props = entry["input_schema"]["properties"]
    assert "exactly one" in props["result_ids"]["description"]
    assert "exactly one" in props["dataset_id"]["description"]
    assert "dataset_id" in props["result_ids"]["description"]
    assert "result_ids" in props["dataset_id"]["description"]
    assert "exactly one" in entry["description"]  # the tool-level sentence is kept
    assert entry["input_schema"]["additionalProperties"] is False


def test_tool_schemas_still_round_trip_through_json_with_sorted_enums():
    dumped = json.dumps(mcp_server.TOOL_SCHEMAS)
    assert json.loads(dumped) == mcp_server.TOOL_SCHEMAS
    assert "oneOf" not in dumped
    for entry in mcp_server.TOOL_SCHEMAS:
        for prop in entry["input_schema"]["properties"].values():
            # The scalar enums (strategy / engine / objective) are built with sorted(...); the
            # tickers enum is list(loader.UNIVERSE) in universe order, which is stable too.
            if "enum" in prop:
                assert prop["enum"] == sorted(prop["enum"])
        assert entry["input_schema"]["additionalProperties"] is False
    # Descriptions are plain str (the cached prefix must stay a stable byte sequence).
    props = _optimize_schema()["input_schema"]["properties"]
    assert all(isinstance(p["description"], str) for p in props.values())


def test_mcp_doc_and_module_docstring_explain_why_the_rule_is_prose_not_oneof():
    assert "oneOf" in (mcp_server.__doc__ or "")
    decisions = _DOC_MCP.split("## Decisions made in build")[1]
    assert "oneOf" in decisions and "exactly one of the two" in " ".join(decisions.split())


# ---------------------------------------------------------------- (10) guardrails docstrings


def test_holdout_handle_docstring_scopes_the_closure_guard_honestly():
    doc = inspect.getdoc(guardrails.HoldoutHandle)
    assert doc is not None
    assert "introspection" in doc and "load_data" in doc
    assert "test_holdout_isolation.py" in doc
    flat = " ".join(_DOC_GUARDRAILS.split())
    assert "introspection" in flat and "`load_data` rejection" in flat


def test_guardrails_doc_status_line_untouched_and_no_stale_words():
    # The status line is pinned by two closeout verifiers; the milestone-5 sentence must not
    # have moved it or introduced any of the words those verifiers forbid.
    status = _DOC_GUARDRAILS.splitlines()[2]
    assert status.startswith("**Weeks:** 7–9 · **Status:** built / green (wk 7)")
    assert "landed in week 8" in status
    for stale in ("lands in week 8", "is added in week 8", "week-8** item", "pending"):
        assert stale not in " ".join(_DOC_GUARDRAILS.split())


def test_resolve_engine_docstring_claim_matches_the_real_import():
    doc = inspect.getdoc(guardrails._resolve_engine)
    assert doc is not None
    assert "from quantforge.ai import guardrails" in doc
    assert "run_backtest" in doc and "function-local" in doc
    # The claim is checked against the real import in mcp_server.py ...
    assert re.search(r"^from quantforge\.ai import guardrails", _MCP_SERVER_SOURCE, re.M)
    # ... and the cycle-avoiding lazy import is still there, with no top-level counterpart.
    assert "from quantforge.ai.mcp_server import ENGINES" in inspect.getsource(
        guardrails._resolve_engine
    )
    assert not re.search(r"^(from|import) quantforge\.ai\.mcp_server", _GUARDRAILS_SOURCE, re.M)


# ================================================================ Milestone 6: ruff format

# The six files the formatter still wanted to touch at "week 8 complete". Listed by path, not
# discovered, so the test also documents WHICH files that milestone was allowed to change.
_FORMATTED_PATHS = (
    "src/quantforge/data/loader.py",
    "src/quantforge/metrics/performance.py",
    "tests/test_interchange_roundtrip.py",
    "tests/test_interchange_verifier.py",
    "tests/test_loader.py",
    "tests/test_loader_constants.py",
)

# Section comments inside `UNIVERSE`, in file order. The formatter moved every ticker onto its
# own line; the comments are the only thing that says WHY each group is there, so they must
# still be present and still in this order.
_UNIVERSE_SECTION_COMMENTS = (
    "# Tech",
    "# ADRs (international, US-listed, USD)",
    "# Financials",
    "# Healthcare",
    "# Consumer",
    "# Energy / Industrial",
)


def _load_test_module(name: str):
    """Import a sibling test file by path (tests/ is deliberately not a package)."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / "tests" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_universe_order_and_length_survive_reformat():
    # tests/test_loader_constants.py restates the universe from the design doc precisely so it
    # cannot just re-import what it checks; comparing against THAT list (not a copy made here)
    # means one source of truth for "the frozen 30" — and the length pin from test_loader.py.
    constants = _load_test_module("test_loader_constants")
    assert loader.UNIVERSE == constants._DOC_UNIVERSE
    assert len(loader.UNIVERSE) == 30
    assert len(set(loader.UNIVERSE)) == 30

    src = (_ROOT / "src/quantforge/data/loader.py").read_text()
    # Slice from the opening `= [` to the closing `]` on its own line (the `]` in `list[str]`
    # must not end the search).
    start = src.index("UNIVERSE: list[str] = [") + len("UNIVERSE: list[str] = [")
    block = src[start : src.index("\n]\n", start) + 1]  # keep the last line's newline
    positions = [block.find(c) for c in _UNIVERSE_SECTION_COMMENTS]
    assert all(pos >= 0 for pos in positions), positions
    assert positions == sorted(positions)  # comments still in file order
    # One ticker per line after the formatter: every ticker is its own `"XXX",` line.
    assert all(f'    "{t}",\n' in block for t in loader.UNIVERSE)
    # The survivorship caveat pinned by test_week3_closeout_verifier is untouched.
    assert "urvivorship" in src


def _ruff_executable() -> str | None:
    # Prefer the interpreter's own bin dir (the venv that ran pytest) so the version pinned in
    # requirements is the one that judges formatting; fall back to PATH.
    for candidate in (
        Path(sys.executable).parent / "ruff",
        Path(sys.executable).parent / "ruff.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def test_six_reformatted_files_are_ruff_format_check_clean():
    ruff = _ruff_executable()
    if ruff is None:
        pytest.skip("ruff executable not found")
    proc = subprocess.run(
        [ruff, "format", "--check", *_FORMATTED_PATHS],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would reformat" not in proc.stdout


# ===========================================================================
# Milestone 7: closeout self-check (handoff.md / 16-tests.md / plan untouched)
# ===========================================================================

_HANDOFF = (_ROOT / "handoff.md").read_text()
_DOC_TESTS = (_ROOT / "docs" / "components" / "16-tests.md").read_text()
_PLAN = (_ROOT / "docs" / "TEN_WEEK_PLAN.md").read_text()


def _handoff_entries(text: str) -> list[tuple[str, str]]:
    """(date, body) per '## YYYY-MM-DD — ...' entry, in file order — the week-8 verifier's split,
    repeated here rather than imported because tests/ is not a package."""
    matches = list(re.finditer(r"^## (\d{4}-\d{2}-\d{2}) — ", text, flags=re.M))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1), text[m.start() : end]))
    return out


def _baseline_ref() -> str | None:
    """The commit this fix run was built on top of ("week 8 complete", 74af8d9), found by
    message rather than pinned as HEAD: these pins compare the working tree against the state
    BEFORE the fix run, and must keep holding after the run itself is committed (HEAD moves;
    the baseline does not)."""
    if not (_ROOT / ".git").exists() or shutil.which("git") is None:
        return None
    proc = subprocess.run(
        ["git", "log", "--format=%H", "--grep=^week 8 complete", "-1"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    sha = proc.stdout.strip()
    return sha or None


def _git_show(rel: str) -> str | None:
    """The baseline-commit version of a repo file, or None when git is unavailable."""
    base = _baseline_ref()
    if base is None:
        return None
    proc = subprocess.run(
        ["git", "show", f"{base}:{rel}"], cwd=_ROOT, capture_output=True, text=True, check=False
    )
    return proc.stdout if proc.returncode == 0 else None


def test_newest_handoff_entry_is_the_2026_09_12_fix_run_and_says_what_matters():
    entries = _handoff_entries(_HANDOFF)
    date, body = entries[0]
    assert date == "2026-09-12"
    assert "open-items" in body.splitlines()[0]
    for needle in (
        "74af8d9",  # the correction: Week 8 WAS committed
        "test_handoff_open_items.py",
        "ANTHROPIC_API_KEY",  # still a placeholder — the live smokes have not run
        "Week 9",  # next build target
        "Agent failures",
        "Open items",
        "Next up",
    ):
        assert needle in body, f"2026-09-12 handoff entry lacks {needle!r}"
    # A real count from a fresh full run, in the bold `**N passed / 2 skipped**` form every
    # earlier entry used for its closeout count (the baseline phrase "1207 passed / 2 skipped /
    # 1 failed" in the same entry is deliberately NOT bold, so it cannot satisfy this pin).
    m = re.search(r"\*\*(\d+) passed / 2 skipped\*\*", body)
    assert m and int(m.group(1)) >= 1208, "closeout count missing, placeholder, or below baseline"
    assert "test_nl_interface.py::test_live_smoke" in body
    assert "test_agent.py::test_live_smoke" in body
    assert "QUANTFORGE_LIVE_AI" in body
    # The Week-8 entry is directly below, and dates stay newest-first.
    assert entries[1][0] == "2026-09-11" and "Week 8" in entries[1][1]
    dates = [d for d, _ in entries]
    assert dates == sorted(dates, reverse=True)


def test_every_earlier_handoff_entry_is_byte_identical_to_head():
    committed = _git_show("handoff.md")
    if committed is None:
        pytest.skip("git or the committed handoff.md is unavailable")
    old = _handoff_entries(committed)
    new = _handoff_entries(_HANDOFF)
    assert old and old[0][0] == "2026-09-11", "baseline should be the week-8 commit (74af8d9)"
    assert new[0][0] == "2026-09-12"
    assert [b for _, b in new[1:]] == [b for _, b in old], "an earlier handoff entry was edited"


def test_tests_doc_has_a_row_for_this_file_and_its_status_header_is_unchanged():
    row = next(
        line
        for line in _DOC_TESTS.splitlines()
        if line.startswith("| `test_handoff_open_items.py`")
    )
    assert "SF-1" in row and "SF-3" in row and "AR-6" in row
    assert "✅ green (2026-09-12)" in row
    # Every suite the row names really exists.
    for name in re.findall(r"`(test_\w+\.py)`", row):
        assert (_ROOT / "tests" / name).is_file(), name
    committed = _git_show("docs/components/16-tests.md")
    if committed is None:
        pytest.skip("git or the committed 16-tests.md is unavailable")
    # The status header (everything above "## Function") is pinned by the week-8 verifier and
    # must not have moved: the only change is the added row.
    header = _DOC_TESTS.split("## Function")[0]
    assert header == committed.split("## Function")[0]
    assert "wk 1–8 suites green; only the\nlive-AI smoke tests skip" in header


def test_plan_untouched_by_the_fix_run():
    assert len(re.findall(r"^- \[x\]", _PLAN, flags=re.M)) == 24
    committed = _git_show("docs/TEN_WEEK_PLAN.md")
    if committed is None:
        pytest.skip("git or the committed plan is unavailable")
    assert _PLAN == committed, "the fix run must tick nothing in docs/TEN_WEEK_PLAN.md"
