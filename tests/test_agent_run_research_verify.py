"""Independent verifier tests for ``agent.run_research`` (week 8, milestone 4).

Written against the milestone spec, not the builder's tests, and deliberately adversarial:

* **Hand-computed holdout.** The one honest number is recomputed here from first principles —
  ``positions.shift(1)`` times ``pct_change``, minus turnover times the guardrail's fixed cost,
  through ``compute_metrics`` — with no engine, guardrail, tool or agent code in the reference.
  Agreement at 1e-12 proves the runner scored the winner on the 2023+ slice with the documented
  lag and cost, and that the number is not a copy of a validation metric.
* **The holdout cost is the guardrail's, not the caller's.** ``cost_bps`` shapes the loop, but
  ``guardrails._HOLDOUT_COST_BPS`` is fixed: a frictionless loop still gets a 10 bps holdout.
  Pinned so nobody can "improve" the holdout after the fact by cheapening trading.
* **Missing holdout is loud.** Prices that end before 2023 make ``run_research`` raise after the
  loop (the paid model calls are still charged) instead of returning a result with a hole in it.
* **Bad arguments are rejected before a client exists**, including the non-string / bool / float
  shapes the spec's examples do not list.
* **Prompt-injection shape.** A proposal that smuggles a ``code`` key into ``params`` becomes a
  rejected record, and only the vetted winner reaches the holdout.
* Repeated configurations, ticker order, detached copies, and consecutive runs each getting a
  fresh, deterministic holdout score.

Fully offline: same ``FakeClient`` shape as ``tests/test_agent.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange
from quantforge.ai import agent, budget, guardrails, mcp_server
from quantforge.data import loader
from quantforge.metrics.performance import _KEYS, compute_metrics

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM"]
GOAL = "find a momentum variant that holds up out of sample"


# ---------------------------------------------------------------- fakes + data


def _usage(tokens_in=1000, tokens_out=200):
    return SimpleNamespace(
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def _tool_use(name, data, block_id):
    block = SimpleNamespace(type="tool_use", id=block_id, name=name, input=data)
    return SimpleNamespace(content=[block], usage=_usage(), stop_reason="tool_use")


def propose(strategy, params, block_id="t1"):
    return _tool_use(
        "propose_experiment", {"strategy": strategy, "params": params, "rationale": "x"}, block_id
    )


def done(block_id="td"):
    return _tool_use("declare_done", {"reason": "enough"}, block_id)


class FakeClient:
    def __init__(self, *responses):
        self.calls: list[dict] = []
        self._queue = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        assert self._queue, "more requests than canned responses"
        return self._queue.pop(0)


def _panel(end: str, seed: int = 7) -> pd.DataFrame:
    """A long 'prices' frame for TICKERS on business days from 2010-01-01 through ``end``."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", end, tz="UTC")
    rets = rng.normal(0.0003, 0.012, size=(len(dates), len(TICKERS)))
    wide = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=TICKERS)
    return interchange.to_long(wide, "prices")


_FULL = _panel("2026-06-30")
_NO_HOLDOUT = _panel("2022-12-31")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _FULL)
    mcp_server.reset_registry()
    ledger = tmp_path / "ai_ledger.json"
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

    def explode():
        raise AssertionError("a real client was constructed")

    monkeypatch.setattr(agent, "_get_client", explode)
    yield ledger
    mcp_server.reset_registry()


def _ledger_calls(ledger) -> int:
    if not ledger.exists():
        return 0
    return json.loads(ledger.read_text())["days"][budget._utc_today()]["calls"]


# ---------------------------------------------------------------- hand-computed reference


def hand_momentum_metrics(
    prices_long: pd.DataFrame, lookback: int, cost_bps: float, tickers=None
) -> dict:
    """Momentum (equal-weight on positive trailing return) on the 2023+ slice, from scratch.

    No engine, no guardrail, no strategy class: just the two rigor rules — a weight decided at
    the close of day t earns day t+1 (``shift(1)``) and each unit of turnover costs ``cost_bps``.
    """
    lo = pd.Timestamp(loader.get_split_bounds()["holdout"][0], tz="UTC")
    hi = pd.Timestamp(loader.get_split_bounds()["holdout"][1], tz="UTC")
    sub = prices_long
    if tickers is not None:
        sub = sub[sub["ticker"].isin(tickers)]
    sub = sub[sub["date"].between(lo, hi)]
    wide = sub.pivot(index="date", columns="ticker", values="close").sort_index()

    mom = wide / wide.shift(lookback) - 1.0
    positive = mom.clip(lower=0.0)
    row_sum = positive.sum(axis=1)
    weights = positive.div(row_sum.where(row_sum != 0), axis=0).fillna(0.0)
    weights.loc[mom.isna().all(axis=1)] = 0.0

    asset_returns = wide.pct_change().fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    gross = (held * asset_returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_bps / 10_000.0
    return compute_metrics(net)


def _assert_close(got: dict, want: dict) -> None:
    assert list(got) == _KEYS
    for key in _KEYS:
        assert got[key] == pytest.approx(want[key], abs=1e-12), key


# ---------------------------------------------------------------- the holdout number


def test_holdout_metrics_match_a_hand_computed_reference():
    client = FakeClient(propose("momentum", {"lookback": 60}), done())
    out = agent.run_research(GOAL, max_iters=3, client=client)

    assert out["best"]["params"]["lookback"] == 60
    assert out["stopped_because"] == "converged"
    reference = hand_momentum_metrics(_FULL, lookback=60, cost_bps=guardrails._HOLDOUT_COST_BPS)
    _assert_close(out["holdout_metrics"], reference)
    # A different window: the holdout number is not the validation number relabelled.
    assert out["holdout_metrics"]["sharpe"] != out["best"]["val_metrics"]["sharpe"]
    assert out["holdout_metrics"]["total_return"] != out["best"]["train_metrics"]["total_return"]


def test_holdout_is_scored_at_the_guardrail_cost_even_when_the_loop_is_frictionless():
    """A frictionless loop must not buy a frictionless holdout (the after-the-fact tuning hole)."""
    client = FakeClient(propose("momentum", {"lookback": 40}), done())
    out = agent.run_research(GOAL, max_iters=3, cost_bps=0.0, client=client)

    fixed = hand_momentum_metrics(_FULL, lookback=40, cost_bps=guardrails._HOLDOUT_COST_BPS)
    free = hand_momentum_metrics(_FULL, lookback=40, cost_bps=0.0)
    assert free["total_return"] > fixed["total_return"]  # the strategy trades, so cost bites
    _assert_close(out["holdout_metrics"], fixed)
    assert out["holdout_metrics"]["total_return"] != pytest.approx(free["total_return"])


def test_ticker_subset_holdout_matches_the_hand_reference_on_that_subset():
    client = FakeClient(propose("momentum", {"lookback": 30}), done())
    out = agent.run_research(GOAL, max_iters=3, tickers=["JPM", "NVDA"], client=client)
    reference = hand_momentum_metrics(
        _FULL, lookback=30, cost_bps=guardrails._HOLDOUT_COST_BPS, tickers=["JPM", "NVDA"]
    )
    _assert_close(out["holdout_metrics"], reference)
    full_universe = hand_momentum_metrics(_FULL, 30, guardrails._HOLDOUT_COST_BPS)
    assert out["holdout_metrics"]["sharpe"] != pytest.approx(full_universe["sharpe"])


def test_data_ending_before_the_holdout_raises_after_the_paid_loop(monkeypatch, isolated):
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _NO_HOLDOUT)
    client = FakeClient(propose("momentum", {"lookback": 60}), done())
    with pytest.raises(ValueError, match="holdout slice is empty"):
        agent.run_research(GOAL, max_iters=3, client=client)
    # The loop ran and was charged; the missing holdout is a loud fault, not a silent None.
    assert len(client.calls) == 2
    assert _ledger_calls(isolated) == 2


# ---------------------------------------------------------------- one shot, fresh per run


def _spies(monkeypatch):
    real_split, real_score = guardrails.split_data, guardrails.score_holdout
    seen = {"handles": [], "score": []}

    def split(prices=None):
        out = real_split(prices)
        seen["handles"].append(out[2])
        return out

    def score(handle, strategy, params, engine="python"):
        seen["score"].append((handle, strategy, dict(params), engine))
        return real_score(handle, strategy, params, engine)

    monkeypatch.setattr(guardrails, "split_data", split)
    monkeypatch.setattr(guardrails, "score_holdout", score)
    return seen


def test_repeated_configuration_best_is_the_earliest_and_holdout_scored_once(monkeypatch):
    seen = _spies(monkeypatch)
    client = FakeClient(
        propose("momentum", {"lookback": 60}, "t1"),
        propose("momentum", {"lookback": 60}, "t2"),
        propose("momentum", {"lookback": 60}, "t3"),
    )
    out = agent.run_research(GOAL, max_iters=3, client=client)

    assert out["stopped_because"] == "max_iters"
    assert [r["iter"] for r in out["history"]] == [1, 2, 3]
    assert out["best"] == {k: out["history"][0][k] for k in agent._CONFIG_KEYS}
    assert len(seen["handles"]) == 1 and len(seen["score"]) == 1
    handle = seen["handles"][0]
    assert handle.consumed is True
    assert handle.start == "2023-01-02"  # first business day of the holdout window
    assert handle.n_days == len(pd.bdate_range("2023-01-01", "2026-06-30"))
    assert json.loads(json.dumps(out)) == out


def test_consecutive_runs_each_get_a_fresh_handle_and_the_same_holdout_number(monkeypatch):
    seen = _spies(monkeypatch)
    first = agent.run_research(
        GOAL, max_iters=3, client=FakeClient(propose("momentum", {"lookback": 60}), done())
    )
    second = agent.run_research(
        GOAL, max_iters=3, client=FakeClient(propose("momentum", {"lookback": 60}), done())
    )
    assert len(seen["handles"]) == 2 and seen["handles"][0] is not seen["handles"][1]
    assert all(h.consumed for h in seen["handles"])
    assert first["holdout_metrics"] == second["holdout_metrics"]
    assert first["best"] == second["best"]


def test_ticker_order_does_not_change_history_or_holdout():
    a = agent.run_research(
        GOAL,
        max_iters=3,
        tickers=["AAPL", "MSFT"],
        client=FakeClient(propose("momentum", {"lookback": 50}), done()),
    )
    mcp_server.reset_registry()
    b = agent.run_research(
        GOAL,
        max_iters=3,
        tickers=["MSFT", "AAPL"],
        client=FakeClient(propose("momentum", {"lookback": 50}), done()),
    )
    assert a["history"] == b["history"]
    assert a["holdout_metrics"] == b["holdout_metrics"]


def test_result_is_detached_from_the_history_and_from_later_mutation():
    client = FakeClient(propose("momentum", {"lookback": 60}), done())
    out = agent.run_research(GOAL, max_iters=3, client=client)
    snapshot = json.loads(json.dumps(out["history"]))
    out["best"]["params"]["lookback"] = 1
    out["best"]["val_metrics"]["sharpe"] = 99.0
    out["holdout_metrics"]["sharpe"] = 99.0
    assert json.loads(json.dumps(out["history"])) == snapshot


# ---------------------------------------------------------------- adversarial inputs


def test_smuggled_code_key_in_params_is_a_rejected_record_and_never_reaches_the_holdout(
    monkeypatch,
):
    seen = _spies(monkeypatch)
    client = FakeClient(
        propose("momentum", {"lookback": 60, "code": "import os"}, "t1"),
        propose("momentum", {"lookback": 90}, "t2"),
        done(),
    )
    out = agent.run_research(GOAL, max_iters=4, client=client)

    rejected, accepted = out["history"]
    assert rejected["error"] is not None and "unknown param" in rejected["error"]
    assert rejected["train_metrics"] is None and rejected["val_metrics"] is None
    assert accepted["error"] is None
    assert out["best"]["params"]["lookback"] == 90
    assert len(seen["score"]) == 1
    _, strategy, params, engine = seen["score"][0]
    assert (strategy, params, engine) == ("momentum", out["best"]["params"], "python")
    assert "code" not in params


@pytest.mark.parametrize(
    "kwargs",
    [
        {"engine": None},
        {"engine": 1},
        {"engine": "PYTHON"},
        {"max_iters": True},
        {"max_iters": 2.0},
        {"max_iters": 0},
        {"cost_bps": "10"},
        {"cost_bps": True},
        {"cost_bps": -0.01},
        {"cost_bps": float("nan")},
        {"cost_bps": float("inf")},
    ],
)
def test_bad_scalar_arguments_raise_before_a_client_is_built(isolated, kwargs):
    # No client passed: reaching ``_get_client`` would raise the fixture's AssertionError, so a
    # ValueError proves validation came first. Nothing loaded, nothing charged.
    with pytest.raises(ValueError):
        agent.run_research(GOAL, **kwargs)
    assert mcp_server._DATASETS == {}
    assert not isolated.exists()


@pytest.mark.parametrize("goal", [None, 42, "   ", "x" * 2001])
def test_bad_goal_raises_before_a_client_is_built(goal):
    with pytest.raises(ValueError):
        agent.run_research(goal)
    assert mcp_server._DATASETS == {}


@pytest.mark.parametrize("tickers", [[], "AAPL", ["AAPL", "AAPL"], [1], ("AAPL",)])
def test_bad_ticker_shapes_raise_before_any_model_call(isolated, tickers):
    client = FakeClient(propose("momentum", {"lookback": 60}), done())
    with pytest.raises(ValueError):
        agent.run_research(GOAL, tickers=tickers, client=client)
    assert client.calls == []
    assert not isolated.exists()


def test_run_research_never_reads_the_holdout_dates_into_the_conversation():
    client = FakeClient(propose("momentum", {"lookback": 60}), done())
    agent.run_research(GOAL, max_iters=3, client=client)
    bounds = loader.get_split_bounds()["holdout"]
    for request in client.calls:
        blob = json.dumps(request, default=str)
        assert bounds[0] not in blob and bounds[1] not in blob
        assert "ds_" not in blob and "res_" not in blob  # no registry handles either
