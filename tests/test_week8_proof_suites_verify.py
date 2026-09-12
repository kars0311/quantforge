"""Independent verifier for the week-8 proof suites (agent-loop holdout clause + public-mode clause).

Re-proves, adversarially and by hand computation, what ``tests/test_holdout_isolation.py`` and
``tests/test_public_mode_no_codegen.py`` claim about ``agent.run_research``:

* The one holdout number is the HONEST one: it equals a direct strategy -> ``PythonEngine`` run on
  the 2023+ slice of the same synthetic panel at the fixed 10 bps cost (to 1e-12), and it is NOT
  the validation number re-labelled. Same again on a ticker subset.
* Strict token check (no 2-dp coincidence allowance): no holdout metric appears anywhere in the
  requests the model saw, at 2, 4 or ``repr`` precision.
* A budget stop still scores the honest number exactly once (a stop reason is not a way to skip,
  or double, the one shot).
* The model smuggling a holdout date as a param value is refused by the whitelist and mints no
  handle; the only 2023+ string the model ever sees is its own echoed proposal.
* PUBLIC_MODE: unvetted strategy and nested params never reach the engine; and with the gate
  itself no-op'd, the whitelist still stops the poisoned turn (defence in depth) while the
  builder's "names public mode" assertion becomes false — proving that assertion is load-bearing.

Everything runs offline with ``ANTHROPIC_API_KEY`` unset; helpers are duplicated minimally so this
file stands alone.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange
from quantforge.ai import agent, budget, guardrails, mcp_server
from quantforge.data import loader
from quantforge.engine.python_engine import PythonEngine
from quantforge.metrics.performance import _KEYS
from quantforge.strategies import STRATEGIES

TICKERS = list(loader.UNIVERSE[:4])
GOAL = "Find a robust momentum lookback."
HOLDOUT_CUTOFF = pd.Timestamp("2023-01-01", tz="UTC")
HOLDOUT_DATE_RE = re.compile(r"20(2[3-9]|[3-9]\d)-\d\d-\d\d")


def _panel(seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2026-06-30", tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), len(TICKERS)))
    wide = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=TICKERS)
    return interchange.to_long(wide, "prices")


_SYNTHETIC = _panel()


def _turn(name: str, data: dict, block_id: str):
    block = SimpleNamespace(type="tool_use", id=block_id, name=name, input=data)
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return SimpleNamespace(content=[block], usage=usage, stop_reason="tool_use")


def propose(strategy, params, tool_id: str, rationale: str = "x"):
    return _turn(
        "propose_experiment",
        {"strategy": strategy, "params": params, "rationale": rationale},
        tool_id,
    )


def done(tool_id: str = "toolu_done"):
    return _turn("declare_done", {"reason": "converged"}, tool_id)


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
    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def run_backtest(self, prices, positions, params=None):
        self.calls.append(dict(params or {}))
        return super().run_backtest(prices, positions, params)


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """Dev mode, synthetic panel into 2026, temp ledger, no real client, spy engine."""
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _SYNTHETIC)
    mcp_server.reset_registry()
    monkeypatch.setenv("AI_LEDGER_PATH", str(tmp_path / "ai_ledger.json"))
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
    monkeypatch.setattr(
        agent, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("real client"))
    )
    engine = SpyEngine()
    monkeypatch.setitem(mcp_server.ENGINES, "python", engine)
    yield engine
    mcp_server.reset_registry()


@pytest.fixture
def public_offline(offline, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1.0")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "5.0")
    return offline


def _hand_holdout_metrics(strategy: str, params: dict, tickers: list[str] | None = None) -> dict:
    """What the honest holdout number must be: strategy -> engine on the 2023+ slice, 10 bps."""
    prices = _SYNTHETIC
    if tickers is not None:
        prices = prices[prices["ticker"].isin(tickers)].reset_index(drop=True)
    lo = pd.Timestamp(loader.get_split_bounds()["holdout"][0], tz="UTC")
    hi = pd.Timestamp(loader.get_split_bounds()["holdout"][1], tz="UTC")
    holdout = interchange.to_wide(
        prices[prices["date"].between(lo, hi)].reset_index(drop=True), "prices"
    )
    assert holdout.index.min() >= HOLDOUT_CUTOFF
    positions = STRATEGIES[strategy]().generate_signals(holdout, params)
    result = PythonEngine().run_backtest(
        holdout, positions, {"cost_bps": 10.0, "strategy": strategy, **params}
    )
    return dict(result.metrics)


def _serialise(client: FakeClient) -> str:
    return json.dumps(client.calls, default=lambda o: vars(o) if hasattr(o, "__dict__") else str(o))


def _token(rendering: str) -> re.Pattern:
    return re.compile(r"(?<![\d.])" + re.escape(rendering) + r"(?![\d])")


# ---------------------------------------------------------------- the honest number, by hand


def test_holdout_metrics_equal_a_direct_engine_run_on_the_2023_plus_slice(offline):
    engine = offline
    client = FakeClient(
        propose("momentum", {"lookback": 60}, "t1"),
        propose("momentum", {"lookback": 120, "top_n": 2}, "t2"),
        done(),
    )
    result = agent.run_research(GOAL, max_iters=5, client=client)
    best = result["best"]
    assert result["stopped_because"] == "converged"

    expected = _hand_holdout_metrics(best["strategy"], best["params"])
    assert list(result["holdout_metrics"]) == _KEYS
    for key in _KEYS:
        assert result["holdout_metrics"][key] == pytest.approx(expected[key], abs=1e-12), key

    # Adversarial: the runner did not simply re-label the validation (or train) number.
    assert result["holdout_metrics"]["sharpe"] != pytest.approx(best["val_metrics"]["sharpe"])
    assert result["holdout_metrics"]["sharpe"] != pytest.approx(best["train_metrics"]["sharpe"])

    # The engine saw exactly 2 experiments x 2 windows + 1 holdout run, and the holdout run was
    # the LAST one, at the fixed cost, for the best config.
    assert len(engine.calls) == 5
    last = engine.calls[-1]
    assert last["cost_bps"] == 10.0 and last["strategy"] == best["strategy"]
    assert {k: last[k] for k in best["params"]} == best["params"]


def test_ticker_subset_scores_the_holdout_on_that_same_subset(offline):
    subset = TICKERS[:2]
    client = FakeClient(propose("momentum", {"lookback": 60, "top_n": 1}, "t1"), done())
    result = agent.run_research(GOAL, max_iters=3, client=client, tickers=subset)
    best = result["best"]

    on_subset = _hand_holdout_metrics(best["strategy"], best["params"], subset)
    on_full = _hand_holdout_metrics(best["strategy"], best["params"], None)
    for key in _KEYS:
        assert result["holdout_metrics"][key] == pytest.approx(on_subset[key], abs=1e-12), key
    assert result["holdout_metrics"]["sharpe"] != pytest.approx(on_full["sharpe"])
    # ...and the agent's datasets were cut to that subset and end before the holdout.
    for wide in mcp_server._DATASETS.values():
        assert list(wide.columns) == subset
        assert wide.index.max() < HOLDOUT_CUTOFF


# ---------------------------------------------------------------- strict leak check


def test_no_holdout_metric_token_in_any_request_strict(offline):
    """No 2-dp coincidence allowance: every rendering must be absent as a whole number token."""
    client = FakeClient(
        propose("momentum", {"lookback": 60}, "t1"),
        propose("mean_reversion", {"lookback": 20}, "t2"),
        propose("momentum", {"lookback": 250, "top_n": 1}, "t3"),
        done(),
    )
    result = agent.run_research(GOAL, max_iters=6, client=client)
    text = _serialise(client)
    assert not HOLDOUT_DATE_RE.search(text)
    for key, value in result["holdout_metrics"].items():
        for rendering in (f"{value:.2f}", f"{value:.4f}", repr(float(value))):
            assert not _token(rendering).search(text), (key, rendering)
    # Sanity that the check can see numbers at all: a validation metric IS present at 4 dp.
    val_sharpe = f"{result['best']['val_metrics']['sharpe']:.4f}"
    assert _token(val_sharpe).search(text)


# ---------------------------------------------------------------- a stop reason is not a loophole


def test_budget_stop_still_scores_the_holdout_exactly_once(offline, monkeypatch):
    engine = offline
    answers = iter([True, True, False])  # third turn is denied before any request is sent
    monkeypatch.setattr(budget, "allow", lambda est=0.0: next(answers))
    scores: list = []
    real_score = guardrails.score_holdout

    def spy(handle, strategy, params, eng="python"):
        scores.append((strategy, params))
        return real_score(handle, strategy, params, eng)

    monkeypatch.setattr(guardrails, "score_holdout", spy)

    client = FakeClient(
        propose("momentum", {"lookback": 60}, "t1"),
        propose("momentum", {"lookback": 120}, "t2"),
        done("never-reached"),
    )
    result = agent.run_research(GOAL, max_iters=5, client=client)
    assert result["stopped_because"] == "budget"
    assert len(client.calls) == 2  # the denied turn sent nothing
    assert len(result["history"]) == 2
    assert scores == [(result["best"]["strategy"], result["best"]["params"])]
    assert result["holdout_metrics"] == pytest.approx(
        _hand_holdout_metrics(result["best"]["strategy"], result["best"]["params"])
    )
    assert len(engine.calls) == 5


# ---------------------------------------------------------------- date smuggled by the model


def test_model_supplied_holdout_date_param_is_refused_and_mints_no_handle(offline, monkeypatch):
    engine = offline
    splits: list = []
    real_split = guardrails.split_data
    monkeypatch.setattr(
        guardrails, "split_data", lambda p=None: (splits.append(1), real_split(p))[1]
    )

    client = FakeClient(
        propose("momentum", {"lookback": 60, "start": "2023-01-01"}, "t1"),
        done("too-early"),
    )
    result = agent.run_research(GOAL, max_iters=2, client=client)
    assert result["stopped_because"] == "max_iters"
    (record,) = result["history"]
    assert record["error"] is not None and "start" in record["error"]
    assert record["train_metrics"] is None and record["val_metrics"] is None
    assert result["best"] is None and result["holdout_metrics"] is None
    assert splits == [] and engine.calls == []
    assert mcp_server._RESULTS == {}
    assert len(mcp_server._DATASETS) == 2
    for wide in mcp_server._DATASETS.values():
        assert wide.index.max() < HOLDOUT_CUTOFF
    # The only 2023+ string in the transcript is the model's own tool_use input echoed back as
    # the assistant turn; every RUNNER-authored message (role=user) is date-free.
    for call in client.calls:
        for message in call["messages"]:
            if message["role"] == "user":
                assert not HOLDOUT_DATE_RE.search(json.dumps(message))


# ---------------------------------------------------------------- public mode, adversarially


@pytest.mark.parametrize(
    "strategy, params, needle",
    [
        pytest.param("pairs", {"lookback": 60}, "not vetted", id="unvetted-strategy"),
        pytest.param("momentum", {"lookback": {"n": 60}}, "flat", id="nested-dict"),
        pytest.param("momentum", {"lookback": [60]}, "flat", id="nested-list"),
        pytest.param("momentum", {"exec": "1+1"}, "exec", id="exec-as-param"),
    ],
)
def test_public_mode_other_poison_shapes_never_reach_the_engine(
    public_offline, strategy, params, needle
):
    engine = public_offline
    client = FakeClient(
        propose(strategy, params, "bad"), propose("momentum", {"lookback": 60}, "ok"), done()
    )
    result = agent.run_research(GOAL, max_iters=4, client=client)
    bad, ok = result["history"]
    assert "public mode" in bad["error"].lower() and needle in bad["error"]
    assert ok["error"] is None
    # No engine call before the honest proposal; then train, validation, holdout.
    assert len(engine.calls) == 3
    assert all(c["lookback"] == 60 and c["strategy"] == "momentum" for c in engine.calls)
    assert result["stopped_because"] == "converged"


def test_gate_removed_whitelist_still_blocks_but_public_mode_wording_disappears(
    public_offline, monkeypatch
):
    """Defence in depth AND proof that the builder's 'names public mode' assertion is load-bearing.

    With ``assert_no_codegen`` no-op'd the poisoned param is still refused (by the whitelist), so
    nothing executes — but the recorded error no longer says "public mode". That is exactly the
    regression the builder's tests must catch, and here it is reproduced inside the suite.
    """
    engine = public_offline
    monkeypatch.setattr(guardrails, "assert_no_codegen", lambda action: None)
    client = FakeClient(propose("momentum", {"code": '__import__("os")'}, "bad"), done())
    result = agent.run_research(GOAL, max_iters=2, client=client)
    (record,) = result["history"]
    assert record["error"] is not None and "code" in record["error"]
    assert "public mode" not in record["error"].lower()  # the builder's assertion would fail
    assert engine.calls == [] and mcp_server._RESULTS == {}
    assert result["best"] is None and result["holdout_metrics"] is None


def test_public_mode_result_is_json_and_carries_no_frames_or_handles(public_offline):
    client = FakeClient(propose("momentum", {"lookback": 60}, "ok"), done())
    result = agent.run_research(GOAL, max_iters=3, client=client)
    text = json.dumps(result)  # would raise on a frame or a handle
    assert "HoldoutHandle" not in text and "ds_" not in text and "res_" not in text
    assert set(result) == {
        "best",
        "holdout_metrics",
        "history",
        "stopped_because",
        "spend_usd",
        "error",
    }
