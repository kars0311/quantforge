"""Rigor: the research agent must NOT be able to read the untouched holdout during iteration.

Guards against the core failure mode — an LLM p-hacking against the test set (RG-4 / SF-8). The
holdout is scored exactly once, at the end.

Scope of THIS file (week 7): the guard-level proof — ``HoldoutHandle`` is opaque (no ``__dict__``,
no frame in any slot, no prices in ``repr``, not picklable) and ``score_holdout`` is single-use,
including when the first run fails. The agent-loop assertion (that ``agent.run_research`` never
touches the handle and the runner scores it once after the loop) is added in week 8 alongside
``ai/agent.py``.
"""

from __future__ import annotations

import copy
import pickle
import re

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange
from quantforge.ai import guardrails, mcp_server
from quantforge.engine.python_engine import PythonEngine
from quantforge.metrics.performance import _KEYS


def _synthetic_long(k: int = 4, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2026-06-30", tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    wide = pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(k)])
    return interchange.to_long(wide, "prices")


@pytest.fixture
def split():
    prices = _synthetic_long()
    train, validation, handle = guardrails.split_data(prices)
    holdout_rows = prices[prices["date"] >= pd.Timestamp("2023-01-01", tz="UTC")]
    return train, validation, handle, holdout_rows


# ---------------------------------------------------------------- opacity (SF-8)


def test_handle_has_no_dict_and_no_frame_in_any_slot(split):
    _, _, handle, _ = split
    assert not hasattr(handle, "__dict__")
    with pytest.raises(TypeError):
        vars(handle)
    assert set(guardrails.HoldoutHandle.__slots__) == {
        "consumed",
        "n_days",
        "start",
        "end",
        "_score",
    }
    for slot in guardrails.HoldoutHandle.__slots__:
        value = getattr(handle, slot)
        assert not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)), slot
    # Nothing container-like: the handle cannot be iterated, indexed, or measured.
    for dunder in ("__iter__", "__getitem__", "__len__"):
        assert not hasattr(handle, dunder)


def test_repr_and_str_expose_dates_only_and_no_price_digits(split):
    _, _, handle, holdout_rows = split
    pattern = r"HoldoutHandle\(start=\d{4}-\d{2}-\d{2}, end=\d{4}-\d{2}-\d{2}, n_days=\d+, consumed=False\)"
    assert re.fullmatch(pattern, repr(handle))
    assert str(handle) == repr(handle)
    assert repr(handle) == (
        f"HoldoutHandle(start={handle.start}, end={handle.end}, "
        f"n_days={handle.n_days}, consumed=False)"
    )
    # No holdout close value, at any common precision, appears in the representation.
    text = repr(handle)
    for px in holdout_rows["close"].head(50):
        assert f"{px:.2f}" not in text and f"{px:.4f}" not in text and repr(float(px)) not in text


def test_handle_is_not_picklable_or_copyable(split):
    _, _, handle, _ = split
    with pytest.raises(TypeError, match="not picklable"):
        pickle.dumps(handle)
    with pytest.raises(TypeError):
        copy.deepcopy(handle)


def test_train_and_validation_never_contain_holdout_dates(split):
    train, validation, _, _ = split
    cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    assert (train.index < cutoff).all()
    assert (validation.index < cutoff).all()


# ---------------------------------------------------------------- one-shot scoring (RG-4)


def test_score_holdout_returns_metric_keys_then_refuses_a_second_call(split):
    _, _, handle, _ = split
    metrics = guardrails.score_holdout(handle, "momentum", {}, "python")
    assert list(metrics) == _KEYS
    assert all(isinstance(v, float) for v in metrics.values())
    assert handle.consumed is True
    assert "consumed=True" in repr(handle)
    with pytest.raises(guardrails.HoldoutAlreadyScored):
        guardrails.score_holdout(handle, "momentum", {}, "python")
    # ...regardless of strategy/params: the *handle* is spent, not the (strategy, params) pair.
    with pytest.raises(guardrails.HoldoutAlreadyScored):
        guardrails.score_holdout(handle, "mean_reversion", {"lookback": 10})


def test_failing_first_run_still_consumes_the_handle(split, monkeypatch):
    _, _, handle, _ = split

    class Boom:
        name = "boom"

        def run_backtest(self, prices, positions, params):
            raise RuntimeError("engine exploded mid-run")

    monkeypatch.setitem(mcp_server.ENGINES, "boom", Boom())
    with pytest.raises(RuntimeError, match="exploded"):
        guardrails.score_holdout(handle, "momentum", {}, "boom")
    assert handle.consumed is True
    with pytest.raises(guardrails.HoldoutAlreadyScored):
        guardrails.score_holdout(handle, "momentum", {}, "python")


def test_score_closure_returns_metrics_only_never_the_frame(split):
    # The only callable that can touch the holdout returns a plain dict of floats.
    _, _, handle, _ = split
    out = handle._score("momentum", {"lookback": 126, "top_n": 0}, "python")
    assert isinstance(out, dict) and list(out) == _KEYS
    assert not any(isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)) for v in out.values())


def test_score_holdout_uses_validated_params_and_fixed_costs(split, monkeypatch):
    _, _, handle, _ = split
    seen: dict = {}

    class Spy(PythonEngine):
        def run_backtest(self, prices, positions, params=None):
            seen.update(params or {})
            return super().run_backtest(prices, positions, params)

    monkeypatch.setitem(mcp_server.ENGINES, "spy", Spy())
    guardrails.score_holdout(handle, "momentum", {"lookback": 60}, "spy")
    assert seen["cost_bps"] == 10.0
    assert seen["strategy"] == "momentum"
    assert seen["lookback"] == 60 and seen["top_n"] == 0  # defaults merged in


def test_agent_cannot_access_holdout_during_iteration():
    """Placeholder name kept from the scaffold; the loop-level assertion lands in week 8.

    Until ``ai/agent.py`` exists this documents the contract the week-8 test will enforce: the
    agent receives only ``(train, validation)`` from ``split_data`` and ``score_holdout`` is called
    once by the runner after the loop. The guard-level proofs above already hold.
    """
    train, validation, handle = guardrails.split_data(_synthetic_long())
    assert isinstance(train, pd.DataFrame) and isinstance(validation, pd.DataFrame)
    assert isinstance(handle, guardrails.HoldoutHandle)
    assert not isinstance(handle, pd.DataFrame)
