"""Verifier tests: ``ai/guardrails`` (week 7, component 10 — RG-4 / SF-3 / SF-5 / SF-8).

Independently re-proves the builder's milestone against docs/components/10-ai-guardrails.md, with
adversarial cases the builder's own tests do not cover:

* **The closure scores the RIGHT slice.** A spy engine captures the frame ``score_holdout`` feeds
  it and we check it is byte-identical to the 2023+ rows of the source panel — no train/validation
  row leaks into the holdout score, and no holdout row is missing from it.
* **Hand-reproduced metrics.** The one-shot score must equal what a human gets by running the same
  strategy + ``PythonEngine`` on the same wide slice with ``cost_bps=10`` — the guard adds
  opacity, not a different number.
* **Engine resolution really goes through ``mcp_server.ENGINES``.** Emptying the registry makes
  even ``"python"`` unknown, and a caller mistake never spends the shot.
* **Rate limiter under contention.** Twelve threads race on one key with limit 5; exactly five
  win. Denials never extend the lockout; a limit of 0 denies everything; stale stamps on disk are
  pruned, fresh ones honored.
* **Public-mode lookalikes.** ``True`` for an int, ``NaN`` for a float, a case-mismatched vetted
  name, an extra key whose value is ``None`` — all rejected.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange
from quantforge.ai import guardrails, mcp_server
from quantforge.data import loader
from quantforge.engine.python_engine import PythonEngine
from quantforge.metrics.performance import _KEYS
from quantforge.strategies import STRATEGIES, validate_params
from quantforge.strategies.momentum import MomentumStrategy

UTC = "UTC"


def _synthetic_long(
    start="2010-01-01", end="2026-06-30", k: int = 4, seed: int = 11
) -> pd.DataFrame:
    """Long interchange 'prices' frame spanning the full fixed range (smoke-test pattern)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end, tz=UTC)
    rets = rng.normal(0.0004, 0.01, size=(len(dates), k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    wide = pd.DataFrame(prices, index=dates, columns=[f"V{i}" for i in range(k)])
    return interchange.to_long(wide, "prices")


def _holdout_wide(prices: pd.DataFrame) -> pd.DataFrame:
    """What the holdout slice SHOULD be, computed independently of guardrails."""
    lo, hi = (pd.Timestamp(b, tz=UTC) for b in loader.get_split_bounds()["holdout"])
    rows = prices[prices["date"].between(lo, hi)].reset_index(drop=True)
    return interchange.to_wide(rows, "prices")


@pytest.fixture
def rate_env(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger" / "ai_ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "3")
    return ledger.with_name("ai_rate_limits.json")


# ---------------------------------------------------------------- check 1: public_mode at call time


def test_public_mode_flips_with_env_without_reimport(monkeypatch):
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    assert guardrails.public_mode() is False
    monkeypatch.setenv("PUBLIC_MODE", "on")
    assert guardrails.public_mode() is True
    monkeypatch.setenv("PUBLIC_MODE", " On ")
    assert guardrails.public_mode() is True
    for other in ("off", "true", "1", "yes", ""):
        monkeypatch.setenv("PUBLIC_MODE", other)
        assert guardrails.public_mode() is False, other
    assert not hasattr(guardrails, "PUBLIC_MODE")


# ---------------------------------------------------------------- check 2: split bounds


def test_split_ranges_are_exactly_the_fixed_bounds():
    prices = _synthetic_long()
    train, validation, handle = guardrails.split_data(prices)

    assert train.index.min() == pd.Timestamp("2010-01-01", tz=UTC)
    assert train.index.max() == pd.Timestamp("2019-12-31", tz=UTC)
    assert validation.index.min() == pd.Timestamp("2020-01-01", tz=UTC)
    # Inclusive bound is 2022-12-31 (a Saturday); the last business day is the 30th.
    assert validation.index.max() == pd.Timestamp("2022-12-30", tz=UTC)
    assert validation.index.max() <= pd.Timestamp("2022-12-31", tz=UTC)

    n_holdout = int((prices["date"] >= pd.Timestamp("2023-01-01", tz=UTC)).sum() // 4)
    assert handle.n_days == n_holdout
    assert train.shape[0] + validation.shape[0] + handle.n_days == prices["date"].nunique()
    assert train.index.intersection(validation.index).empty
    # Wide outputs are the contract's wide form (index=date, columns=ticker), all 4 tickers.
    assert list(train.columns) == [f"V{i}" for i in range(4)]
    assert train.index.name == "date" and train.columns.name == "ticker"


def test_split_slices_are_exact_row_subsets_of_the_source():
    # Not just the right date range: the VALUES in train/validation are the source's values.
    prices = _synthetic_long()
    train, validation, _ = guardrails.split_data(prices)
    src = interchange.to_wide(prices, "prices")
    pd.testing.assert_frame_equal(train, src.loc[train.index])
    pd.testing.assert_frame_equal(validation, src.loc[validation.index])


def test_split_defaults_to_loader_and_rejects_wrong_kind(monkeypatch):
    called = {"n": 0}

    def fake_load_prices():
        called["n"] += 1
        return _synthetic_long(end="2020-06-30")

    monkeypatch.setattr(loader, "load_prices", fake_load_prices)
    train, validation, handle = guardrails.split_data()
    assert called["n"] == 1 and handle.n_days == 0 and validation.shape[0] > 0

    # A wide frame (the wrong shape) is a schema error, not a KeyError deep in slicing.
    with pytest.raises(interchange.SchemaError):
        guardrails.split_data(interchange.to_wide(_synthetic_long(end="2011-01-01"), "prices"))


# ---------------------------------------------------------------- check 3: opacity


def test_handle_is_opaque_unpicklable_and_repr_is_metadata_only():
    prices = _synthetic_long()
    _, _, handle = guardrails.split_data(prices)
    assert hasattr(handle, "__dict__") is False
    with pytest.raises(TypeError, match="not picklable"):
        pickle.dumps(handle)
    with pytest.raises(TypeError):
        pickle.dumps(handle, protocol=0)
    assert re.fullmatch(
        r"HoldoutHandle\(start=2023-01-02, end=2026-06-30, n_days=\d+, consumed=False\)",
        repr(handle),
    )
    assert str(handle) == repr(handle)
    # Every public attribute is a plain scalar; the only callable is the scorer.
    assert isinstance(handle.n_days, int) and isinstance(handle.start, str)
    assert not isinstance(handle.consumed, pd.DataFrame)
    for name in dir(handle):
        if name.startswith("__"):
            continue
        assert not isinstance(getattr(handle, name), (pd.DataFrame, pd.Series, np.ndarray)), name
    for dunder in ("__iter__", "__getitem__", "__len__", "__contains__"):
        assert not hasattr(handle, dunder)
    assert "HoldoutHandle" in repr(handle) and "close" not in repr(handle)


# ---------------------------------------------------------------- check 4: one-shot scoring


def test_score_holdout_returns_metric_keys_once_then_raises():
    _, _, handle = guardrails.split_data(_synthetic_long())
    metrics = guardrails.score_holdout(handle, "momentum", {}, "python")
    assert set(metrics) == set(_KEYS) and list(metrics) == _KEYS
    assert all(isinstance(v, float) and math.isfinite(v) for v in metrics.values())
    assert handle.consumed is True
    with pytest.raises(guardrails.HoldoutAlreadyScored):
        guardrails.score_holdout(handle, "momentum", {}, "python")
    # Default engine argument is "python".
    _, _, h2 = guardrails.split_data(_synthetic_long(seed=3))
    assert list(guardrails.score_holdout(h2, "mean_reversion", {"lookback": 10})) == _KEYS


def test_score_holdout_feeds_the_engine_exactly_the_holdout_slice(monkeypatch):
    """Adversarial leak check: the frame the engine sees IS the 2023+ rows — nothing more, less."""
    prices = _synthetic_long()
    expected = _holdout_wide(prices)
    seen: dict = {}

    class Spy(PythonEngine):
        def run_backtest(self, prices, positions, params=None):
            seen["frame"] = prices.copy()
            seen["positions"] = positions.copy()
            seen["params"] = dict(params or {})
            return super().run_backtest(prices, positions, params)

    monkeypatch.setitem(mcp_server.ENGINES, "spy", Spy())
    _, _, handle = guardrails.split_data(prices)
    guardrails.score_holdout(handle, "momentum", {"lookback": 40, "top_n": 2}, "spy")

    pd.testing.assert_frame_equal(seen["frame"], expected)
    assert seen["frame"].index.min() == pd.Timestamp("2023-01-02", tz=UTC)
    assert seen["frame"].shape[0] == handle.n_days
    assert seen["params"] == {"cost_bps": 10.0, "strategy": "momentum", "lookback": 40, "top_n": 2}
    assert seen["positions"].shape == expected.shape


def test_score_holdout_equals_hand_run_of_strategy_plus_engine():
    """The guard adds opacity, not a different number: reproduce the score by hand."""
    prices = _synthetic_long(seed=5)
    holdout = _holdout_wide(prices)
    params = {"lookback": 60, "top_n": 2}
    validated = validate_params("momentum", params)
    positions = MomentumStrategy().generate_signals(holdout, validated)
    expected = (
        PythonEngine()
        .run_backtest(holdout, positions, {"cost_bps": 10.0, "strategy": "momentum", **validated})
        .metrics
    )

    _, _, handle = guardrails.split_data(prices)
    got = guardrails.score_holdout(handle, "momentum", params, "python")
    for key in _KEYS:
        assert got[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-15), key
    # Costs are really charged: the same run with zero costs is not identical.
    free = PythonEngine().run_backtest(holdout, positions, {"cost_bps": 0.0}).metrics
    assert free["total_return"] > got["total_return"]


def test_engine_resolution_goes_through_mcp_server_registry(monkeypatch):
    _, _, handle = guardrails.split_data(_synthetic_long())
    monkeypatch.setattr(mcp_server, "ENGINES", {})
    with pytest.raises(ValueError, match="unknown engine"):
        guardrails.score_holdout(handle, "momentum", {}, "python")
    assert handle.consumed is False  # a caller mistake never spends the shot
    monkeypatch.setattr(mcp_server, "ENGINES", {"python": PythonEngine()})
    assert list(guardrails.score_holdout(handle, "momentum", {})) == _KEYS


def test_exploding_strategy_on_first_run_still_consumes(monkeypatch):
    _, _, handle = guardrails.split_data(_synthetic_long())

    class Exploding(MomentumStrategy):
        def generate_signals(self, prices, params):
            raise RuntimeError("strategy blew up")

    monkeypatch.setitem(STRATEGIES, "momentum", Exploding)
    with pytest.raises(RuntimeError, match="blew up"):
        guardrails.score_holdout(handle, "momentum", {})
    assert handle.consumed is True
    monkeypatch.setitem(STRATEGIES, "momentum", MomentumStrategy)
    with pytest.raises(guardrails.HoldoutAlreadyScored):
        guardrails.score_holdout(handle, "momentum", {})


def test_score_holdout_does_not_mutate_caller_params():
    _, _, handle = guardrails.split_data(_synthetic_long())
    params = {"lookback": 30}
    guardrails.score_holdout(handle, "momentum", params)
    assert params == {"lookback": 30}


def test_empty_holdout_handle_reports_nominal_bounds_and_refuses():
    _, _, handle = guardrails.split_data(_synthetic_long(end="2022-12-31"))
    assert handle.n_days == 0 and handle.start == "2023-01-01" and handle.end == "2026-06-30"
    with pytest.raises(ValueError, match="holdout slice is empty"):
        guardrails.score_holdout(handle, "momentum", {})
    # The refusal happened inside the shot: an empty holdout is still single-use.
    assert handle.consumed is True


# ---------------------------------------------------------------- check 5: rate limit


def test_rate_limit_boundary_and_key_independence(rate_env):
    assert [guardrails.rate_limit("k") for _ in range(4)] == [True, True, True, False]
    assert guardrails.rate_limit("other") is True
    state = json.loads(rate_env.read_text())
    assert set(state) == {"k", "other"} and len(state["k"]) == 3 and len(state["other"]) == 1
    assert all(isinstance(t, float) for t in state["k"])


def test_rate_limit_denials_do_not_extend_the_lockout(rate_env, monkeypatch):
    clock = {"t": 2_000_000.0}
    monkeypatch.setattr(guardrails, "_now", lambda: clock["t"])
    for _ in range(3):
        assert guardrails.rate_limit("k") is True
    for dt in (100.0, 1000.0, 3000.0):  # hammering while locked out
        clock["t"] = 2_000_000.0 + dt
        assert guardrails.rate_limit("k") is False
    clock["t"] = 2_000_000.0 + 3600.0  # one hour after the ORIGINAL calls, not the last denial
    assert guardrails.rate_limit("k") is True


def test_rate_limit_zero_denies_everything_and_records_nothing(rate_env, monkeypatch):
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "0")
    assert guardrails.rate_limit("k") is False
    assert json.loads(rate_env.read_text())["k"] == []


def test_rate_limit_honors_fresh_stamps_and_prunes_stale_ones_from_disk(rate_env, monkeypatch):
    now = 3_000_000.0
    monkeypatch.setattr(guardrails, "_now", lambda: now)
    rate_env.parent.mkdir(parents=True, exist_ok=True)
    rate_env.write_text(json.dumps({"k": [now - 10, now - 20, now - 30], "stale": [now - 7200]}))
    assert guardrails.rate_limit("k") is False  # three fresh stamps from a "previous process"
    assert guardrails.rate_limit("stale") is True
    assert json.loads(rate_env.read_text())["stale"] == [now]  # the 2-hour-old stamp is gone


def test_rate_limit_is_atomic_under_thread_contention(rate_env, monkeypatch):
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "5")
    results: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(12)

    def worker():
        barrier.wait()
        ok = guardrails.rate_limit("shared")
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 5 and len(results) == 12
    assert len(json.loads(rate_env.read_text())["shared"]) == 5


def test_rate_limit_state_lives_beside_the_ledger(rate_env):
    guardrails.rate_limit("k")
    assert guardrails._rate_state_path() == rate_env
    assert rate_env.parent == Path(rate_env).parent and rate_env.name == "ai_rate_limits.json"


# ---------------------------------------------------------------- check 6: assert_no_codegen


@pytest.fixture
def public(monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")


def test_public_mode_gate_verifier_checks(public):
    V = guardrails.PublicModeViolation
    with pytest.raises(V):
        guardrails.assert_no_codegen({"strategy": "momentum", "params": {"code": "x"}})
    with pytest.raises(V):
        guardrails.assert_no_codegen({"strategy": "momentum", "params": {}, "exec": "x"})
    assert (
        guardrails.assert_no_codegen({"strategy": "momentum", "params": {"lookback": 50}}) is None
    )


@pytest.mark.parametrize(
    "action",
    [
        pytest.param({"strategy": "momentum", "params": {"lookback": True}}, id="bool-lookalike"),
        pytest.param({"strategy": "momentum", "params": {"lookback": 126.5}}, id="fractional-int"),
        pytest.param(
            {"strategy": "mean_reversion", "params": {"entry_z": float("nan")}}, id="nan-float"
        ),
        pytest.param(
            {"strategy": "mean_reversion", "params": {"entry_z": float("inf")}}, id="inf-float"
        ),
        pytest.param({"strategy": "Momentum", "params": {}}, id="case-mismatch"),
        pytest.param({"strategy": "momentum", "params": {}, "code": None}, id="extra-key-none"),
        pytest.param({"strategy": "momentum", "params": {}, "Strategy": "x"}, id="extra-key-case"),
        pytest.param({"strategy": "momentum", "params": {1: 126}}, id="non-str-param-name"),
        pytest.param({"strategy": "momentum", "params": {"lookback": None}}, id="none-value"),
        pytest.param({"strategy": "momentum", "params": {"lookback": (126,)}}, id="tuple-value"),
        pytest.param({"strategy": None, "params": {}}, id="none-strategy"),
        pytest.param([("strategy", "momentum"), ("params", {})], id="list-of-pairs"),
        pytest.param(
            {"strategy": "mean_reversion", "params": {"mode": "short_only"}}, id="bad-choice"
        ),
    ],
)
def test_public_mode_rejects_lookalikes(public, action):
    with pytest.raises(guardrails.PublicModeViolation):
        guardrails.assert_no_codegen(action)


def test_public_mode_accepts_json_style_int_as_float(public):
    # JSON round-trips often turn 126 into 126.0; validate_params accepts that, so must the gate.
    assert (
        guardrails.assert_no_codegen({"strategy": "momentum", "params": {"lookback": 126.0}})
        is None
    )


def test_public_mode_gate_is_call_time(monkeypatch):
    bad = {"strategy": "momentum", "params": {}, "exec": "x"}
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    assert guardrails.assert_no_codegen(bad) is None
    monkeypatch.setenv("PUBLIC_MODE", "on")
    with pytest.raises(guardrails.PublicModeViolation):
        guardrails.assert_no_codegen(bad)
    monkeypatch.setenv("PUBLIC_MODE", "off")
    assert guardrails.assert_no_codegen(bad) is None


# ---------------------------------------------------------------- checks 7 & 8: registry + env


def test_vetted_set_matches_registry_and_env_example_lists_rate_limit():
    assert guardrails.VETTED_STRATEGIES == set(STRATEGIES)
    assert guardrails.MAX_AGENT_ITERS == 10
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    assert re.search(r"^AI_RATE_LIMIT_PER_HOUR=20\s*$", text, flags=re.M)
