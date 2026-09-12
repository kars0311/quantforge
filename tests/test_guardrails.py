"""Unit tests for ``quantforge.ai.guardrails`` (component 10): split, rate limit, public mode.

Everything runs offline on a synthetic long "prices" frame built with the smoke test's
``_synthetic_prices`` pattern and converted through ``interchange.to_long``, so the loader's
network path is never touched. Rate-limit state is redirected into ``tmp_path`` via
``AI_LEDGER_PATH`` (the state file is derived from the ledger path) and the clock is
monkeypatched via ``guardrails._now``.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange
from quantforge.ai import guardrails
from quantforge.data import loader

BOUNDS = loader.get_split_bounds()


def _synthetic_long(
    start="2010-01-01", end="2026-06-30", k: int = 4, seed: int = 0
) -> pd.DataFrame:
    """Long interchange 'prices' frame: geometric random walk on business days, tz-aware UTC."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end, tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    wide = pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(k)])
    return interchange.to_long(wide, "prices")


@pytest.fixture
def rate_state(tmp_path, monkeypatch):
    """Isolate rate-limit state in tmp_path; pin the limit to 3; return the state file path."""
    ledger = tmp_path / "cache" / "ai_ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "3")
    return ledger.with_name("ai_rate_limits.json")


# ---------------------------------------------------------------- public_mode


def test_public_mode_read_at_call_time_without_reimport(monkeypatch):
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    assert guardrails.public_mode() is False
    monkeypatch.setenv("PUBLIC_MODE", "on")
    assert guardrails.public_mode() is True
    monkeypatch.setenv("PUBLIC_MODE", "ON")
    assert guardrails.public_mode() is True
    monkeypatch.setenv("PUBLIC_MODE", "off")
    assert guardrails.public_mode() is False
    assert not hasattr(guardrails, "PUBLIC_MODE"), "import-time constant must be gone"


def test_constants_pinned():
    assert guardrails.MAX_AGENT_ITERS == 10
    assert guardrails.VETTED_STRATEGIES == {"momentum", "mean_reversion"}
    assert issubclass(guardrails.HoldoutAlreadyScored, RuntimeError)
    assert issubclass(guardrails.PublicModeViolation, ValueError)


# ---------------------------------------------------------------- split_data


def test_split_uses_fixed_bounds_and_is_disjoint_and_complete():
    prices = _synthetic_long()
    train, validation, handle = guardrails.split_data(prices)

    # Wide outputs are the shape strategies consume, and round-trip the contract.
    for wide in (train, validation):
        assert isinstance(wide.index, pd.DatetimeIndex) and wide.index.tz is not None
        interchange.validate_frame(interchange.to_long(wide, "prices"), "prices")

    lo, hi = (pd.Timestamp(b, tz="UTC") for b in BOUNDS["train"])
    assert train.index.min() >= lo and train.index.max() <= hi
    assert train.index.min().date().isoformat() == "2010-01-01"
    assert train.index.max().date().isoformat() == "2019-12-31"

    lo, hi = (pd.Timestamp(b, tz="UTC") for b in BOUNDS["validation"])
    assert validation.index.min() >= lo and validation.index.max() <= hi
    assert validation.index.min().date().isoformat() == "2020-01-01"
    # 2022-12-31 is a Saturday; the last business day is the 30th.
    assert validation.index.max().date().isoformat() == "2022-12-30"

    # Disjoint, and together they account for every in-range date.
    assert train.index.intersection(validation.index).empty
    n_holdout_dates = prices.loc[
        prices["date"] >= pd.Timestamp("2023-01-01", tz="UTC"), "date"
    ].nunique()
    assert handle.n_days == n_holdout_dates
    assert train.shape[0] + validation.shape[0] + handle.n_days == prices["date"].nunique()
    assert handle.start == "2023-01-02" and handle.end == "2026-06-30"
    assert handle.consumed is False


def test_split_drops_rows_outside_the_fixed_range():
    # Rows before 2010 and after END are not in any slice — there is one split definition and
    # nothing outside it is ever seen (RG-4).
    prices = _synthetic_long(start="2008-01-01", end="2027-12-31")
    train, validation, handle = guardrails.split_data(prices)
    in_range = prices["date"].between(
        pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")
    )
    assert (
        train.shape[0] + validation.shape[0] + handle.n_days
        == prices.loc[in_range, "date"].nunique()
    )
    assert train.index.min().date().isoformat() == "2010-01-01"
    assert handle.end == "2026-06-30"


def test_split_has_no_ratio_arguments():
    params = inspect.signature(guardrails.split_data).parameters
    assert list(params) == ["prices"]


def test_split_validates_contract():
    bad = _synthetic_long().rename(columns={"close": "px"})
    with pytest.raises(interchange.SchemaError):
        guardrails.split_data(bad)


def test_empty_holdout_yields_zero_day_handle_that_refuses_to_score():
    prices = _synthetic_long(start="2010-01-01", end="2021-12-31")
    train, validation, handle = guardrails.split_data(prices)
    assert handle.n_days == 0
    assert train.shape[0] > 0 and validation.shape[0] > 0
    with pytest.raises(ValueError, match="holdout slice is empty"):
        guardrails.score_holdout(handle, "momentum", {}, "python")


def test_score_holdout_rejects_bad_inputs_without_spending_the_shot():
    _, _, handle = guardrails.split_data(_synthetic_long())
    with pytest.raises(ValueError):
        guardrails.score_holdout(handle, "pairs", {})
    with pytest.raises(ValueError):
        guardrails.score_holdout(handle, "momentum", {"lookback": 999})
    with pytest.raises(ValueError, match="unknown engine"):
        guardrails.score_holdout(handle, "momentum", {}, "matlab")
    with pytest.raises(TypeError):
        guardrails.score_holdout(object(), "momentum", {})  # type: ignore[arg-type]
    assert handle.consumed is False


# ---------------------------------------------------------------- rate_limit


def test_rate_limit_boundary_allows_limit_then_denies(rate_state):
    assert [guardrails.rate_limit("k") for _ in range(3)] == [True, True, True]
    assert guardrails.rate_limit("k") is False
    assert guardrails.rate_limit("k") is False
    # Denials are not recorded: exactly `limit` timestamps on disk.
    assert len(json.loads(rate_state.read_text())["k"]) == 3


def test_rate_limit_keys_are_independent(rate_state):
    for _ in range(3):
        assert guardrails.rate_limit("k") is True
    assert guardrails.rate_limit("k") is False
    assert guardrails.rate_limit("other") is True


def test_rate_limit_window_expires_via_clock(rate_state, monkeypatch):
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(guardrails, "_now", lambda: clock["t"])
    for _ in range(3):  # calls at t, t+1, t+2
        assert guardrails.rate_limit("k") is True
        clock["t"] += 1.0
    assert guardrails.rate_limit("k") is False
    clock["t"] = 1_000_000.0 + 3599.0
    assert guardrails.rate_limit("k") is False  # first call is 3599 s old: still in the window
    clock["t"] = 1_000_000.0 + 3600.0  # exactly 3600 s after the first call -> only it is pruned
    assert guardrails.rate_limit("k") is True
    assert guardrails.rate_limit("k") is False  # t+1 and t+2 calls are still inside


def test_rate_limit_persists_across_a_fresh_process(rate_state):
    # A fresh interpreter (not ``importlib.reload``, which would rebind the module's exception
    # classes under every other test) must see the three calls this process recorded: a
    # Streamlit restart cannot reset anyone's hourly count.
    for _ in range(3):
        assert guardrails.rate_limit("k") is True
    assert rate_state.exists()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from quantforge.ai import guardrails; print(guardrails.rate_limit('k'))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


def test_rate_limit_reads_env_at_call_time(rate_state, monkeypatch):
    assert guardrails.rate_limit("k") is True
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "1")
    assert guardrails.rate_limit("k") is False
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "not-a-number")  # -> default 20
    assert guardrails.rate_limit("k") is True


@pytest.mark.parametrize("bad", ["", None, 3, b"k"])
def test_rate_limit_rejects_bad_keys(rate_state, bad):
    with pytest.raises(ValueError):
        guardrails.rate_limit(bad)  # type: ignore[arg-type]


def test_rate_limit_state_file_sits_beside_ledger_and_leaves_no_temp_files(rate_state):
    guardrails.rate_limit("k")
    names = sorted(p.name for p in rate_state.parent.iterdir())
    assert names == ["ai_rate_limits.json", "ai_rate_limits.json.lock"]


def test_rate_limit_corrupt_state_treated_as_empty(rate_state):
    rate_state.parent.mkdir(parents=True, exist_ok=True)
    rate_state.write_text("{not json")
    assert guardrails.rate_limit("k") is True


# ---------------------------------------------------------------- assert_no_codegen (dev)


def test_assert_no_codegen_is_noop_outside_public_mode(monkeypatch):
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    assert guardrails.assert_no_codegen({"code": "import os"}) is None
    assert guardrails.assert_no_codegen({"strategy": "pairs", "params": {"x": [1]}}) is None
    assert guardrails.assert_no_codegen("not even a dict") is None  # type: ignore[arg-type]
