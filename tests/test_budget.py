"""Unit tests for ``quantforge.ai.budget`` — the spend ledger + kill-switch (component 09).

Every test redirects ``AI_LEDGER_PATH`` into ``tmp_path`` so the repo's real ``data_cache/`` is
never touched, and pins the caps / PUBLIC_MODE / AI_DISABLED env explicitly so a developer's
``.env`` cannot leak into the assertions. The clock is monkeypatched via ``budget._utc_today``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import threading

import pytest

from quantforge.ai import budget

HAIKU = "claude-haiku-4-5"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Point the module at a fresh ledger and a clean dev-mode environment; return its path."""
    path = tmp_path / "ledger" / "ai_ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(path))
    monkeypatch.setenv("PUBLIC_MODE", "off")
    monkeypatch.delenv("AI_DISABLED", raising=False)
    monkeypatch.delenv("AI_BUDGET_USD_DAILY", raising=False)
    monkeypatch.delenv("AI_BUDGET_USD_TOTAL", raising=False)
    return path


def _set_caps(monkeypatch, daily: float, total: float) -> None:
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", str(daily))
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", str(total))


# ---------------------------------------------------------------- estimate


def test_estimate_is_hand_computable():
    # 1M input tokens at $1/MTok + 200k output tokens at $5/MTok = $1 + $1.
    assert budget.estimate(HAIKU, 1_000_000, 200_000) == 2.0
    assert budget.estimate("claude-sonnet-5", 500_000, 100_000) == pytest.approx(1.0 + 1.0)
    assert budget.estimate("claude-opus-5", 0, 0) == 0.0


def test_estimate_unknown_model_raises_and_names_known_models():
    with pytest.raises(ValueError, match="claude-haiku-4-5"):
        budget.estimate("claude-opus-5-20260401", 10, 10)


@pytest.mark.parametrize("bad_in, bad_out", [(-1, 0), (0, -1), (True, 0), (0, False), (1.5, 0)])
def test_estimate_rejects_bad_token_counts(bad_in, bad_out):
    with pytest.raises(ValueError):
        budget.estimate(HAIKU, bad_in, bad_out)


def test_prices_use_bare_model_ids():
    assert all("-20" not in model for model in budget.PRICES_PER_MTOK), "no date-suffixed IDs"


# ---------------------------------------------------------------- allow / caps


def test_daily_cap_boundary_is_strict(ledger, monkeypatch):
    _set_caps(monkeypatch, daily=1.0, total=100.0)
    assert budget.allow(1.0) is True  # exactly on the cap is allowed
    budget.charge(0.7, model=HAIKU, tokens_in=1, tokens_out=1)
    assert budget.allow(0.3) is True
    assert budget.allow(0.3000001) is False
    assert budget.allow(0.31) is False


def test_total_cap_enforced_across_days(ledger, monkeypatch):
    _set_caps(monkeypatch, daily=10.0, total=1.0)
    monkeypatch.setattr(budget, "_utc_today", lambda: "2026-01-01")
    budget.charge(0.6, model=HAIKU, tokens_in=1, tokens_out=1)
    monkeypatch.setattr(budget, "_utc_today", lambda: "2026-01-02")
    assert budget.allow(0.4) is True  # daily is fresh, total has exactly 0.4 left
    assert budget.allow(0.41) is False  # daily fine; total cap bites


def test_utc_rollover_resets_daily_but_not_total(ledger, monkeypatch):
    _set_caps(monkeypatch, daily=1.0, total=10.0)
    monkeypatch.setattr(budget, "_utc_today", lambda: "2026-03-01")
    budget.charge(0.9, model=HAIKU, tokens_in=10, tokens_out=20)
    assert budget.allow(0.2) is False
    assert budget.remaining() == pytest.approx({"daily": 0.1, "total": 9.1})

    monkeypatch.setattr(budget, "_utc_today", lambda: "2026-03-02")
    assert budget.allow(0.2) is True
    assert budget.remaining() == pytest.approx({"daily": 1.0, "total": 9.1})

    on_disk = json.loads(ledger.read_text())
    assert set(on_disk["days"]) == {"2026-03-01"}
    assert on_disk["days"]["2026-03-01"] == {
        "usd": 0.9,
        "calls": 1,
        "tokens_in": 10,
        "tokens_out": 20,
    }


def test_kill_switch_denies_even_with_zero_spend(ledger, monkeypatch):
    _set_caps(monkeypatch, daily=100.0, total=100.0)
    assert budget.allow(0.0) is True
    monkeypatch.setenv("AI_DISABLED", "on")
    assert budget.allow(0.0) is False
    monkeypatch.setenv("AI_DISABLED", "ON")  # case-insensitive
    assert budget.allow(0.0) is False
    monkeypatch.setenv("AI_DISABLED", "off")
    assert budget.allow(0.0) is True


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), "1.0", None, True])
def test_allow_never_raises_on_bad_estimate(ledger, monkeypatch, bad):
    _set_caps(monkeypatch, daily=100.0, total=100.0)
    assert budget.allow(bad) is False


def test_public_mode_with_caps_unset_denies(ledger, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    assert budget.allow(0.0) is False
    assert budget.remaining() == {"daily": 0.0, "total": 0.0}


def test_public_mode_with_unparseable_cap_denies(ledger, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "five dollars")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "25")
    assert budget.allow(0.0) is False


def test_dev_mode_with_caps_unset_uses_generous_defaults(ledger):
    assert budget.allow(2.0) is True
    assert budget.allow(2.0000001) is False
    assert budget.remaining() == {"daily": 2.0, "total": 10.0}


def test_caps_are_read_at_call_time(ledger, monkeypatch):
    _set_caps(monkeypatch, daily=1.0, total=1.0)
    assert budget.allow(0.5) is True
    _set_caps(monkeypatch, daily=0.1, total=1.0)
    assert budget.allow(0.5) is False


# ---------------------------------------------------------------- corrupt ledger policy


def test_corrupt_ledger_fails_closed_in_public_mode(ledger, monkeypatch):
    _set_caps(monkeypatch, daily=100.0, total=100.0)
    ledger.parent.mkdir(parents=True)
    ledger.write_text("{")
    monkeypatch.setenv("PUBLIC_MODE", "on")
    assert budget.allow(0.0) is False  # and does not raise
    assert budget.remaining() == {"daily": 0.0, "total": 0.0}
    with pytest.raises(budget.LedgerCorruptError):
        budget.charge(0.1, model=HAIKU, tokens_in=1, tokens_out=1)
    assert ledger.read_text() == "{", "public mode must not overwrite a corrupt ledger"


def test_corrupt_ledger_is_empty_plus_warning_in_dev(ledger, monkeypatch, caplog):
    _set_caps(monkeypatch, daily=100.0, total=100.0)
    ledger.parent.mkdir(parents=True)
    ledger.write_text("{")
    with caplog.at_level(logging.WARNING, logger="quantforge.ai.budget"):
        assert budget.allow(0.0) is True
    assert any("corrupt" in rec.getMessage() for rec in caplog.records)


def test_wrong_shape_ledger_counts_as_corrupt(ledger, monkeypatch):
    _set_caps(monkeypatch, daily=100.0, total=100.0)
    monkeypatch.setenv("PUBLIC_MODE", "on")
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"days": {"2026-01-01": {"usd": "lots"}}}))
    assert budget.allow(0.0) is False


# ---------------------------------------------------------------- persistence


def test_ledger_persists_across_module_reload(ledger, monkeypatch):
    _set_caps(monkeypatch, daily=5.0, total=5.0)
    monkeypatch.setattr(budget, "_utc_today", lambda: "2026-06-30")
    budget.charge(1.25, model=HAIKU, tokens_in=100, tokens_out=50)
    budget.charge(0.25, model="claude-sonnet-5", tokens_in=7, tokens_out=3)

    reloaded = importlib.reload(budget)
    monkeypatch.setattr(reloaded, "_utc_today", lambda: "2026-06-30")
    assert reloaded.remaining() == pytest.approx({"daily": 3.5, "total": 3.5})

    on_disk = json.loads(ledger.read_text())
    assert on_disk == {
        "days": {"2026-06-30": {"usd": 1.5, "calls": 2, "tokens_in": 107, "tokens_out": 53}}
    }
    assert not list(ledger.parent.glob("*.tmp")), "atomic write must not leave temp files"


def test_remaining_never_negative(ledger, monkeypatch):
    _set_caps(monkeypatch, daily=1.0, total=1.0)
    budget.charge(1.5, model=HAIKU, tokens_in=1, tokens_out=1)  # charge records reality
    assert budget.remaining() == {"daily": 0.0, "total": 0.0}
    assert budget.allow(0.0) is False


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), "0.1", True])
def test_charge_rejects_bad_usd(ledger, bad):
    with pytest.raises(ValueError):
        budget.charge(bad, model=HAIKU, tokens_in=1, tokens_out=1)


def test_charge_rejects_bad_tokens_and_model(ledger):
    with pytest.raises(ValueError):
        budget.charge(0.1, model=HAIKU, tokens_in=-1, tokens_out=1)
    with pytest.raises(ValueError):
        budget.charge(0.1, model="", tokens_in=1, tokens_out=1)


def test_concurrent_charges_sum_exactly(ledger, monkeypatch):
    _set_caps(monkeypatch, daily=100.0, total=100.0)
    n_threads, per_thread, amount = 8, 10, 0.125  # 0.125 is exact in binary -> exact sum

    def worker() -> None:
        for _ in range(per_thread):
            budget.charge(amount, model=HAIKU, tokens_in=3, tokens_out=2)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    bucket = json.loads(ledger.read_text())["days"][budget._utc_today()]
    assert bucket["usd"] == n_threads * per_thread * amount == 10.0
    assert bucket["calls"] == n_threads * per_thread
    assert bucket["tokens_in"] == n_threads * per_thread * 3
    assert bucket["tokens_out"] == n_threads * per_thread * 2


def test_module_never_imports_anthropic():
    assert "anthropic" not in inspect.getsource(budget).lower()
