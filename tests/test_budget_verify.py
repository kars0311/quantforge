"""Independent verifier tests for ``quantforge.ai.budget`` (component 09).

Written adversarially against the builder's implementation: seeded ledgers with surprising
contents, environment flipped mid-test, unreadable paths, float accumulation across many small
charges, and a guard that the repo's real ``data_cache/`` is never touched. Every test redirects
``AI_LEDGER_PATH`` into ``tmp_path`` and pins the budget env explicitly.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest

from quantforge.ai import budget

HAIKU = "claude-haiku-4-5"
REPO_ROOT = Path(__file__).resolve().parents[1]

_REPO_LEDGER_FILES = ("ai_ledger.json", "ai_ledger.json.lock", "ai_rate_limits.json")


def _repo_ledger_state() -> dict[str, tuple[int, int] | None]:
    """``{name: (size, mtime_ns) | None}`` for the real ``data_cache/`` ledger files."""
    state: dict[str, tuple[int, int] | None] = {}
    for name in _REPO_LEDGER_FILES:
        path = REPO_ROOT / "data_cache" / name
        state[name] = (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None
    return state


# Captured at collection time, i.e. before any test in the session runs. The hygiene test below
# compares against this rather than asserting the files are absent: on a developer's machine a
# real AI query legitimately writes ``data_cache/ai_ledger.json`` (that is where dev-mode spend
# is recorded), and a suite that goes red forever after the first real call would just get
# deleted. What must hold is that the *test run* neither created nor modified these files.
_REPO_LEDGER_STATE_AT_COLLECTION = _repo_ledger_state()


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Fresh ledger path in tmp_path + a clean dev-mode env with generous caps."""
    path = tmp_path / "ledger" / "ai_ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(path))
    monkeypatch.setenv("PUBLIC_MODE", "off")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "100")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "100")
    monkeypatch.delenv("AI_DISABLED", raising=False)
    return path


def _seed(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger))


# ---------------------------------------------------------------- estimate / prices


def test_prices_match_claude_api_skill_table():
    # USD per MTok (input, output) from the claude-api skill's model table; bare IDs only.
    assert budget.PRICES_PER_MTOK == {
        "claude-haiku-4-5": (1.0, 5.0),
        "claude-sonnet-5": (2.0, 10.0),
        "claude-opus-5": (5.0, 25.0),
    }


def test_estimate_opus_hand_computed_odd_numbers():
    # 123,456 in * $5/MTok = 0.61728 ; 7,890 out * $25/MTok = 0.19725 ; sum 0.81453
    assert budget.estimate("claude-opus-5", 123_456, 7_890) == pytest.approx(0.81453)
    # Output tokens are 5x input for every model: same count must cost 5x.
    assert budget.estimate(HAIKU, 0, 1_000) == pytest.approx(5 * budget.estimate(HAIKU, 1_000, 0))


def test_estimate_unknown_model_message_names_every_known_model():
    with pytest.raises(ValueError) as excinfo:
        budget.estimate("gpt-4", 1, 1)
    for known in budget.PRICES_PER_MTOK:
        assert known in str(excinfo.value)


def test_estimate_rejects_string_token_counts():
    with pytest.raises(ValueError):
        budget.estimate(HAIKU, "1000", 0)  # type: ignore[arg-type]


# ---------------------------------------------------------------- allow semantics


def test_allow_is_read_only_and_never_creates_the_ledger_file(ledger):
    for _ in range(3):
        assert budget.allow(1.0) is True
    assert not ledger.exists(), "allow() must not write the ledger"
    # The advisory lock lives in a sibling file named <ledger>.lock, nothing else is created.
    assert {p.name for p in ledger.parent.iterdir()} == {ledger.name + ".lock"}


def test_future_dated_bucket_counts_toward_total_but_not_today(ledger, monkeypatch):
    # A bucket from a day that is not "today" (here: the far future, e.g. after a clock
    # misconfiguration) must still count against the lifetime cap but not the daily one.
    monkeypatch.setattr(budget, "_utc_today", lambda: "2026-05-05")
    _seed(
        ledger, {"days": {"2099-01-01": {"usd": 5.0, "calls": 1, "tokens_in": 1, "tokens_out": 1}}}
    )
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "6")
    assert budget.remaining() == pytest.approx({"daily": 1.0, "total": 1.0})
    assert budget.allow(1.0) is True
    assert budget.allow(1.01) is False  # both caps have exactly 1.0 left


def test_kill_switch_wins_over_corrupt_ledger_in_public_mode_without_raising(ledger, monkeypatch):
    _seed(ledger, {"days": {}})
    ledger.write_text("not json at all")
    monkeypatch.setenv("PUBLIC_MODE", "on")
    monkeypatch.setenv("AI_DISABLED", "On")
    assert budget.allow(0.0) is False
    monkeypatch.setenv("AI_DISABLED", "off")
    assert budget.allow(0.0) is False  # still fail-closed on the corrupt file


def test_ledger_path_is_read_at_call_time(ledger, tmp_path, monkeypatch):
    budget.charge(0.5, model=HAIKU, tokens_in=1, tokens_out=1)
    other = tmp_path / "elsewhere" / "ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(other))
    assert budget.remaining()["daily"] == pytest.approx(100.0)  # fresh ledger, nothing spent
    budget.charge(0.25, model=HAIKU, tokens_in=1, tokens_out=1)
    assert json.loads(other.read_text())["days"][budget._utc_today()]["usd"] == 0.25
    assert json.loads(ledger.read_text())["days"][budget._utc_today()]["usd"] == 0.5


def test_default_ledger_path_when_env_unset(monkeypatch):
    monkeypatch.delenv("AI_LEDGER_PATH", raising=False)
    assert budget._ledger_path() == Path("data_cache/ai_ledger.json")
    monkeypatch.setenv("AI_LEDGER_PATH", "")  # blank counts as unset
    assert budget._ledger_path() == Path("data_cache/ai_ledger.json")


def test_unreadable_ledger_path_is_a_directory(ledger, monkeypatch, caplog):
    ledger.mkdir(parents=True)  # the "file" is a directory: open() raises IsADirectoryError
    monkeypatch.setenv("PUBLIC_MODE", "on")
    assert budget.allow(0.0) is False
    monkeypatch.setenv("PUBLIC_MODE", "off")
    with caplog.at_level(logging.WARNING, logger="quantforge.ai.budget"):
        assert budget.allow(0.0) is True
    assert any("corrupt" in rec.getMessage() for rec in caplog.records)


def test_explicit_zero_or_negative_cap_denies_everything_even_in_dev(ledger, monkeypatch):
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "0")
    assert budget.allow(0.0) is False
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "-1")
    assert budget.allow(0.0) is False
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "0.5")
    assert budget.allow(0.5) is True


def test_public_mode_with_valid_caps_enforces_the_same_boundary(ledger, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "1")
    budget.charge(0.7, model=HAIKU, tokens_in=1, tokens_out=1)
    assert budget.allow(0.3) is True
    assert budget.allow(0.31) is False
    assert budget.remaining() == pytest.approx({"daily": 0.3, "total": 0.3})


def test_infinite_cap_in_env_falls_back_rather_than_allowing_unbounded(ledger, monkeypatch):
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "inf")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "1e400")  # parses to inf
    # dev: falls back to 2 / 10
    assert budget.allow(2.0) is True
    assert budget.allow(2.01) is False
    # public: falls back to 0 -> deny
    monkeypatch.setenv("PUBLIC_MODE", "on")
    assert budget.allow(0.0) is False


def test_allow_accepts_int_estimate_and_rejects_bool(ledger):
    assert budget.allow(1) is True
    assert budget.allow(False) is False


# ---------------------------------------------------------------- charge / persistence


def test_many_small_charges_accumulate_with_float_drift_but_stay_consistent(ledger, monkeypatch):
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1")
    for _ in range(10):
        budget.charge(0.1, model=HAIKU, tokens_in=1, tokens_out=1)
    bucket = json.loads(ledger.read_text())["days"][budget._utc_today()]
    assert bucket["calls"] == 10
    assert bucket["usd"] == pytest.approx(1.0)
    # remaining() and allow() must agree: whatever remaining says is left is exactly allowed.
    left = budget.remaining()["daily"]
    assert 0.0 <= left < 1e-9
    assert budget.allow(left) is True
    assert budget.allow(left + 1e-6) is False


def test_corrupt_ledger_in_dev_is_rebuilt_by_charge(ledger, caplog):
    _seed(ledger, {})
    ledger.write_text("{")
    with caplog.at_level(logging.WARNING, logger="quantforge.ai.budget"):
        budget.charge(0.2, model=HAIKU, tokens_in=4, tokens_out=6)
    on_disk = json.loads(ledger.read_text())
    assert on_disk == {
        "days": {budget._utc_today(): {"usd": 0.2, "calls": 1, "tokens_in": 4, "tokens_out": 6}}
    }
    assert any("corrupt" in rec.getMessage() for rec in caplog.records)


def test_partial_bucket_without_counters_is_tolerated(ledger, monkeypatch):
    today = budget._utc_today()
    _seed(ledger, {"days": {today: {"usd": 0.5}}})  # older/hand-edited bucket, counters missing
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1")
    assert budget.allow(0.5) is True
    assert budget.allow(0.51) is False
    budget.charge(0.25, model=HAIKU, tokens_in=2, tokens_out=3)
    assert json.loads(ledger.read_text())["days"][today] == {
        "usd": 0.75,
        "calls": 1,
        "tokens_in": 2,
        "tokens_out": 3,
    }


def test_ledger_written_as_valid_json_keyed_by_iso_utc_date(ledger, monkeypatch):
    monkeypatch.setattr(budget, "_utc_today", lambda: "2026-12-31")
    budget.charge(0.01, model="claude-sonnet-5", tokens_in=1, tokens_out=1)
    monkeypatch.setattr(budget, "_utc_today", lambda: "2027-01-01")
    budget.charge(0.02, model="claude-sonnet-5", tokens_in=1, tokens_out=1)
    on_disk = json.loads(ledger.read_text())
    assert list(on_disk) == ["days"]
    assert set(on_disk["days"]) == {"2026-12-31", "2027-01-01"}
    assert budget.remaining()["total"] == pytest.approx(100.0 - 0.03)


def test_real_utc_today_is_iso_date_string():
    today = budget._utc_today()
    assert len(today) == 10 and today[4] == "-" and today[7] == "-"


def test_concurrent_allow_and_charge_interleave_safely(ledger):
    errors: list[BaseException] = []
    amount, n_workers, per_worker = 0.0625, 6, 8  # 1/16 is exact in binary

    def charger() -> None:
        try:
            for _ in range(per_worker):
                budget.charge(amount, model=HAIKU, tokens_in=1, tokens_out=1)
        except BaseException as exc:  # noqa: BLE001 — collected and re-raised below
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(per_worker * 3):
                assert budget.allow(0.0) is True
                assert budget.remaining()["daily"] >= 0.0
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=charger) for _ in range(n_workers)]
    threads += [threading.Thread(target=reader) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    bucket = json.loads(ledger.read_text())["days"][budget._utc_today()]
    assert bucket["usd"] == n_workers * per_worker * amount == 3.0
    assert bucket["calls"] == n_workers * per_worker


# ---------------------------------------------------------------- repo hygiene


def test_repo_data_cache_never_receives_ledger_files():
    # Every budget test redirects AI_LEDGER_PATH; if one forgot, these files would be created
    # or rewritten here (they are gitignored, so `git status` alone would not reveal the leak).
    # A file that already existed, unchanged, before collection is a developer's real spend
    # record, not a leak — see ``_REPO_LEDGER_STATE_AT_COLLECTION``.
    now = _repo_ledger_state()
    for name in _REPO_LEDGER_FILES:
        assert now[name] == _REPO_LEDGER_STATE_AT_COLLECTION[name], (
            f"{name} was created or modified by the test session"
        )


def test_config_files_document_the_new_env_vars():
    env_example = (REPO_ROOT / ".env.example").read_text()
    assert "AI_DISABLED=off" in env_example
    assert "AI_LEDGER_PATH=data_cache/ai_ledger.json" in env_example
    # Since 2026-09-12 the ignore is a prefix glob (`ai_ledger.json*`) so the ledger, its `.lock`,
    # a `.corrupt` quarantine and the mkstemp `.<rand>.tmp` are all covered by one line.
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "data_cache/ai_ledger.json*" in gitignore.splitlines()


# ---------------------------------------------------------------- tampered-ledger integrity


def test_negative_usd_in_ledger_cannot_lift_the_total_cap(ledger, monkeypatch):
    # Adversarial: charge() can never write a negative amount, so a negative bucket means the
    # file was hand-edited. Before validation caught it, spent_total went to -50 and a $40 call
    # sailed through a $1 lifetime cap. Now it is "corrupt": fail closed in public mode, and in
    # dev the tampered history is discarded (with a warning) rather than trusted.
    _seed(
        ledger,
        {"days": {"2026-01-01": {"usd": -50.0, "calls": 1, "tokens_in": 1, "tokens_out": 1}}},
    )
    monkeypatch.setattr(budget, "_utc_today", lambda: "2026-09-11")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "100")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "1")
    monkeypatch.setenv("PUBLIC_MODE", "on")
    assert budget.allow(40.0) is False
    assert budget.remaining() == {"daily": 0.0, "total": 0.0}
    monkeypatch.setenv("PUBLIC_MODE", "off")
    assert budget.allow(1.0) is True  # treated as empty: exactly the $1 total cap is left
    assert budget.allow(1.01) is False  # ...and not one cent more
    assert budget.remaining() == pytest.approx({"daily": 100.0, "total": 1.0})


@pytest.mark.parametrize(
    "bucket",
    [
        {"usd": 0.1, "calls": "x"},
        {"usd": 0.1, "tokens_in": -5},
        {"usd": 0.1, "tokens_out": True},
        {"usd": 0.1, "calls": 1.5},
    ],
)
def test_non_int_counters_follow_the_corrupt_ledger_policy(ledger, monkeypatch, caplog, bucket):
    # Adversarial: a bad counter must not surface as a raw ValueError from int("x") inside
    # charge(); it takes the same documented path as any other corrupt file.
    today = budget._utc_today()
    _seed(ledger, {"days": {today: bucket}})
    monkeypatch.setenv("PUBLIC_MODE", "on")
    assert budget.allow(0.0) is False
    with pytest.raises(budget.LedgerCorruptError):
        budget.charge(0.1, model=HAIKU, tokens_in=1, tokens_out=1)
    monkeypatch.setenv("PUBLIC_MODE", "off")
    with caplog.at_level(logging.WARNING, logger="quantforge.ai.budget"):
        budget.charge(0.1, model=HAIKU, tokens_in=2, tokens_out=3)
    assert any("corrupt" in rec.getMessage() for rec in caplog.records)
    assert json.loads(ledger.read_text())["days"][today] == {
        "usd": 0.1,
        "calls": 1,
        "tokens_in": 2,
        "tokens_out": 3,
    }


def test_charge_then_allow_hand_computed_round_trip(ledger, monkeypatch):
    # Hand-computed end to end: sonnet 250k in ($0.50) + 20k out ($0.20) = $0.70 charged; with a
    # $1 daily cap exactly $0.30 remains, so a haiku call priced at 60k in + 48k out
    # (0.06 + 0.24 = $0.30) lands on the cap and is allowed, one more output token is not.
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1")
    cost = budget.estimate("claude-sonnet-5", 250_000, 20_000)
    assert cost == pytest.approx(0.70)
    budget.charge(cost, model="claude-sonnet-5", tokens_in=250_000, tokens_out=20_000)
    on_cap = budget.estimate(HAIKU, 60_000, 48_000)
    assert on_cap == pytest.approx(0.30)
    assert budget.allow(on_cap) is True
    assert budget.allow(budget.estimate(HAIKU, 60_000, 48_001)) is False
    assert budget.remaining()["daily"] == pytest.approx(0.30)
