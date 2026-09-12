"""Token-spend accounting + hard kill-switch for every Claude API call in the app.

Why this module exists (SF-1 / SF-2 in the brief)
--------------------------------------------------
QuantForge ships as a **public URL with paid AI hooks**. Anyone who finds the demo can hammer the
natural-language interface or the research agent, and every request costs real money on the
author's personal API account. A budget gate is therefore a hard requirement, not polish:

    estimate(...) -> allow(est) -> [call the model] -> charge(actual, ...)

No code path in the repo may call the model without passing through ``allow``/``charge``
(``tests/test_public_mode_no_codegen.py`` and the grep-level check in
``docs/components/09-ai-budget.md`` enforce that). This module deliberately does **not** import the
SDK client — it prices token counts, it never talks to the network — so it can be unit-tested in
milliseconds and cannot itself become a leak.

Design decisions, and why
-------------------------
* **Fail closed.** ``allow()`` is a gate, not an error: it never raises. Anything that goes wrong
  (missing caps in ``PUBLIC_MODE``, a corrupt ledger, an unparseable estimate) resolves to
  ``False`` — the UI then degrades to cached scenarios. The one deliberate exception is local
  development (``PUBLIC_MODE`` off), where a corrupt ledger is treated as empty with a warning so a
  stray edit doesn't brick the dev loop; that laxity is exactly what public mode removes.
* **Caps are read at call time**, not at import: the kill-switch (``AI_DISABLED=on``) must take
  effect the moment the operator flips it, without a redeploy or process restart.
* **UTC days.** "Today" is the UTC calendar date so the ledger keys match AWS Budgets' daily
  granularity and never jump backwards across a DST change on the host.
* **One JSON file, two locks.** The deploy is a single Fargate task, so a JSON ledger on disk is
  sufficient. Streamlit serves each session from its own thread, hence a module-level
  ``threading.Lock``; and because a restart or a second worker process could briefly overlap, an
  advisory ``fcntl.flock`` on ``<ledger>.lock`` guards the file across processes. Writes go
  through a temp file + ``os.replace`` so a crash mid-write can never leave a half-written (i.e.
  corrupt → fail-closed) ledger behind. If the app ever scales past one container, the
  ``_read_ledger``/``_write_ledger`` pair is the seam to swap for DynamoDB or Redis — a documented
  seam, not something built this summer.
* **Strict ``>`` at the cap.** A call that lands exactly on the cap is allowed; only exceeding it
  is denied. This keeps the boundary tests unambiguous and matches how AWS Budgets alerts work.

Environment variables (see ``docs/components/18-runtime-config.md``)
--------------------------------------------------------------------
``AI_BUDGET_USD_DAILY`` / ``AI_BUDGET_USD_TOTAL``  hard caps in USD; missing or unparseable →
    ``0.0`` (deny everything) when ``PUBLIC_MODE=on``, else the dev defaults 2.0 / 10.0.
``AI_DISABLED``      ``on`` (case-insensitive) denies every call instantly.
``AI_LEDGER_PATH``   JSON ledger location; default ``data_cache/ai_ledger.json``.
``PUBLIC_MODE``      ``on`` selects the fail-closed behaviours above.
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# USD per **million** tokens, (input, output). These are the first-party API rates from the
# claude-api skill's model table (Haiku 4.5: $1/$5, Sonnet 5: $2/$10, Opus 5: $5/$25). Keys are
# the bare model IDs the API accepts — never date-suffixed variants. Cache reads/writes are billed
# at different multipliers; we ignore them here on purpose, which only ever *over*-estimates cost
# (cache reads are cheaper than fresh input), so the gate errs on the side of denying.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
}

# Generous local-development caps used only when PUBLIC_MODE is off and the env is silent.
# In public mode a silent env means "deny" — nobody should ever accidentally run uncapped.
_DEV_DEFAULT_DAILY_USD = 2.0
_DEV_DEFAULT_TOTAL_USD = 10.0

_DEFAULT_LEDGER_PATH = "data_cache/ai_ledger.json"

# Serialises every ledger read-modify-write across Streamlit's session threads. The file lock in
# ``_locked`` handles other *processes*; this handles other *threads* in this one.
_LOCK = threading.Lock()


class LedgerCorruptError(RuntimeError):
    """The ledger file exists but is not the JSON shape this module writes.

    Raised internally so callers can decide the policy: ``allow`` maps it to ``False`` in public
    mode (fail closed) and to "empty ledger + warning" in dev; ``charge`` refuses to overwrite a
    corrupt ledger in public mode so a human inspects it rather than silently resetting spend.
    """


# --------------------------------------------------------------------------------------------
# Clock / environment helpers (each is a tiny function so tests can monkeypatch it in isolation)
# --------------------------------------------------------------------------------------------


def _utc_today() -> str:
    """Return today's UTC date as ``YYYY-MM-DD`` — the ledger's day key.

    A separate function (not inlined) so tests can monkeypatch the clock to prove day rollover
    without sleeping until midnight.
    """
    return datetime.now(timezone.utc).date().isoformat()


def _public_mode() -> bool:
    """True when ``PUBLIC_MODE=on``. Read at call time: flipping the env must not need a restart."""
    return os.getenv("PUBLIC_MODE", "off").strip().lower() == "on"


def _kill_switch_on() -> bool:
    """True when the operator has set ``AI_DISABLED=on`` (case-insensitive)."""
    return os.getenv("AI_DISABLED", "off").strip().lower() == "on"


def _ledger_path() -> Path:
    """Resolve ``AI_LEDGER_PATH`` at call time so tests (and deploys) can redirect the ledger."""
    return Path(os.getenv("AI_LEDGER_PATH") or _DEFAULT_LEDGER_PATH)


def _cap_from_env(name: str, dev_default: float) -> float:
    """Parse a USD cap from the environment with the fail-closed fallback policy.

    Missing, blank, unparseable, or non-finite values fall back to ``0.0`` (deny) in public mode
    and to ``dev_default`` otherwise. We never guess a "reasonable" cap in production: an operator
    who forgot to set one gets zero AI calls, not a surprise bill.
    """
    raw = os.getenv(name)
    fallback = 0.0 if _public_mode() else dev_default
    if raw is None or not raw.strip():
        return fallback
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using cap %.2f", name, raw, fallback)
        return fallback
    if not math.isfinite(value):
        logger.warning("%s=%r is not finite; using cap %.2f", name, raw, fallback)
        return fallback
    return value


def _caps() -> tuple[float, float]:
    """Return ``(daily_cap, total_cap)`` using the policy in ``_cap_from_env``."""
    return (
        _cap_from_env("AI_BUDGET_USD_DAILY", _DEV_DEFAULT_DAILY_USD),
        _cap_from_env("AI_BUDGET_USD_TOTAL", _DEV_DEFAULT_TOTAL_USD),
    )


# --------------------------------------------------------------------------------------------
# Ledger persistence
# --------------------------------------------------------------------------------------------


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold both the thread lock and an advisory file lock on ``<ledger>.lock``.

    The lock file is separate from the ledger because the ledger itself is replaced atomically via
    ``os.replace`` — locking an inode that is about to be swapped out would let a second process
    lock the *new* inode concurrently. The parent directory is created here (it is needed for the
    lock file regardless of whether the ledger exists yet).
    """
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(path.name + ".lock")
        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _empty_ledger() -> dict:
    """The ledger shape on first run — also what dev mode substitutes for a corrupt file."""
    return {"days": {}}


def _validate_ledger(data: object) -> dict:
    """Return ``data`` if it has the expected shape, else raise ``LedgerCorruptError``.

    Shape: ``{"days": {"YYYY-MM-DD": {"usd": float, "calls": int, "tokens_in": int,
    "tokens_out": int}}}``. We validate on read rather than trusting the file because a corrupt
    ``usd`` (e.g. a string) would otherwise blow up inside ``allow`` as a ``TypeError`` — and the
    whole point is that ``allow`` resolves every failure to a deliberate policy, not a traceback.
    """
    if not isinstance(data, dict) or not isinstance(data.get("days"), dict):
        raise LedgerCorruptError("ledger root must be {'days': {...}}")
    for day, bucket in data["days"].items():
        if not isinstance(day, str) or not isinstance(bucket, dict):
            raise LedgerCorruptError(f"bad day bucket for {day!r}")
        usd = bucket.get("usd", 0.0)
        if isinstance(usd, bool) or not isinstance(usd, (int, float)) or not math.isfinite(usd):
            raise LedgerCorruptError(f"non-numeric usd for {day!r}: {usd!r}")
        if usd < 0:
            # ``charge`` can never write a negative amount, so one on disk means the file was
            # hand-edited. Accepting it would let ``spent_total`` go negative and quietly lift
            # the lifetime cap — the opposite of fail-closed.
            raise LedgerCorruptError(f"negative usd for {day!r}: {usd!r}")
        for counter in ("calls", "tokens_in", "tokens_out"):
            value = bucket.get(counter, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LedgerCorruptError(f"bad {counter} for {day!r}: {value!r}")
    return data


def _read_ledger(path: Path) -> dict:
    """Load and validate the ledger; a missing file is an empty ledger (first run).

    Must be called with ``_locked(path)`` held. Any parse/shape/IO problem surfaces as
    ``LedgerCorruptError`` so callers apply one consistent policy.
    """
    if not path.exists():
        return _empty_ledger()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        raise LedgerCorruptError(f"cannot read ledger at {path}: {exc}") from exc
    return _validate_ledger(data)


def _write_ledger(path: Path, ledger: dict) -> None:
    """Atomically persist the ledger (temp file in the same dir + ``os.replace``).

    Must be called with ``_locked(path)`` held. Writing to a sibling temp file and renaming means
    readers only ever see the old complete file or the new complete file — never a truncated one,
    which in public mode would fail closed and silently disable the demo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # Don't leave temp files littering the cache dir on failure.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _spent(ledger: dict) -> tuple[float, float]:
    """Return ``(spent_today, spent_total)`` in USD from a validated ledger."""
    days = ledger["days"]
    today = float(days.get(_utc_today(), {}).get("usd", 0.0))
    total = float(sum(bucket.get("usd", 0.0) for bucket in days.values()))
    return today, total


def _load_ledger_with_policy(path: Path) -> dict:
    """Read the ledger applying the corrupt-file policy: dev → empty + warning, public → raise.

    Must be called with ``_locked(path)`` held. Split out so ``allow``, ``charge`` and
    ``remaining`` share exactly one interpretation of "the file is broken".
    """
    try:
        return _read_ledger(path)
    except LedgerCorruptError:
        if _public_mode():
            raise
        logger.warning("AI ledger at %s is corrupt; treating as empty (dev mode)", path)
        return _empty_ledger()


def _check_token_count(name: str, value: int) -> None:
    """Reject bools and negatives: ``True`` silently counting as one token is a classic bug."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def estimate(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return the estimated USD cost of a call: pure arithmetic against ``PRICES_PER_MTOK``.

    Called *before* every model call (with ``count_tokens`` or a conservative guess for the input
    and ``max_tokens`` for the output) to feed ``allow``. An unknown model raises rather than
    returning 0 — silently under-estimating an unpriced model is exactly the failure this module
    exists to prevent, so we fail loud and name the models we do know.
    """
    if model not in PRICES_PER_MTOK:
        known = ", ".join(sorted(PRICES_PER_MTOK))
        raise ValueError(f"unknown model {model!r}; priced models are: {known}")
    _check_token_count("tokens_in", tokens_in)
    _check_token_count("tokens_out", tokens_out)
    in_rate, out_rate = PRICES_PER_MTOK[model]
    return tokens_in / 1e6 * in_rate + tokens_out / 1e6 * out_rate


def allow(estimated_usd: float = 0.0) -> bool:
    """Return True only if a call of ~``estimated_usd`` stays within the daily AND total caps.

    This is a gate, never an error: it **never raises**. It returns False when
      * the kill-switch ``AI_DISABLED=on`` is set;
      * ``estimated_usd`` is not a finite, non-negative number;
      * ``spent_today + estimated_usd > AI_BUDGET_USD_DAILY`` (strict: landing on the cap is OK);
      * ``spent_total + estimated_usd > AI_BUDGET_USD_TOTAL``;
      * either cap resolves to ``<= 0`` (a zero cap is "no budget", so even free calls are denied);
      * the ledger is corrupt/unreadable and ``PUBLIC_MODE=on`` (fail closed);
      * anything else unexpected goes wrong (logged, then denied).
    Caps and the kill-switch are re-read from the environment on every call.
    """
    try:
        if _kill_switch_on():
            return False
        if (
            isinstance(estimated_usd, bool)
            or not isinstance(estimated_usd, (int, float))
            or not math.isfinite(estimated_usd)
            or estimated_usd < 0
        ):
            return False
        daily_cap, total_cap = _caps()
        if daily_cap <= 0.0 or total_cap <= 0.0:
            # A zero cap means "no budget", not "a budget of $0 you may land exactly on". This is
            # what makes a missing cap in PUBLIC_MODE deny everything, including free calls.
            return False
        path = _ledger_path()
        with _locked(path):
            ledger = _load_ledger_with_policy(path)
        spent_today, spent_total = _spent(ledger)
        if spent_today + estimated_usd > daily_cap:
            return False
        if spent_total + estimated_usd > total_cap:
            return False
        return True
    except LedgerCorruptError as exc:
        logger.error("AI ledger corrupt in PUBLIC_MODE; denying: %s", exc)
        return False
    except Exception:  # noqa: BLE001 — a gate must resolve every failure to a decision
        logger.exception("budget.allow failed; denying the call")
        return False


def charge(usd: float, *, model: str, tokens_in: int, tokens_out: int) -> None:
    """Record actual spend (from the API response's ``usage`` fields) into today's UTC bucket.

    ``usd`` is what the caller computed from real usage — typically ``estimate(model, in, out)``
    with the response's token counts — so the ledger reflects money actually spent, not the
    pre-call guess. Validation is strict (finite, ≥ 0; ints, not bools) because a bad value here
    corrupts every future ``allow`` decision.

    In dev mode a corrupt ledger is replaced with a fresh one (with a warning). In public mode we
    raise ``LedgerCorruptError`` instead: overwriting would reset the spend history and re-open a
    gate that ``allow`` has deliberately closed.
    """
    if isinstance(usd, bool) or not isinstance(usd, (int, float)):
        raise ValueError(f"usd must be a number, got {usd!r}")
    if not math.isfinite(usd) or usd < 0:
        raise ValueError(f"usd must be finite and >= 0, got {usd!r}")
    if not isinstance(model, str) or not model:
        raise ValueError(f"model must be a non-empty string, got {model!r}")
    _check_token_count("tokens_in", tokens_in)
    _check_token_count("tokens_out", tokens_out)

    path = _ledger_path()
    with _locked(path):
        ledger = _load_ledger_with_policy(path)
        bucket = ledger["days"].setdefault(
            _utc_today(), {"usd": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0}
        )
        bucket["usd"] = float(bucket.get("usd", 0.0)) + float(usd)
        bucket["calls"] = int(bucket.get("calls", 0)) + 1
        bucket["tokens_in"] = int(bucket.get("tokens_in", 0)) + tokens_in
        bucket["tokens_out"] = int(bucket.get("tokens_out", 0)) + tokens_out
        _write_ledger(path, ledger)
    logger.info("AI spend: $%.5f on %s (%d in / %d out)", usd, model, tokens_in, tokens_out)


def remaining() -> dict[str, float]:
    """Return ``{"daily": usd_left_today, "total": usd_left}`` for the UI's transparency panel.

    Uses the same cap resolution as ``allow`` so the number shown always agrees with the gate.
    Never negative (a cap lowered below existing spend shows 0, not a confusing minus). Applies
    the same corrupt-ledger policy as ``allow``: in public mode that means 0 remaining rather than
    an exception, because this feeds a display, not a decision.
    """
    daily_cap, total_cap = _caps()
    path = _ledger_path()
    try:
        with _locked(path):
            ledger = _load_ledger_with_policy(path)
    except LedgerCorruptError as exc:
        logger.error("AI ledger corrupt in PUBLIC_MODE; reporting 0 remaining: %s", exc)
        return {"daily": 0.0, "total": 0.0}
    spent_today, spent_total = _spent(ledger)
    return {
        "daily": max(daily_cap - spent_today, 0.0),
        "total": max(total_cap - spent_total, 0.0),
    }
