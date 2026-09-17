"""Independent verifier tests for milestone 1 of the handoff open-items fix run (2026-09-12).

Milestone: config + hygiene alignment (.env.example, requirements.txt, .gitignore). These go
beyond the builder's string-matching pins in tests/test_handoff_open_items.py by exercising the
BEHAVIOUR the config files are supposed to produce:

- The example caps, fed through the real ``budget._caps`` policy, resolve to the dev defaults in
  BOTH modes — and (adversarial) dropping them under ``PUBLIC_MODE=on`` collapses to 0/0, which is
  exactly why the example must ship non-blank values.
- ``.env.example`` is a valid dotenv file (python-dotenv parses it to the same values the regex
  pins see), so a ``cp .env.example .env`` round-trips.
- The three sources of truth for the caps — ``budget.py``, ``.env.example`` and the table in
  docs/components/18-runtime-config.md — agree with each other, not just pairwise.
- The .gitignore globs cover the sidecar names the code REALLY produces (captured from a live
  ``_write_ledger`` / ``_locked`` call, not hand-typed) without over-matching: ``.env.example``,
  ``tests/fixtures/*.parquet`` and an unrelated ``data_cache/`` file stay trackable.
- The new ``jsonschema`` requirement line parses as a PEP 508 requirement and the package is
  importable, so tests/test_mcp_server.py can actually run from a fresh ``pip install -r``.

Milestone 2 — budget.py accepts ``numbers.Integral`` token counts. Beyond the builder's pins:

- A token count that is genuinely numpy-typed because it came off a pandas ``Series.sum()`` (the
  realistic source, not a hand-built ``np.int64``) flows through estimate -> charge -> ``allow``
  with a HAND-COMPUTED ledger and an unchanged cap decision.
- Mutation test (adversarial): if the coercion were dropped and only the check relaxed, ``charge``
  with a numpy count would blow up inside ``json.dump`` *after* validation — the very failure the
  spec calls out — so the ``int(value)`` return is proven load-bearing, not cosmetic.
- Rejections the builder did not cover: ``Fraction(3, 1)`` and ``Decimal(3)`` (numeric, exact,
  but not Integral), ``None``, complex, ``np.float32``, negative ``np.int8``/``np.int16``, and
  ``np.bool_`` in every argument position — all with the exact
  ``{name} must be a non-negative int, got <repr>`` message shape.
- ``int`` subclasses (``IntEnum``) are accepted and come out as exact builtin ``int``; a
  ``np.uint64`` at its maximum does not overflow (Python ints are unbounded after coercion).
- Mixed int / numpy charges accumulate exactly, every counter on disk is a builtin ``int``, the
  log line renders the coerced values, and 8 concurrent numpy-typed charges still sum exactly.
- ``allow`` was not to be touched: its source still contains no ``_check_token_count`` call.

Milestone 3 — agent.py pre-spend validation, None-Sharpe guard, tool_choice/thinking pin,
holdout-cost docstring. Beyond the builder's pins:

- ``_call`` refuses ``tools=[]`` AND ``tools=None`` before ``messages.create`` (a client whose
  ``create`` is a tripwire) and before ``budget.charge`` (a spy); the happy path with ONE tool
  still sends, still marks that tool as the cache breakpoint, and charges a HAND-COMPUTED
  $0.004 (1000 in + 200 out on Sonnet 5 at $2/$10 per MTok) — so the guard is a guard, not a
  regression.
- ``_val_sharpe``: None, NaN and -inf all rank as ``-inf`` and tie-break to the EARLIEST iter; a
  finite NEGATIVE Sharpe still beats a None one (``-3.0 > -inf``); a history that went through
  the real ``_round_metrics`` -> ``json.dumps`` -> ``json.loads`` path (NaN becomes ``null``
  becomes ``None``) ranks without a TypeError — the scenario the spec names.
- ``run_research`` pre-spend order proven against the REAL ``_get_client`` (only the SDK
  constructor ``anthropic.Anthropic`` is replaced by a tripwire): every non-str engine kind
  (list, None, int, bytes, tuple) and every bad ticker shape (unknown, duplicate, tuple, dict,
  nested list, None element) raises ValueError with the tool's own message, the constructor is
  never called, ``agent._client`` stays None, the registry stays empty and no ledger appears.
- Request shape (adversarial): the recorded ``tool_choice`` is EQUAL to but NOT the module
  constant (a request-side mutation could not leak back), the constant is unchanged after a
  run, the model id is a ``budget.PRICES_PER_MTOK`` key, and the request carries no
  ``temperature``/``top_p``/``top_k``/``thinking``/``output_config`` (all 400 or shape-changing
  on Sonnet 5 per the claude-api skill).
- Holdout cost: with the loop at the MOST flattering allowed cost (``cost_bps=0.0``) the holdout
  metrics still equal an independent hand-scoring of the same winner at
  ``guardrails._HOLDOUT_COST_BPS`` on a fresh handle, and DIFFER from a hand-scoring at 0 bps —
  so the pin proves the cost actually moved the number, not just that a dict field was set.

Milestone 4 — nl_interface.py: query-shape gate BEFORE the rate limit (a free clarification);
``_raw_plan`` refuses a non-list ``tickers``. Beyond the builder's pins:

- The gate predicate in ``handle`` is proven EQUIVALENT to ``_validate_query`` over a battery of
  boundary strings (exactly 2000 chars, 2001, whitespace-padded, Unicode whitespace, multi-byte
  characters, a ``str`` subclass): if any input could pass ``handle``'s gate and still fail
  ``_validate_query`` inside ``_parse_call``, a rate slot would be burned and a ValueError would
  escape — the exact bug the milestone fixes, reintroduced one layer down.
- The 2000-char boundary is admitted (consumes exactly one slot, charges the hand-computed parse
  cost) while 2001 is not; interleaved bad/valid queries under ``AI_RATE_LIMIT_PER_HOUR=2`` leave
  exactly two stamps, so the free path is free by count, not just by "no file".
- Adversarial: ``budget._ledger_path`` (the one resolver both the ledger and the derived
  rate-state path go through) is a tripwire — a bad query returns the clarification without
  ever resolving where the state would live.
- ``tickers`` as a dict whose KEYS are valid tickers (``{"AAPL": 1, "MSFT": 2}``) is the sneaky
  case: ``list()`` would have produced a perfectly valid ticker list and silently run a backtest
  the user never asked for; through the full ``handle`` path it is now a clarification with
  ``fallback=None``, ``load_data`` is never reached (tripwire), no explain call is made, and the
  parse call's spend is still charged and reported.
- The str case ``"AAPL"`` through ``handle`` proves the OLD outcome (``list()`` -> unknown ticker
  ``"A"`` -> fallback ``invalid``) is gone: fallback is ``None`` and ``load_data`` is untouched.

Milestone 5 — mcp_server + guardrails doc/docstring corrections (text-only code changes).
Beyond the builder's string pins, these prove the CLAIMS the new prose makes:

- Text-only: after blanking every string constant, the AST of both modules equals HEAD's; the
  only string constants that differ (docstrings aside) are the two selector descriptions.
- Validation order with a SPY on the gate: for each malformed shape the gate is never invoked
  (the docstring's "never sees it"), and for a codegen-shaped payload it is invoked exactly once
  with exactly ``{strategy, params}`` and wins over the engine/cost checks that follow.
- (Adversarial) the strict schema ACCEPTS both-selectors and no-selector payloads under
  ``jsonschema`` — the exactly-one rule really is absent from the schema, which is why it must
  live in the descriptions — and the function body / MCP wire refuse both with the message.
- The ``_resolve_engine`` cycle claim, in a fresh interpreter: importing guardrails does NOT load
  mcp_server; calling ``_resolve_engine`` does; importing mcp_server loads guardrails.
- The ``HoldoutHandle`` sentence is honest, not a hedge: in-process introspection of the closure
  cell DOES reach the holdout frame, while every tool-mediated path (repr, vars, pickle, MCP
  ``load_data``) is refused.

Milestone 6 — ``ruff format`` over the six not-yet-clean files with zero behaviour change.
Beyond the builder's pins (UNIVERSE order/length, section comments present, subprocess
``--check`` on the six paths):

- Every one of the six files is proven identical to ``git show HEAD:<path>`` at THREE levels:
  ``ast.dump`` (semantics, docstrings included), the verbatim comment stream (AST equality is
  blind to comments — the ``# Tech`` / ``# ADRs`` labels are exactly what had to survive), and
  the significant-token stream with only commas allowed to differ (the one magic trailing comma
  the signature wrap adds). After the closeout commit HEAD catches up and the pin stays true.
- Scope (adversarial): among ALL ``.py`` files that differ from HEAD, the ones whose AST is
  unchanged are a subset of the six — nothing else was silently reformatted in this run.
- The repo-wide ``ruff format --check .`` count equals the number of ``.py`` files git knows
  about (tracked + untracked-not-ignored), so "already formatted" really covers the tree.
- The check is load-bearing (adversarial): the HEAD-era layout (ticker rows, hanging-indent
  signature) written to a temp file FAILS ``--check`` under the same ``--isolated
  --line-length 100`` settings pyproject pins, while the six real files pass those settings —
  and formatting that snippet reproduces the per-ticker explosion with the comment in place.
- Each section comment still labels EXACTLY its hand-typed ticker group (the builder only
  checked the comments' relative order, not that no ticker slid across a section boundary).
- ``load_prices``'s wrapped signature kept every parameter, kind and default; ``ANN`` /
  ``RISK_FREE`` kept their values, and ``compute_metrics`` on a three-return series equals a
  hand computation of every headline metric (max drawdown exactly -1%).

Milestone 7 — closeout: the 2026-09-12 handoff entry, the 16-tests row, the closeout-verifier
pin advances. Beyond the builder's entry-level self-check:

- Whole-file reconstruction: the working handoff.md IS ``HEAD preamble + one new entry + HEAD
  body`` byte for byte (the entry-split pins are blind to the preamble and the separators).
- The closeout count is tied to a real full ``--collect-only`` by arithmetic: ``N + 2`` equals
  the full collection minus whatever THIS file has grown past the 119 the entry records, and
  the entry's 69 / 119 per-file numbers are checked against real collections of those files.
- Adversarial: the builder's own newest-entry pin and the week-8 verifier's byte-identity pin
  are executed against (a) a placeholder ``**NNNN passed / 2 skipped**`` and (b) an earlier
  entry with one byte changed — both must raise, and the loose ``\\d+ passed / 2 skipped``
  regex is shown to be satisfied by the un-bold baseline phrase (why the bold form matters).
- The only ``def test_`` name that differs from HEAD across the week-3..8 verifiers is the one
  documented rename; every spec-listed advanced pin carries the dated comment; the functions
  whose source changed in those files are a subset of an explicit allowlist.
- 16-tests.md differs from HEAD by exactly one inserted line, directly below the week-8
  verifier row, with the table's column count.
- The entry's "Next up" names the Week-9 surfaces (with the markdown wrap normalised), quotes
  the exact live-smoke command, and claims no Week-9 work; its "app shell imports no
  ``quantforge.ai``" claim is checked against the real import list of app/streamlit_app.py.

Offline, no network, no API key.
"""

from __future__ import annotations

import ast
import asyncio
import difflib
import inspect
import io
import json
import logging
import math
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tokenize
from decimal import Decimal
from enum import IntEnum
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pandas as pd
import pytest
from dotenv import dotenv_values
from mcp.server.fastmcp.exceptions import ToolError
from packaging.requirements import Requirement

from quantforge import interchange
from quantforge.ai import agent, budget, guardrails, mcp_server, nl_interface
from quantforge.data import loader
from quantforge.metrics import performance

_ROOT = Path(__file__).resolve().parents[1]
_ENV_EXAMPLE_PATH = _ROOT / ".env.example"
_ENV_EXAMPLE = _ENV_EXAMPLE_PATH.read_text()
_DOC = (_ROOT / "docs" / "components" / "18-runtime-config.md").read_text()
_REQUIREMENTS = (_ROOT / "requirements.txt").read_text()


def _env_example_vars() -> dict[str, str]:
    return dict(re.findall(r"^([A-Z_]+)=(.*)$", _ENV_EXAMPLE, flags=re.M))


def _git_available() -> bool:
    return shutil.which("git") is not None and (_ROOT / ".git").exists()


def _is_ignored(rel: str) -> bool:
    proc = subprocess.run(
        ["git", "check-ignore", "-q", rel],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# .env.example: behaviour through the real budget policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("public_mode", ["off", "on"])
def test_example_caps_resolve_to_dev_defaults_through_budget_policy(monkeypatch, public_mode):
    """Copying the example into a real .env must yield the documented 2 / 10 in EITHER mode.

    Before the fix the file said 5 / 25, so a public deploy that copied the example would have
    run with caps the docs never mentioned. ``_cap_from_env`` is the real parser, so this also
    proves the values are parseable floats (a stray unit or comment would fall back / to 0).
    """
    values = _env_example_vars()
    monkeypatch.setenv("PUBLIC_MODE", public_mode)
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", values["AI_BUDGET_USD_DAILY"])
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", values["AI_BUDGET_USD_TOTAL"])
    assert budget._caps() == (budget._DEV_DEFAULT_DAILY_USD, budget._DEV_DEFAULT_TOTAL_USD)
    assert budget._caps() == (2.0, 10.0)


def test_adversarial_blank_example_caps_would_fail_closed_in_public_mode(monkeypatch):
    """Why the example MUST carry explicit values: blank caps + PUBLIC_MODE=on ⇒ 0 / 0 (deny all).

    This is the failure a reader would hit if the example ever shipped ``AI_BUDGET_USD_DAILY=``
    like it ships ``DEMO_PASSCODE=``. It also proves the previous test is not vacuous.
    """
    monkeypatch.setenv("PUBLIC_MODE", "on")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "")
    assert budget._caps() == (0.0, 0.0)
    # And the example itself never leaves either cap blank.
    values = _env_example_vars()
    assert values["AI_BUDGET_USD_DAILY"].strip() and values["AI_BUDGET_USD_TOTAL"].strip()


def test_env_example_is_valid_dotenv_and_round_trips():
    """``cp .env.example .env`` must load cleanly via python-dotenv with the same values the
    regex-based pins read — a malformed line would silently vanish under ``load_dotenv``."""
    parsed = dotenv_values(_ENV_EXAMPLE_PATH)
    assert parsed == _env_example_vars()
    assert set(parsed) == {
        "ANTHROPIC_API_KEY",
        "AI_BUDGET_USD_DAILY",
        "AI_BUDGET_USD_TOTAL",
        "AI_DISABLED",
        "AI_LEDGER_PATH",
        "AI_RATE_LIMIT_PER_HOUR",
        "PUBLIC_MODE",
        "DEMO_PASSCODE",
    }
    assert parsed["AI_BUDGET_USD_DAILY"] == "2" and parsed["AI_BUDGET_USD_TOTAL"] == "10"
    assert parsed["ANTHROPIC_API_KEY"] == "..." and parsed["DEMO_PASSCODE"] == ""


def test_code_example_and_doc_table_agree_on_the_caps():
    """Three-way consistency: budget.py defaults == .env.example values == the docs table."""
    values = _env_example_vars()

    def doc_default(var: str) -> str:
        row = re.search(rf"^\| `{var}` \|[^|]*\| `(\d+)` \(dev\) \|", _DOC, flags=re.M)
        assert row, f"{var} row missing from the 18-runtime-config.md table"
        return row.group(1)

    assert doc_default("AI_BUDGET_USD_DAILY") == values["AI_BUDGET_USD_DAILY"]
    assert doc_default("AI_BUDGET_USD_TOTAL") == values["AI_BUDGET_USD_TOTAL"]
    assert float(values["AI_BUDGET_USD_DAILY"]) == budget._DEV_DEFAULT_DAILY_USD
    assert float(values["AI_BUDGET_USD_TOTAL"]) == budget._DEV_DEFAULT_TOTAL_USD
    # Status line was APPENDED to, not rewritten: the week-8 pins' substrings survive alongside
    # the new alignment note.
    status = next(ln for ln in _DOC.splitlines() if ln.startswith("**Weeks:**"))
    assert "unchanged by week 8" in status and "agent.py" in status
    assert "aligned to the 2/10 dev defaults" in status


# ---------------------------------------------------------------------------
# .gitignore: covers what the code REALLY writes, and nothing it shouldn't
# ---------------------------------------------------------------------------


def test_gitignore_covers_sidecars_the_code_actually_creates(tmp_path, monkeypatch):
    """Capture the real temp / lock names from a live ledger write and check-ignore THOSE.

    Hand-typing ``.Ab12xy.tmp`` proves the glob matches a guess; this proves it matches what
    ``tempfile.mkstemp(prefix=path.name + ".")`` and ``_locked`` produce today, so a future
    rename of the temp scheme breaks this test instead of leaking a file into a commit.
    """
    if not _git_available():
        pytest.skip("git or the repo metadata is unavailable")
    created: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def spy_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        created.append(Path(name).name)
        return fd, name

    monkeypatch.setattr(budget.tempfile, "mkstemp", spy_mkstemp)
    ledger_name = Path(budget._DEFAULT_LEDGER_PATH).name
    ledger = tmp_path / ledger_name
    with budget._locked(ledger):
        budget._write_ledger(ledger, budget._empty_ledger())
    assert created, "_write_ledger no longer goes through tempfile.mkstemp"
    lock_name = next(p.name for p in tmp_path.iterdir() if p.name.endswith(".lock"))
    assert lock_name == ledger_name + ".lock"
    rel_dir = Path(budget._DEFAULT_LEDGER_PATH).parent
    for name in (ledger_name, lock_name, *created):
        assert _is_ignored(str(rel_dir / name)), f"{name} would be committed by `git add -A`"
    # Rate-limit state lives beside the ledger (guardrails derives the path) — same treatment.
    for suffix in ("", ".lock", ".corrupt"):
        assert _is_ignored(str(rel_dir / ("ai_rate_limits.json" + suffix)))


def test_gitignore_globs_do_not_over_match():
    """Adversarial: the new prefix globs must not swallow files that MUST stay committable."""
    if not _git_available():
        pytest.skip("git or the repo metadata is unavailable")
    for rel in (
        ".env.example",  # the onboarding template — `.env.*.local` must not catch it
        "tests/fixtures/example.parquet",  # the negated fixture rule still wins
        "data_cache/other_state.json",  # only the two AI state prefixes are ignored
        "data_cache/ledger.json",  # a *different* ledger name is not covered (and shouldn't be)
    ):
        assert not _is_ignored(rel), f"{rel} is unexpectedly gitignored"
    # And the files this milestone edited are all still tracked.
    tracked = subprocess.run(
        ["git", "ls-files", ".env.example", ".gitignore", "requirements.txt"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    assert sorted(tracked) == [".env.example", ".gitignore", "requirements.txt"]


def test_gitignore_has_no_leftover_exact_ai_state_lines():
    """The four exact lines were REPLACED by two globs, not left beside them (no dead rules)."""
    lines = (_ROOT / ".gitignore").read_text().splitlines()
    assert lines.count("data_cache/ai_ledger.json*") == 1
    assert lines.count("data_cache/ai_rate_limits.json*") == 1
    for stale in (
        "data_cache/ai_ledger.json",
        "data_cache/ai_ledger.json.lock",
        "data_cache/ai_rate_limits.json",
        "data_cache/ai_rate_limits.json.lock",
    ):
        assert stale not in lines
    for keep in (".env", ".env.local", ".env.*.local"):
        assert keep in lines


# ---------------------------------------------------------------------------
# requirements.txt: jsonschema is a real, installable, importable requirement
# ---------------------------------------------------------------------------


def test_jsonschema_requirement_line_parses_and_package_imports():
    lines = [ln.split("#", 1)[0].strip() for ln in _REQUIREMENTS.splitlines()]
    matches = [ln for ln in lines if ln.startswith("jsonschema")]
    assert len(matches) == 1, "jsonschema must appear exactly once"
    req = Requirement(matches[0])
    assert req.name == "jsonschema" and str(req.specifier) == ""  # unpinned, file style
    # The thing the line exists for: the test that imports it can actually import it.
    jsonschema = pytest.importorskip("jsonschema")
    assert hasattr(jsonschema, "validate")
    # And it is the only direct test import that was previously undeclared: every third-party
    # top-level import in tests/test_mcp_server.py is a declared requirement or stdlib.
    declared = {Requirement(ln).name.lower() for ln in lines if ln}
    src = (_ROOT / "tests" / "test_mcp_server.py").read_text()
    imported = set(re.findall(r"^(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", src, flags=re.M))
    third_party = imported - {"quantforge", "conftest"}
    stdlib = set(sys.stdlib_module_names)
    for mod in third_party - stdlib:
        assert mod.lower() in declared, f"{mod} imported by test_mcp_server.py but undeclared"


# ---------------------------------------------------------------------------
# Milestone 2: budget.py token counts accept any numbers.Integral (verifier)
# ---------------------------------------------------------------------------

_HAIKU = "claude-haiku-4-5"


@pytest.fixture
def dev_ledger(tmp_path, monkeypatch):
    """Fresh ledger under tmp_path in dev mode with explicit caps (mirrors tests/test_budget.py)."""
    path = tmp_path / "ledger" / "ai_ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(path))
    monkeypatch.setenv("PUBLIC_MODE", "off")
    monkeypatch.delenv("AI_DISABLED", raising=False)
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "1.0")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "100.0")
    return path


def test_pandas_summed_usage_counters_round_trip_with_hand_computed_ledger(dev_ledger):
    # The realistic source of a numpy count: per-call usage rows summed off a DataFrame.
    usage = pd.DataFrame({"input_tokens": [300_000, 200_000], "output_tokens": [60_000, 40_000]})
    tokens_in, tokens_out = usage["input_tokens"].sum(), usage["output_tokens"].sum()
    assert isinstance(tokens_in, np.integer) and not isinstance(tokens_in, int)

    usd = budget.estimate(_HAIKU, tokens_in, tokens_out)
    # Hand-computed: 500k in @ $1/MTok = $0.50, 100k out @ $5/MTok = $0.50.
    assert usd == 1.0 and type(usd) is float
    budget.charge(usd, model=_HAIKU, tokens_in=tokens_in, tokens_out=tokens_out)

    with open(dev_ledger, encoding="utf-8") as fh:
        bucket = json.load(fh)["days"][budget._utc_today()]
    assert bucket == {"usd": 1.0, "calls": 1, "tokens_in": 500_000, "tokens_out": 100_000}
    assert all(type(bucket[k]) is int for k in ("calls", "tokens_in", "tokens_out"))
    # The gate reads the numpy-charged ledger exactly as an int-charged one: daily cap is $1.0,
    # landing on it is allowed, one cent over is denied.
    assert budget.allow(0.0) is True
    assert budget.allow(0.01) is False
    assert budget.remaining() == {"daily": 0.0, "total": 99.0}


def test_adversarial_dropping_the_coercion_would_fail_inside_json_dump(dev_ledger, monkeypatch):
    """Mutation test: relax the check WITHOUT coercing (return the raw scalar) and prove ``charge``
    then dies in ``json.dump`` after validation passed — the post-spend failure the fix prevents.
    This is what makes ``return int(value)`` load-bearing rather than a style choice."""
    # A nested context so only THIS patch is undone afterwards (a bare ``monkeypatch.undo()``
    # would also drop the fixture's AI_LEDGER_PATH and send the next charge into the repo).
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(budget, "_check_token_count", lambda name, value: value)
        with pytest.raises(TypeError, match="not JSON serializable"):
            budget.charge(0.1, model=_HAIKU, tokens_in=np.int64(7), tokens_out=3)
    # The atomic writer cleaned up after itself: no ledger and no leaked temp file.
    assert not dev_ledger.exists()
    assert not list(dev_ledger.parent.glob("ai_ledger.json.*.tmp"))
    # With the real implementation the identical call succeeds and the file parses.
    budget.charge(0.1, model=_HAIKU, tokens_in=np.int64(7), tokens_out=3)
    assert json.loads(dev_ledger.read_text())["days"][budget._utc_today()]["tokens_in"] == 7


@pytest.mark.parametrize(
    "bad",
    [
        Fraction(3, 1),
        Decimal(3),
        None,
        3 + 0j,
        np.float32(2.0),
        np.int8(-1),
        np.int16(np.iinfo(np.int16).min),
        np.bool_(False),
        [3],
    ],
    ids=[
        "Fraction",
        "Decimal",
        "None",
        "complex",
        "np.float32",
        "np.int8(-1)",
        "np.int16(min)",
        "np.bool_(False)",
        "list",
    ],
)
def test_non_integral_or_negative_values_rejected_with_exact_message_in_every_position(
    bad, dev_ledger
):
    for name, args in (("tokens_in", (bad, 0)), ("tokens_out", (0, bad))):
        with pytest.raises(ValueError) as exc:
            budget.estimate(_HAIKU, *args)
        assert str(exc.value) == f"{name} must be a non-negative int, got {bad!r}"
    with pytest.raises(ValueError, match=r"^tokens_out must be a non-negative int"):
        budget.charge(0.1, model=_HAIKU, tokens_in=0, tokens_out=bad)
    with pytest.raises(ValueError, match=r"^tokens_in must be a non-negative int"):
        budget.charge(0.1, model=_HAIKU, tokens_in=bad, tokens_out=0)
    assert not dev_ledger.exists(), "validation must reject before the ledger is touched"


def test_int_subclasses_and_max_uint64_are_accepted_and_coerced_to_builtin_int():
    class Tokens(IntEnum):
        SMALL = 4

    got = budget._check_token_count("tokens_in", Tokens.SMALL)
    assert got == 4 and type(got) is int  # exactly int, not the IntEnum subclass
    # Coercion happens before arithmetic, so the unbounded Python int carries the value: no
    # numpy overflow/wraparound can distort a cost estimate.
    top = np.uint64(np.iinfo(np.uint64).max)
    assert budget._check_token_count("x", top) == 2**64 - 1
    assert budget.estimate(_HAIKU, top, 0) == (2**64 - 1) / 1e6 * 1.0
    # Zero is a valid count for every integral flavour (a cache-only turn has 0 fresh input).
    assert budget.estimate(_HAIKU, np.uint8(0), np.int64(0)) == 0.0


def test_mixed_int_and_numpy_charges_accumulate_exactly_and_log_coerced_values(dev_ledger, caplog):
    with caplog.at_level(logging.INFO, logger="quantforge.ai.budget"):
        budget.charge(0.125, model=_HAIKU, tokens_in=5, tokens_out=5)
        budget.charge(0.125, model=_HAIKU, tokens_in=np.int64(2**40), tokens_out=np.uint16(65535))
        budget.charge(0.125, model=_HAIKU, tokens_in=np.int32(1), tokens_out=np.uint8(1))
    bucket = json.loads(dev_ledger.read_text())["days"][budget._utc_today()]
    assert bucket["usd"] == 0.375 and bucket["calls"] == 3
    assert bucket["tokens_in"] == 5 + 2**40 + 1 and type(bucket["tokens_in"]) is int
    assert bucket["tokens_out"] == 5 + 65535 + 1 and type(bucket["tokens_out"]) is int
    # The log line formats the coerced ints (the rebinding guarantees log and ledger agree).
    assert f"({2**40} in / 65535 out)" in caplog.text
    # And the module's own strict reader (`isinstance(value, int)`) accepts what it wrote back.
    with budget._locked(dev_ledger):
        assert budget._read_ledger(dev_ledger)["days"][budget._utc_today()] == bucket


def test_concurrent_numpy_typed_charges_sum_exactly(dev_ledger, monkeypatch):
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "100.0")
    n_threads, per_thread = 8, 10

    def worker() -> None:
        for _ in range(per_thread):
            budget.charge(0.125, model=_HAIKU, tokens_in=np.int64(3), tokens_out=np.uint8(2))

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bucket = json.loads(dev_ledger.read_text())["days"][budget._utc_today()]
    assert bucket["usd"] == 10.0 and bucket["calls"] == 80
    assert bucket["tokens_in"] == 240 and bucket["tokens_out"] == 160


def test_allow_untouched_callers_rebind_and_docstrings_reflect_the_spec():
    # `allow` must not have been touched: no token-count validation lives in it.
    assert "_check_token_count" not in inspect.getsource(budget.allow)
    # Every caller of the check rebinds the coerced value (the whole point of returning it).
    for fn in (budget.estimate, budget.charge):
        src = inspect.getsource(fn)
        assert 'tokens_in = _check_token_count("tokens_in", tokens_in)' in src
        assert 'tokens_out = _check_token_count("tokens_out", tokens_out)' in src
    # Whitespace-normalised so a re-wrapped docstring line cannot break the pin.
    charge_doc = " ".join(budget.charge.__doc__.split())
    assert "numbers.Integral" in charge_doc and "numpy ints included" in charge_doc
    assert "ints, not bools" not in charge_doc
    check_doc = " ".join(budget._check_token_count.__doc__.split())
    assert "numbers.Integral" in check_doc and "JSON" in check_doc
    # Signature advertises the coercion.
    assert inspect.signature(budget._check_token_count).return_annotation in ("int", int)


# ---------------------------------------------------------------------------
# Milestone 3: agent.py — minimal helpers (tests/ is not a package; copied from test_agent.py)
# ---------------------------------------------------------------------------

_AGENT_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM"]
_AGENT_GOAL = "find a momentum variant with Sharpe > 1 on validation"
_ONE_TOOL = [
    {
        "name": "declare_done",
        "description": "x",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    }
]


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


def _propose(strategy, params, tool_id="toolu_1"):
    data = {"strategy": strategy, "params": params, "rationale": "try it"}
    return _tool_use_response("propose_experiment", data, tool_id)


def _done(tool_id="toolu_done"):
    return _tool_use_response("declare_done", {"reason": "converged"}, tool_id)


class _FakeClient:
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


class _TripwireClient:
    """A client whose ``messages.create`` must never run: proves the refusal came first."""

    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)

    @staticmethod
    def _create(**kwargs):
        raise AssertionError("messages.create was reached")


def _agent_synthetic_long(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2026-06-30", tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), len(_AGENT_TICKERS)))
    wide = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=_AGENT_TICKERS
    )
    return interchange.to_long(wide, "prices")


_AGENT_SYNTHETIC = _agent_synthetic_long()


@pytest.fixture
def agent_env(tmp_path, monkeypatch):
    """Synthetic prices, temp ledger, dev env, NO key, real ``_get_client`` behind a tripwire.

    Unlike the builder's fixture this leaves ``agent._get_client`` alone and replaces only the
    SDK constructor it calls, so the "client never built" claim is tested through the real code
    path rather than a stub of it.
    """
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _AGENT_SYNTHETIC)
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
    constructed: list[object] = []

    def tripwire(*args, **kwargs):
        constructed.append(kwargs)
        raise AssertionError("anthropic.Anthropic() was constructed")

    monkeypatch.setattr(agent.anthropic, "Anthropic", tripwire)
    yield SimpleNamespace(ledger=ledger, constructed=constructed)
    mcp_server.reset_registry()


# ---------------------------------------------------------------- (5) _call: empty tools


@pytest.mark.parametrize("tools", [[], None], ids=["empty-list", "None"])
def test_call_refuses_missing_tools_before_create_and_before_charge(agent_env, monkeypatch, tools):
    charged: list[float] = []
    monkeypatch.setattr(budget, "charge", lambda usd, **kw: charged.append(usd))
    with pytest.raises(ValueError, match="at least one tool"):
        agent._call(
            _TripwireClient(),
            system="s",
            messages=[{"role": "user", "content": "x"}],
            tools=tools,
            tool_choice=agent._TOOL_CHOICE,
            max_tokens=10,
        )
    assert charged == [], "a refused call must not reach budget.charge"
    assert not agent_env.ledger.exists()


def test_call_with_exactly_one_tool_still_sends_marks_it_and_charges_hand_computed_usd(agent_env):
    client = _FakeClient(_done())
    response, usd = agent._call(
        client,
        system="s",
        messages=[{"role": "user", "content": "x"}],
        tools=_ONE_TOOL,
        tool_choice=agent._TOOL_CHOICE,
        max_tokens=10,
    )
    # 1000 input tokens at $2/MTok + 200 output tokens at $10/MTok = $0.002 + $0.002.
    assert usd == pytest.approx(0.004)
    (req,) = client.calls
    assert len(req["tools"]) == 1
    assert req["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in _ONE_TOOL[0], "the caller's tool list must not be mutated"
    bucket = json.loads(agent_env.ledger.read_text())["days"][budget._utc_today()]
    assert bucket["usd"] == pytest.approx(0.004) and bucket["calls"] == 1


def test_call_source_guard_precedes_request_construction():
    src = inspect.getsource(agent._call)
    assert src.index("if not tools:") < src.index("cached_tools = [")
    assert src.index("if not tools:") < src.index("client.messages.create(")
    assert src.count("messages.create(") == 1
    assert Path(agent.__file__).read_text().count("messages.create(") == 1


# ---------------------------------------------------------------- (6) None Sharpe


def _rec(iter_, sharpe, error=None):
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


def test_none_nan_and_neg_inf_all_rank_minus_inf_and_tie_break_to_earliest_iter():
    for bad in (None, float("nan"), -math.inf, math.inf):
        assert agent._val_sharpe({"val_metrics": {"sharpe": bad}}) == -math.inf
    # Same key for all three -> the earliest iter wins, whichever kind it is.
    history = [_rec(3, float("nan")), _rec(1, None), _rec(2, -math.inf)]
    assert agent._best_record(history)["iter"] == 1
    history = [_rec(2, None), _rec(1, float("nan"))]
    assert agent._best_record(history)["iter"] == 1


def test_finite_negative_sharpe_beats_a_none_sharpe():
    best = agent._select_best([_rec(1, None), _rec(2, -3.0)])
    assert best["val_metrics"]["sharpe"] == -3.0
    # And the returned config is a copy, not an alias into the history.
    history = [_rec(1, None), _rec(2, 0.4)]
    best = agent._select_best(history)
    best["params"]["lookback"] = 999
    assert history[1]["params"]["lookback"] == 60


def test_history_round_tripped_through_round_metrics_and_json_ranks_without_type_error():
    # The realistic path: a NaN Sharpe -> _round_metrics (null) -> json -> None on reload.
    raw = [
        _rec(1, float("nan")),
        _rec(2, 0.9),
        _rec(3, float("inf")),
        _rec(4, 0.9),  # tie with iter 2 -> iter 2 must win
    ]
    for r in raw:
        r["val_metrics"] = agent._round_metrics(r["val_metrics"])
        r["train_metrics"] = agent._round_metrics(r["train_metrics"])
    reloaded = json.loads(json.dumps(raw))
    assert reloaded[0]["val_metrics"]["sharpe"] is None
    assert reloaded[2]["val_metrics"]["sharpe"] is None
    best = agent._best_record(reloaded)
    assert best["iter"] == 2 and best["val_metrics"]["sharpe"] == 0.9


def test_val_sharpe_still_rejects_a_non_numeric_string():
    # The guard is for None only; a corrupt value must still fail loudly, not rank silently.
    with pytest.raises((TypeError, ValueError)):
        agent._val_sharpe({"val_metrics": {"sharpe": "n/a"}})


# ---------------------------------------------------------------- (7) engine / tickers pre-spend


@pytest.mark.parametrize(
    "engine",
    [["python"], None, 1, b"python", ("python",), {"python": 1}, "R"],
    ids=["list", "None", "int", "bytes", "tuple", "dict", "unknown-str"],
)
def test_bad_engine_raises_value_error_before_client_data_or_spend(agent_env, engine):
    with pytest.raises(ValueError, match="registered engines") as exc:
        agent.run_research(_AGENT_GOAL, engine=engine, client=None)
    # The documented message names what was passed and the live registry.
    assert repr(engine) in str(exc.value) and "'python'" in str(exc.value)
    assert agent_env.constructed == [] and agent._client is None
    assert mcp_server._DATASETS == {}
    assert not agent_env.ledger.exists()


@pytest.mark.parametrize(
    "tickers, message",
    [
        (["AAPL", "ZZZZ"], "not in the fixed universe"),
        (["ZZZZ"], "not in the fixed universe"),
        (["AAPL", "AAPL"], "duplicate"),
        ([], "non-empty list"),
        (("AAPL",), "non-empty list"),
        ({"AAPL": 1}, "non-empty list"),
        ("AAPL", "non-empty list"),
        ([["AAPL"]], "must be strings"),
        ([None], "must be strings"),
        (["aapl"], "not in the fixed universe"),
    ],
    ids=[
        "unknown-2nd",
        "unknown-only",
        "duplicate",
        "empty",
        "tuple",
        "dict",
        "str",
        "nested-list",
        "None-element",
        "lowercase",
    ],
)
def test_bad_tickers_fail_with_tool_message_through_real_get_client_path(
    agent_env, tickers, message
):
    with pytest.raises(ValueError, match=message):
        agent.run_research(_AGENT_GOAL, tickers=tickers, client=None)
    assert agent_env.constructed == [], "the SDK constructor must never be reached"
    assert agent._client is None
    assert mcp_server._DATASETS == {}
    assert not agent_env.ledger.exists()


def test_valid_subset_tickers_pass_the_pre_check_and_reach_the_loop(agent_env):
    # The new early check must not reject a legitimate subset (regression guard on item 7).
    client = _FakeClient(_propose("momentum", {"lookback": 60}), _done())
    out = agent.run_research(_AGENT_GOAL, tickers=["AAPL", "MSFT"], max_iters=3, client=client)
    assert out["stopped_because"] == "converged" and out["best"] is not None
    assert agent_env.constructed == []  # a caller-supplied client is used, never built
    # Registry values are the wide price frames; both windows carry only the subset.
    assert len(mcp_server._DATASETS) == 2
    for wide in mcp_server._DATASETS.values():
        assert sorted(wide.columns) == ["AAPL", "MSFT"]


def test_validation_precedes_client_construction_even_when_a_client_is_supplied(agent_env):
    # A queued proposal proves that a bad engine short-circuits before the first model call.
    client = _FakeClient(_propose("momentum", {"lookback": 60}), _done())
    with pytest.raises(ValueError, match="registered engines"):
        agent.run_research(_AGENT_GOAL, engine=["python"], client=client)
    assert client.calls == []
    with pytest.raises(ValueError, match="fixed universe"):
        agent.run_research(_AGENT_GOAL, tickers=["ZZZZ"], client=client)
    assert client.calls == []
    assert not agent_env.ledger.exists()


# ---------------------------------------------------------------- (8) tool_choice / thinking


_SONNET5_REJECTED_OR_SHAPE_CHANGING = ("thinking", "temperature", "top_p", "top_k", "output_config")


def test_request_shape_is_exact_and_tool_choice_is_a_faithful_non_aliased_copy(agent_env):
    client = _FakeClient(_propose("momentum", {"lookback": 60}), _done())
    before = json.dumps(agent._TOOL_CHOICE, sort_keys=True)
    agent.run_research(_AGENT_GOAL, max_iters=3, client=client)
    assert len(client.calls) == 2
    for req in client.calls:
        assert set(req) == {"model", "max_tokens", "system", "tools", "tool_choice", "messages"}
        for key in _SONNET5_REJECTED_OR_SHAPE_CHANGING:
            assert key not in req
        assert req["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
        assert req["model"] == agent.MODEL == "claude-sonnet-5"
        assert req["model"] in budget.PRICES_PER_MTOK
    # The constant survived the run unchanged (a request-side mutation could not leak back).
    assert json.dumps(agent._TOOL_CHOICE, sort_keys=True) == before
    # Per the claude-api skill, forced `any`/`tool` returns 400 only on these two ids.
    assert not agent.MODEL.startswith(("claude-fable-5-1", "claude-mythos-5-1"))


def test_call_and_loop_sources_never_set_thinking_or_sampling_knobs():
    for fn in (agent._call, agent._research_loop):
        src = inspect.getsource(fn)
        for key in _SONNET5_REJECTED_OR_SHAPE_CHANGING:
            assert f'"{key}"' not in src and f"'{key}'" not in src, f"{fn.__name__} sets {key}"
    # The decision and its provenance are written where the next reader will look.
    doc = " ".join(agent._call.__doc__.split())
    assert "claude-api" in doc and "Bedrock" in doc and "model-migration" in doc


# ---------------------------------------------------------------- (12) holdout cost is fixed


class _RecordingEngine:
    """Wraps the registered engine, keeping the (prices, positions, params) of every run."""

    def __init__(self, inner):
        self._inner = inner
        self.runs: list[tuple[pd.DataFrame, pd.DataFrame, dict]] = []

    def run_backtest(self, prices, positions, params):
        self.runs.append((prices.copy(), positions.copy(), dict(params)))
        return self._inner.run_backtest(prices, positions, params)


def test_holdout_at_guardrail_cost_matches_hand_scoring_and_differs_from_loop_cost(
    agent_env, monkeypatch
):
    real_engine = mcp_server.ENGINES["python"]
    spy = _RecordingEngine(real_engine)
    monkeypatch.setitem(mcp_server.ENGINES, "python", spy)
    client = _FakeClient(_propose("momentum", {"lookback": 60}), _done())

    # cost_bps=0.0 is the most flattering allowed loop cost; the holdout must ignore it.
    out = agent.run_research(_AGENT_GOAL, max_iters=3, cost_bps=0.0, client=client)
    assert out["best"] is not None and out["holdout_metrics"] is not None
    assert len(spy.runs) == 3
    train_params, val_params, holdout_params = (p for _, _, p in spy.runs)
    assert train_params["cost_bps"] == 0.0 and val_params["cost_bps"] == 0.0
    assert holdout_params["cost_bps"] == guardrails._HOLDOUT_COST_BPS == 10.0

    # Independent hand-scoring on a FRESH handle at the guardrail cost reproduces the number.
    monkeypatch.setitem(mcp_server.ENGINES, "python", real_engine)
    _, _, handle = guardrails.split_data(_AGENT_SYNTHETIC)
    by_hand = guardrails.score_holdout(handle, "momentum", {"lookback": 60}, "python")
    assert out["holdout_metrics"] == pytest.approx({k: float(v) for k, v in by_hand.items()})

    # And re-running the very same holdout inputs at the loop's 0 bps gives a DIFFERENT (and
    # rosier) Sharpe, so the pin is on a number the cost actually moves.
    prices, positions, _ = spy.runs[-1]
    at_zero = real_engine.run_backtest(
        prices, positions, {**holdout_params, "cost_bps": 0.0}
    ).metrics
    assert at_zero["sharpe"] != pytest.approx(out["holdout_metrics"]["sharpe"])
    assert at_zero["sharpe"] > out["holdout_metrics"]["sharpe"]
    # The holdout window itself is data the loop never saw.
    holdout_lo = pd.Timestamp(loader.get_split_bounds()["validation"][1], tz="UTC")
    assert prices.index.min() > holdout_lo


def test_run_research_docstring_and_agent_doc_state_the_fixed_holdout_cost():
    doc = " ".join(agent.run_research.__doc__.split())
    assert "guardrails._HOLDOUT_COST_BPS" in doc and "regardless of the cost_bps" in doc.replace(
        "``", ""
    )
    assert "tickers are validated first" in doc
    md = (_ROOT / "docs" / "components" / "13-ai-agent.md").read_text()
    decisions = md.split("## Decisions made in build")[1].split("\n## ")[0]
    assert "_HOLDOUT_COST_BPS" in decisions and "tool_choice" in decisions
    assert "Bedrock" in decisions and "claude-fable-5-1" in decisions


# ---------------------------------------------------------------------------
# Milestone 4: nl_interface — query gate before the rate limit; non-list tickers (verifier)
# ---------------------------------------------------------------------------

_NL_ESSENTIALS = {"strategy": "momentum", "start": "2015-01-01", "end": "2019-12-31"}


def _nl_plan(plan: dict):
    """A canned parse response: one ``plan_backtest`` tool_use block."""
    return _tool_use_response(nl_interface.PLAN_TOOL["name"], plan, "toolu_nl")


def _nl_parse_usd() -> float:
    """Hand-computed cost of ``_tool_use_response``'s fixed usage (1000 in / 200 out) on Haiku."""
    in_rate, out_rate = budget.PRICES_PER_MTOK[nl_interface.MODEL]
    return 1000 / 1e6 * in_rate + 200 / 1e6 * out_rate


@pytest.fixture
def nl_env(tmp_path, monkeypatch):
    """Synthetic prices, EMPTY temp state dir, dev env, no key, SDK constructor tripwired.

    Like ``agent_env`` this leaves ``nl_interface._get_client`` alone and replaces only the SDK
    constructor, so any path that would build a real client fails loudly through the real code.
    """
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _AGENT_SYNTHETIC)
    mcp_server.reset_registry()
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

    def tripwire(*args, **kwargs):
        raise AssertionError("anthropic.Anthropic() was constructed")

    monkeypatch.setattr(nl_interface.anthropic, "Anthropic", tripwire)
    yield SimpleNamespace(ledger=ledger, rate_state=ledger.with_name("ai_rate_limits.json"))
    mcp_server.reset_registry()


def _stamps(rate_state: Path, key: str) -> list[float]:
    if not rate_state.exists():
        return []
    return json.loads(rate_state.read_text()).get(key, [])


def _today_spend(ledger: Path) -> float | None:
    if not ledger.exists():
        return None
    bucket = json.loads(ledger.read_text())["days"].get(budget._utc_today())
    return None if bucket is None else float(bucket["usd"])


class _Str(str):
    """A ``str`` subclass: must be treated as a string by both gates."""


_GATE_BATTERY = [
    "x" * 2000,  # exactly the limit: admitted
    "x" * 2001,  # one over: clarify
    " " * 1999 + "x",  # 2000 chars, mostly whitespace: admitted
    " " * 2000 + "x",  # 2001 chars: clarify (over-long, not blank)
    " x ",  # padded: admitted (len counts the padding; strip only decides blankness)
    "　 \t\n",  # Unicode whitespace only: blank -> clarify
    "é" * 2000,  # multi-byte code points at the limit: admitted (len is code points)
    "é" * 2001,  # and one over: clarify
    _Str("q"),  # str subclass: admitted
    _Str(""),  # str subclass, blank: clarify
]


class _RateTripped(Exception):
    """Sentinel: ``handle`` reached ``guardrails.rate_limit`` (the query gate passed)."""


@pytest.mark.parametrize("query", _GATE_BATTERY, ids=lambda q: f"len{len(q)}-{type(q).__name__}")
def test_handle_gate_predicate_is_exactly_validate_query(nl_env, monkeypatch, query):
    # Any disagreement between the two predicates would re-create the original bug one layer
    # down: a query that passes handle's gate, consumes a slot, then raises in _parse_call.
    def tripped(key):
        raise _RateTripped()

    monkeypatch.setattr(guardrails, "rate_limit", tripped)
    try:
        nl_interface._validate_query(query)
        lower_accepts = True
    except ValueError:
        lower_accepts = False

    if lower_accepts:
        with pytest.raises(_RateTripped):
            nl_interface.handle(query, session_key="k", client=_TripwireClient())
    else:
        out = nl_interface.handle(query, session_key="k", client=_TripwireClient())
        assert out["plan"] == {"clarify": nl_interface._CLARIFY_EMPTY_QUERY}
        assert out["fallback"] is None and out["spend_usd"] == 0.0 and out["metrics"] is None
        assert not nl_env.ledger.exists() and not nl_env.rate_state.exists()


def test_bad_query_never_resolves_the_state_path_at_all(nl_env, monkeypatch):
    # Both the ledger and the derived rate-state path go through budget._ledger_path. Tripping
    # it proves neither gate got as far as asking WHERE its state lives.
    def tripped():
        raise AssertionError("budget._ledger_path() was called")

    monkeypatch.setattr(budget, "_ledger_path", tripped)
    for bad in ("", "\t\n", "x" * 2001):
        out = nl_interface.handle(bad, session_key="k", client=_TripwireClient())
        assert out["plan"] == {"clarify": nl_interface._CLARIFY_EMPTY_QUERY}
        assert out["explanation"] == nl_interface._CLARIFY_EMPTY_QUERY
        assert out["fallback"] is None and out["spend_usd"] == 0.0
    # The lower-level API keeps its contract and is equally free.
    with pytest.raises(ValueError):
        nl_interface.parse_with_spend("", client=_TripwireClient())


def test_boundary_2000_is_admitted_and_charged_while_2001_is_free(nl_env):
    client = _FakeClient(_nl_plan({"clarify": "Which strategy?"}))
    out = nl_interface.handle("x" * 2000, session_key="k", client=client)
    assert out["plan"] == {"clarify": "Which strategy?"} and out["fallback"] is None
    assert len(client.calls) == 1
    assert client.calls[0]["messages"] == [{"role": "user", "content": "x" * 2000}]
    assert out["spend_usd"] == pytest.approx(_nl_parse_usd())
    assert _today_spend(nl_env.ledger) == pytest.approx(_nl_parse_usd())
    assert len(_stamps(nl_env.rate_state, "k")) == 1

    over = nl_interface.handle("x" * 2001, session_key="k", client=_TripwireClient())
    assert over["plan"] == {"clarify": nl_interface._CLARIFY_EMPTY_QUERY}
    assert over["spend_usd"] == 0.0
    assert _today_spend(nl_env.ledger) == pytest.approx(_nl_parse_usd())  # unchanged
    assert len(_stamps(nl_env.rate_state, "k")) == 1  # unchanged


def test_interleaved_bad_queries_leave_exactly_the_valid_stamps(nl_env, monkeypatch):
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "2")
    sequence = ["", "q1", "   ", "q2", "x" * 2001, "q3", ""]
    outcomes = []
    for query in sequence:
        client = _FakeClient(_nl_plan({"clarify": "Which strategy?"}))
        out = nl_interface.handle(query, session_key="k", client=client)
        outcomes.append((out["fallback"], out["plan"], len(client.calls)))
    free = (None, {"clarify": nl_interface._CLARIFY_EMPTY_QUERY}, 0)
    admitted = (None, {"clarify": "Which strategy?"}, 1)
    limited = ("rate", None, 0)
    assert outcomes == [free, admitted, free, admitted, free, limited, free]
    assert len(_stamps(nl_env.rate_state, "k")) == 2
    assert _today_spend(nl_env.ledger) == pytest.approx(2 * _nl_parse_usd())


def test_bad_query_result_is_a_fresh_dict_each_call(nl_env):
    first = nl_interface.handle("", session_key="k", client=_TripwireClient())
    first["plan"]["clarify"] = "mutated"
    second = nl_interface.handle("", session_key="k", client=_TripwireClient())
    assert second["plan"] == {"clarify": nl_interface._CLARIFY_EMPTY_QUERY}
    assert nl_interface._CLARIFY_EMPTY_QUERY.endswith("2000 characters.")


@pytest.mark.parametrize("bad", [None, 0, 1.5, b"", bytearray(b"q"), ("q",), {"q": 1}])
def test_non_string_query_raises_before_the_state_path_is_resolved(nl_env, monkeypatch, bad):
    def tripped():
        raise AssertionError("budget._ledger_path() was called")

    monkeypatch.setattr(budget, "_ledger_path", tripped)
    with pytest.raises(ValueError, match="query must be a string"):
        nl_interface.handle(bad, session_key="k", client=_TripwireClient())


# ---------------------------------------------------------------- _raw_plan tickers, end-to-end


def _trip_load_data(monkeypatch):
    def tripped(*args, **kwargs):
        raise AssertionError(f"load_data reached with {args!r}")

    monkeypatch.setattr(mcp_server, "load_data", tripped)


def test_dict_tickers_with_valid_keys_no_longer_runs_a_backtest_the_user_never_asked_for(
    nl_env, monkeypatch
):
    # The dangerous case: list({"AAPL": 1, "MSFT": 2}) == ["AAPL", "MSFT"] is a VALID ticker
    # list, so the mangling would not have been caught downstream — a backtest would have run on
    # tickers the model never put in an array.
    _trip_load_data(monkeypatch)
    client = _FakeClient(_nl_plan({**_NL_ESSENTIALS, "tickers": {"AAPL": 1, "MSFT": 2}}))
    out = nl_interface.handle("backtest momentum", session_key="k", client=client)
    assert set(out["plan"]) == {"clarify"} and "tickers" in out["plan"]["clarify"]
    assert out["explanation"] == out["plan"]["clarify"]
    assert out["fallback"] is None and out["metrics"] is None
    assert len(client.calls) == 1  # parse only; no explain on a clarification
    assert out["spend_usd"] == pytest.approx(_nl_parse_usd())
    assert _today_spend(nl_env.ledger) == pytest.approx(_nl_parse_usd())
    assert mcp_server._DATASETS == {} and mcp_server._RESULTS == {}  # nothing was loaded or run


def test_str_tickers_through_handle_is_a_clarification_not_the_old_invalid_fallback(
    nl_env, monkeypatch
):
    # Before the fix "AAPL" became ["A", "A", "P", "L"], reached load_data, and surfaced as
    # fallback "invalid" (unknown ticker 'A'). Now load_data is never reached.
    _trip_load_data(monkeypatch)
    client = _FakeClient(_nl_plan({**_NL_ESSENTIALS, "tickers": "AAPL"}))
    out = nl_interface.handle("backtest momentum on AAPL", session_key="k", client=client)
    assert out["fallback"] is None
    assert set(out["plan"]) == {"clarify"}
    assert "'A'" not in out["explanation"]


def test_non_list_tickers_clarify_holds_in_public_mode_too(nl_env, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    monkeypatch.setenv("AI_BUDGET_USD_DAILY", "2")
    monkeypatch.setenv("AI_BUDGET_USD_TOTAL", "10")
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "5")
    _trip_load_data(monkeypatch)
    client = _FakeClient(_nl_plan({**_NL_ESSENTIALS, "params": {"lookback": 60}, "tickers": 7}))
    out = nl_interface.handle("backtest momentum", session_key="k", client=client)
    assert out["fallback"] is None and set(out["plan"]) == {"clarify"}
    assert len(client.calls) == 1


@pytest.mark.parametrize("tickers", [("AAPL", "MSFT"), frozenset({"AAPL"}), {"AAPL"}], ids=repr)
def test_raw_plan_rejects_every_non_list_container_even_with_valid_members(tickers):
    out = nl_interface._raw_plan(_nl_plan({**_NL_ESSENTIALS, "tickers": tickers}))
    assert set(out) == {"clarify"}


def test_raw_plan_precedence_clarify_then_essentials_then_tickers():
    # An explicit model clarification wins over everything.
    out = nl_interface._raw_plan(_nl_plan({**_NL_ESSENTIALS, "clarify": "Dates?", "tickers": 7}))
    assert out == {"clarify": "Dates?"}
    # A missing essential is reported as unparseable, not as a tickers problem.
    out = nl_interface._raw_plan(
        _nl_plan({"strategy": "momentum", "start": "2015-01-01", "tickers": 7})
    )
    assert out == {"clarify": nl_interface._CLARIFY_UNPARSEABLE}
    # Only when everything essential is present does the tickers shape decide.
    out = nl_interface._raw_plan(_nl_plan({**_NL_ESSENTIALS, "tickers": 7}))
    assert out == {"clarify": nl_interface._CLARIFY_TICKERS_NOT_LIST}
    assert out["clarify"] != nl_interface._CLARIFY_UNPARSEABLE


def test_raw_plan_list_tickers_are_copied_not_aliased():
    given = ["AAPL", "MSFT"]
    out = nl_interface._raw_plan(_nl_plan({**_NL_ESSENTIALS, "tickers": given}))
    assert out["tickers"] == given and out["tickers"] is not given


def test_clarify_constant_is_derived_from_the_limit_not_retyped():
    src = inspect.getsource(nl_interface)
    # The only literal "2000" in the module is the constant's own definition (and a comment).
    literal_lines = [
        ln for ln in src.splitlines() if "2000" in ln and not ln.lstrip().startswith("#")
    ]
    assert literal_lines == ["_MAX_QUERY_CHARS = 2000"]
    assert "{_MAX_QUERY_CHARS}" in src.split("_CLARIFY_EMPTY_QUERY = (")[1].split(")")[0]


def test_nl_doc_flow_comment_and_design_note_both_record_the_gate_order():
    md = (_ROOT / "docs" / "components" / "12-nl-interface.md").read_text()
    flow = md.split("def handle(")[1].split("def parse(")[0]
    assert (
        flow.index("query shape")
        < flow.index("rate_limit(session_key)")
        < flow.index("budget.allow")
    )
    assert "records no" in flow and "slot" in flow
    notes = md.split("## Design notes")[1].split("## Decisions made in build")[0]
    assert "query validation → rate limit → budget" in notes
    assert "_raw_plan" in notes and "non-list" in notes
    # The fallback vocabulary is unchanged: a clarification is still `None`, not a new code.
    table = md.split("### Fallback vocabulary")[1].split("## Design notes")[0]
    codes = re.findall(r"^\| `([a-z_]+)` \|", table, flags=re.M)
    assert codes == ["rate", "budget", "public_mode", "invalid", "api_error"]


# ---------------------------------------------------------------------------
# Milestone 5: mcp_server + guardrails doc/docstring corrections (verifier)
# ---------------------------------------------------------------------------
#
# The builder's own tests pin strings. These prove the CLAIMS the new prose makes are true of
# the running code, and that nothing executable moved.

_M5_ROOT_SRC = _ROOT / "src" / "quantforge" / "ai"
_M5_TICKERS = ["AAPL", "MSFT", "NVDA"]


def _m5_synthetic_long(seed: int = 7) -> pd.DataFrame:
    """Long interchange 'prices' frame on real universe tickers (test_holdout_isolation pattern)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2026-06-30", tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(len(dates), len(_M5_TICKERS)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return interchange.to_long(pd.DataFrame(prices, index=dates, columns=_M5_TICKERS), "prices")


_M5_PRICES = _m5_synthetic_long()


@pytest.fixture
def m5_public(monkeypatch) -> str:
    """dataset_id for a loaded synthetic window, PUBLIC_MODE=on, loader never consulted."""
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _M5_PRICES)
    monkeypatch.setenv("PUBLIC_MODE", "on")
    mcp_server.reset_registry()
    ds = mcp_server.load_data("2015-01-01", "2019-12-31", _M5_TICKERS)["dataset_id"]
    yield ds
    mcp_server.reset_registry()


# ---------------------------------------------------------------- text-only proof vs HEAD


def _blank_all_strings(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.dump(tree)


def _non_docstring_strings(source: str) -> list[str]:
    """Every str constant in evaluation order, with docstrings blanked so they do not count."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                first.value.value = ""
    return [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


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


def _head_source(rel: str) -> str:
    """The file as it was at the baseline commit (see _baseline_ref), not at HEAD."""
    base = _baseline_ref()
    if base is None:
        pytest.skip("git or the baseline commit is unavailable")
    return subprocess.run(
        ["git", "show", f"{base}:{rel}"], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout


@pytest.mark.skipif(not _git_available(), reason="needs git + HEAD to diff against")
@pytest.mark.parametrize(
    "rel, expected_changed",
    [
        ("src/quantforge/ai/guardrails.py", []),
        (
            "src/quantforge/ai/mcp_server.py",
            [
                "Handles returned by run_backtest (at least two, unique). Mutually exclusive "
                "with dataset_id: pass exactly one of the two.",
                "Handle returned by load_data (needs >= 2 tickers). Mutually exclusive with "
                "result_ids: pass exactly one of the two.",
            ],
        ),
    ],
)
def test_milestone_is_text_only_ast_equals_head_with_strings_blanked(rel, expected_changed):
    head, cur = _head_source(rel), (_ROOT / rel).read_text()
    a, b = _blank_all_strings(head), _blank_all_strings(cur)
    if a != b:  # pragma: no cover - diagnostic only
        diff = "\n".join(difflib.unified_diff(a.split("("), b.split("("), lineterm="", n=1))
        pytest.fail(f"executable AST of {rel} differs from HEAD:\n{diff[:4000]}")
    # Same structure; now the only non-docstring string constants that changed are the two
    # selector descriptions (mcp_server) or nothing at all (guardrails).
    before, after = _non_docstring_strings(head), _non_docstring_strings(cur)
    assert len(before) == len(after)
    changed = [y for x, y in zip(before, after, strict=True) if x != y]
    assert changed == expected_changed
    # And those descriptions are plain ASCII — the cached prompt prefix gained no odd bytes.
    assert all(s.isascii() for s in expected_changed)


def test_tool_schemas_are_rebuilt_identically_and_advertised_unchanged():
    # The prefix the NL interface caches is the module constant; a rebuild must be byte-equal.
    assert json.dumps(mcp_server._build_tool_schemas()) == json.dumps(mcp_server.TOOL_SCHEMAS)
    server = mcp_server.build_server()
    listed = {t.name: t.inputSchema for t in asyncio.run(server.list_tools())}
    wanted = next(s for s in mcp_server.TOOL_SCHEMAS if s["name"] == "optimize_portfolio")
    assert listed["optimize_portfolio"] == wanted["input_schema"]  # the new text goes over the wire


# ---------------------------------------------------------------- validation order with a SPY


@pytest.fixture
def gate_spy(monkeypatch):
    """Wrap the real gate so every call it receives is recorded (the docstring's 'never sees')."""
    real = guardrails.assert_no_codegen
    calls: list[dict] = []

    def spy(action):
        calls.append({"keys": set(action), "action": dict(action)})
        return real(action)

    monkeypatch.setattr(guardrails, "assert_no_codegen", spy)
    return calls


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"strategy": "momentum", "params": ["code"]}, "params must be a dict"),
        ({"strategy": "momentum", "params": ("code",)}, "params must be a dict"),
        ({"strategy": "momentum", "params": "lookback=20"}, "params must be a dict"),
        ({"strategy": ["momentum"], "params": {}}, "strategy must be a string"),
        ({"strategy": b"momentum", "params": {}}, "strategy must be a string"),
        ({"strategy": None, "params": None}, "strategy must be a string"),
    ],
)
def test_malformed_shapes_never_reach_the_gate(m5_public, gate_spy, kwargs, match):
    with pytest.raises(ValueError, match=match) as info:
        mcp_server.run_backtest(m5_public, **kwargs)
    assert type(info.value) is ValueError
    assert gate_spy == []  # not "raised the right type" — the gate was never invoked at all
    assert mcp_server._RESULTS == {}


def test_unknown_dataset_never_reaches_the_gate_even_with_a_codegen_payload(gate_spy, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "on")
    mcp_server.reset_registry()
    with pytest.raises(ValueError, match="unknown dataset_id") as info:
        mcp_server.run_backtest("ds_nope", strategy="code", params={"code": "x"})
    assert type(info.value) is ValueError
    assert gate_spy == []


def test_codegen_payload_hits_the_gate_exactly_once_with_exactly_strategy_and_params(
    m5_public, gate_spy
):
    # Engine and cost are ALSO invalid here: the gate must win over both (it runs first).
    with pytest.raises(guardrails.PublicModeViolation) as info:
        mcp_server.run_backtest(
            m5_public, strategy="momentum", params={"code": "x"}, engine="nope", cost_bps=999
        )
    assert "code" in str(info.value)
    assert len(gate_spy) == 1
    assert gate_spy[0]["keys"] == {"strategy", "params"}
    assert gate_spy[0]["action"] == {"strategy": "momentum", "params": {"code": "x"}}
    assert mcp_server._RESULTS == {}


def test_gate_passes_then_engine_and_cost_are_checked_in_that_order(m5_public, gate_spy):
    good = {"lookback": 60, "top_n": 2}
    with pytest.raises(ValueError, match="unknown engine"):
        mcp_server.run_backtest(m5_public, "momentum", good, engine="nope", cost_bps=999)
    assert len(gate_spy) == 1  # the gate ran and PASSED; the engine check came after it
    with pytest.raises(ValueError, match="cost_bps"):
        mcp_server.run_backtest(m5_public, "momentum", good, engine="python", cost_bps=999)
    assert len(gate_spy) == 2
    assert mcp_server._RESULTS == {}


def test_gate_is_a_no_op_outside_public_mode_but_still_called(m5_public, gate_spy, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "off")
    with pytest.raises(ValueError) as info:
        mcp_server.run_backtest(m5_public, "momentum", {"code": "x"})
    assert type(info.value) is ValueError  # the whitelist, not the gate, rejected it
    assert len(gate_spy) == 1  # "a no-op outside public mode" — invoked, returns None


def test_docstring_numbered_steps_are_one_to_seven_in_body_order():
    doc = inspect.getdoc(mcp_server.run_backtest)
    steps = [int(n) for n in re.findall(r"\((\d)\)", doc)]
    assert steps == [1, 2, 3, 4, 5, 6, 7]
    named = [
        "_get_dataset",
        "params",
        "strategy",
        "assert_no_codegen",
        "validate_params",
        "ENGINES",
        "_validate_cost_bps",
    ]
    positions = [doc.index(f"({i})") for i in range(1, 8)]
    for i, name in enumerate(named):
        segment = doc[positions[i] : positions[i + 1] if i < 6 else len(doc)]
        assert name in segment, f"step {i + 1} should mention {name}"


# ---------------------------------------------------------------- (adversarial) schema vs rule


def _optimize_schema() -> dict:
    return next(s for s in mcp_server.TOOL_SCHEMAS if s["name"] == "optimize_portfolio")[
        "input_schema"
    ]


def _wire_call(name: str, args: dict) -> tuple[dict | None, str | None]:
    server = mcp_server.build_server()
    try:
        out = asyncio.run(server.call_tool(name, args))
    except ToolError as exc:
        return None, str(exc)
    if getattr(out, "isError", False):
        return None, "".join(getattr(c, "text", "") for c in out.content)
    return {"ok": True}, None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"result_ids": ["res_a", "res_b"], "dataset_id": "ds_x"},
        {"result_ids": ["res_a", "res_b"], "dataset_id": "ds_x", "objective": "min_volatility"},
    ],
)
def test_schema_accepts_both_and_neither_selector_so_the_rule_must_live_in_prose(payload):
    schema = _optimize_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(
        payload, schema
    )  # passes: no oneOf, no required -> the schema cannot say it
    assert "required" not in schema and "oneOf" not in json.dumps(schema)
    # ... which is exactly why the descriptions carry the rule and the function enforces it.
    with pytest.raises(ValueError, match="exactly one"):
        mcp_server.optimize_portfolio(**payload)
    # And over the MCP wire (validate_input=True, schema passes) the body's message surfaces.
    mcp_server.reset_registry()
    ok, err = _wire_call("optimize_portfolio", payload)
    assert ok is None and err is not None and "exactly one" in err


def test_descriptions_state_the_rule_from_all_three_places_and_name_each_other():
    entry = next(s for s in mcp_server.TOOL_SCHEMAS if s["name"] == "optimize_portfolio")
    props = entry["input_schema"]["properties"]
    for own, other in (("result_ids", "dataset_id"), ("dataset_id", "result_ids")):
        d = props[own]["description"]
        assert "exactly one of the two" in d and f"Mutually exclusive with {other}" in d
    assert "exactly one of the two selectors" in entry["description"]
    # objective was not touched.
    assert props["objective"]["description"] == "Optimization objective."
    assert props["objective"]["enum"] == sorted(mcp_server.OBJECTIVES)


# ---------------------------------------------------------------- _resolve_engine cycle claim


def _fresh_interpreter(code: str) -> str:
    env = {"PYTHONPATH": str(_ROOT / "src"), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=_ROOT, env=env, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def test_importing_guardrails_does_not_load_mcp_server_but_resolving_an_engine_does():
    out = _fresh_interpreter(
        "import sys\n"
        "from quantforge.ai import guardrails\n"
        "print('quantforge.ai.mcp_server' in sys.modules)\n"
        "print(type(guardrails._resolve_engine('python')).__name__)\n"
        "print('quantforge.ai.mcp_server' in sys.modules)\n"
        "try:\n    guardrails._resolve_engine('nope')\nexcept ValueError as e:\n    print('ValueError', 'nope' in str(e))\n"
    )
    assert out.splitlines() == ["False", "PythonEngine", "True", "ValueError True"]


def test_importing_mcp_server_loads_guardrails_at_top_level_so_the_reverse_import_would_cycle():
    out = _fresh_interpreter(
        "import sys\n"
        "import quantforge.ai.mcp_server as m\n"
        "print('quantforge.ai.guardrails' in sys.modules)\n"
        "print(m.guardrails.assert_no_codegen is sys.modules['quantforge.ai.guardrails'].assert_no_codegen)\n"
    )
    assert out.splitlines() == ["True", "True"]
    # The docstring's quoted import line is the literal top-level statement in mcp_server.py,
    # and guardrails.py has no top-level counterpart (only the function-local one).
    mcp_src = (_M5_ROOT_SRC / "mcp_server.py").read_text()
    top_level_imports = [
        n
        for n in ast.parse(mcp_src).body
        if isinstance(n, ast.ImportFrom) and n.module == "quantforge.ai"
    ]
    assert [a.name for n in top_level_imports for a in n.names] == ["guardrails"]
    g_tree = ast.parse((_M5_ROOT_SRC / "guardrails.py").read_text())
    assert not any(
        isinstance(n, (ast.Import, ast.ImportFrom)) and "mcp_server" in ast.dump(n)
        for n in g_tree.body
    )
    fn = next(
        n for n in g_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve_engine"
    )
    local = [n for n in fn.body if isinstance(n, ast.ImportFrom)]
    assert len(local) == 1 and local[0].module == "quantforge.ai.mcp_server"
    assert [a.name for a in local[0].names] == ["ENGINES"]


# ---------------------------------------------------------------- HoldoutHandle: honest scope


def test_holdout_handle_sentence_is_honest_introspection_reaches_the_frame_tools_do_not(
    monkeypatch,
):
    train, validation, handle = guardrails.split_data(_M5_PRICES)
    assert handle.n_days > 0
    # Deliberate in-process introspection of the closure cell DOES reach the holdout frame — the
    # docstring now says so instead of implying otherwise.
    cells = [c.cell_contents for c in handle._score.__closure__]
    frames = [c for c in cells if isinstance(c, pd.DataFrame)]
    assert len(frames) == 1
    holdout = frames[0]
    assert holdout.shape == (handle.n_days, len(_M5_TICKERS))
    assert holdout.index[0].date().isoformat() == handle.start
    assert holdout.index[-1].date().isoformat() == handle.end
    assert holdout.index[0] > validation.index[-1] > train.index[-1]  # disjoint, after validation
    # Every tool-mediated path the sentence lists is refused.
    text = repr(handle)
    assert handle.start in text and handle.end in text
    assert not any(f"{v:.4f}" in text for v in holdout.iloc[0].tolist())
    with pytest.raises(TypeError):
        vars(handle)
    with pytest.raises(TypeError):
        pickle.dumps(handle)
    assert not hasattr(handle, "__dict__") and not hasattr(handle, "__getitem__")
    # ... and the structural guarantee the sentence points at: MCP load_data refuses the dates.
    monkeypatch.setattr(mcp_server, "_price_source", lambda: _M5_PRICES)
    mcp_server.reset_registry()
    with pytest.raises(ValueError, match="holdout"):
        mcp_server.load_data(handle.start, handle.end, _M5_TICKERS)
    assert mcp_server._DATASETS == {}


def test_holdout_docstring_points_at_a_test_file_that_really_proves_load_data_rejection():
    doc = inspect.getdoc(guardrails.HoldoutHandle)
    named = re.search(r"tests/(test_holdout_isolation\.py)", doc).group(1)
    src = (_ROOT / "tests" / named).read_text()
    assert "mcp_server.load_data(" in src and "holdout" in src
    # The doc mirrors the same scoping sentence (accidental / tool-mediated vs introspection).
    md = " ".join((_ROOT / "docs" / "components" / "10-ai-guardrails.md").read_text().split())
    for phrase in ("accidental or tool-mediated", "introspection", "`load_data` rejection"):
        assert phrase in doc.replace("``", "`") or phrase in md
        assert phrase in md


# ================================================================ Milestone 6: ruff format

_M6_PATHS = (
    "src/quantforge/data/loader.py",
    "src/quantforge/metrics/performance.py",
    "tests/test_interchange_roundtrip.py",
    "tests/test_interchange_verifier.py",
    "tests/test_loader.py",
    "tests/test_loader_constants.py",
)

# Hand-typed from docs/components/02-data-loader.md: which tickers each section comment labels.
# The formatter put every ticker on its own line; a comment that slid one row up or down would
# still be "present and in order" yet mislabel a ticker — this pins the grouping itself.
_M6_UNIVERSE_GROUPS = [
    ("# Tech", ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "CRM", "ADBE", "ORCL", "AVGO"]),
    ("# ADRs (international, US-listed, USD)", ["TSM", "ASML", "SAP", "TM", "NVO", "SONY"]),
    ("# Financials", ["JPM", "GS", "V"]),
    ("# Healthcare", ["JNJ", "UNH", "PFE"]),
    ("# Consumer", ["PG", "KO", "MCD", "WMT", "HD"]),
    ("# Energy / Industrial", ["XOM", "CVX", "CAT"]),
]

# The layout the six files had at "week 8 complete" (git show f2e8cd0:src/quantforge/data/
# loader.py lines 31-46 and 116-117), reduced to a self-contained snippet so the test does not
# depend on HEAD still being that commit.
_M6_HEAD_ERA_LAYOUT = """\
UNIVERSE: list[str] = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "CRM", "ADBE", "ORCL", "AVGO",
    # Financials
    "JPM", "GS", "V",
]


def load_prices(tickers: list[str] | None = None, start: str = "2010-01-01", end: str = "x",
                cache_dir: str = "data_cache") -> None:
    return None
"""

_M6_TOKEN_NOISE = {
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
    tokenize.ENDMARKER,
}


def _m6_ruff() -> str | None:
    """The venv's ruff first (the version requirements.txt pins), then PATH; None if absent."""
    for name in ("ruff", "ruff.exe"):
        candidate = Path(sys.executable).parent / name
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def _m6_comments_and_tokens(source: str) -> tuple[list[str], list[str]]:
    toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    comments = [t.string for t in toks if t.type == tokenize.COMMENT]
    significant = [t.string for t in toks if t.type not in _M6_TOKEN_NOISE]
    return comments, significant


def _m6_ast_equal(a: str, b: str) -> bool:
    return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def _m6_git_py_files() -> set[str]:
    out = []
    for extra in ([], ["--others", "--exclude-standard"]):
        proc = subprocess.run(
            ["git", "ls-files", *extra, "--", "*.py"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        out.extend(proc.stdout.split())
    return set(out)


# ---------------------------------------------------------------- identical to HEAD, 3 levels


@pytest.mark.skipif(not _git_available(), reason="needs git + HEAD to diff against")
@pytest.mark.parametrize("rel", _M6_PATHS)
def test_reformatted_file_is_ast_comment_and_token_identical_to_head(rel):
    head, cur = _head_source(rel), (_ROOT / rel).read_text()
    assert head.strip(), f"{rel} is not tracked at HEAD"
    # Level 1: semantics (docstrings are AST constants, so a docstring edit would fail here).
    assert _m6_ast_equal(head, cur), f"AST of {rel} differs from HEAD"
    # Level 2: every comment verbatim and in order — AST equality cannot see these.
    head_comments, head_tokens = _m6_comments_and_tokens(head)
    cur_comments, cur_tokens = _m6_comments_and_tokens(cur)
    assert cur_comments == head_comments, f"a comment in {rel} changed"
    # Level 3: the significant token stream — string literals keep their exact quoting, no
    # number was rewritten — with commas the ONLY token the formatter may add (magic trailing
    # comma when it wraps a signature or call).
    assert [t for t in cur_tokens if t != ","] == [t for t in head_tokens if t != ","]
    assert cur_tokens.count(",") >= head_tokens.count(",")


@pytest.mark.skipif(not _git_available(), reason="needs git + HEAD to diff against")
def test_format_only_changes_versus_head_are_confined_to_the_six_files():
    # Adversarial scope check: any .py that differs from the baseline commit only in layout must
    # be one of the six (``--diff-filter=M``: files ADDED since the baseline have no baseline
    # source to compare against and are real changes by definition). Every other file this fix run touched carries a real (AST-visible) change; a seventh
    # file the formatter quietly rewrote would show up here as AST-identical and fail.
    proc = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=M", _baseline_ref() or "HEAD", "--", "*.py"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    changed = proc.stdout.split()
    layout_only = [
        rel for rel in changed if _m6_ast_equal(_head_source(rel), (_ROOT / rel).read_text())
    ]
    assert set(layout_only) <= set(_M6_PATHS), layout_only
    # And each of the six that still differs from HEAD differs (by construction) only in layout.
    for rel in _M6_PATHS:
        if rel in changed:
            assert rel in layout_only


# ---------------------------------------------------------------- ruff --check, repo-wide


def test_repo_wide_format_check_is_clean_and_counts_every_python_file_git_knows():
    ruff = _m6_ruff()
    if ruff is None:
        pytest.skip("ruff executable not found")
    proc = subprocess.run(
        [ruff, "format", "--check", "."], cwd=_ROOT, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would reformat" not in proc.stdout
    m = re.fullmatch(r"(\d+) files? already formatted", proc.stdout.strip())
    assert m, proc.stdout
    n_formatted = int(m.group(1))
    # "already formatted" must cover the whole tree, not a subset ruff happened to discover.
    assert n_formatted >= 85  # 79 clean at "week 8 complete" + the six from this milestone
    if _git_available():
        assert n_formatted == len(_m6_git_py_files())


def test_check_is_load_bearing_head_era_layout_fails_it_and_the_six_pass_the_same_settings(
    tmp_path,
):
    ruff = _m6_ruff()
    if ruff is None:
        pytest.skip("ruff executable not found")
    # The only formatter setting pyproject.toml carries is line-length = 100; --isolated plus
    # that flag is therefore the same judge, applied to a file outside the repo's config scope.
    settings = ["--isolated", "--line-length", "100"]
    old = tmp_path / "head_era_layout.py"
    old.write_text(_M6_HEAD_ERA_LAYOUT)
    proc = subprocess.run(
        [ruff, "format", "--check", *settings, str(old)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Would reformat" in proc.stdout
    # Same settings, the six real files: clean — so the repo config is not hiding a difference.
    proc = subprocess.run(
        [ruff, "format", "--check", *settings, *_M6_PATHS],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Formatting the snippet reproduces exactly what the milestone did: one ticker per line,
    # the section comment still immediately above its first ticker, the signature wrapped with
    # a trailing comma — and the AST untouched.
    subprocess.run([ruff, "format", *settings, str(old)], capture_output=True, check=True)
    new = old.read_text()
    assert _m6_ast_equal(_M6_HEAD_ERA_LAYOUT, new)
    assert '    # Tech\n    "AAPL",\n    "MSFT",\n' in new
    assert '    "AVGO",\n    # Financials\n    "JPM",\n' in new
    assert "def load_prices(\n    tickers: list[str] | None = None,\n" in new
    assert '    cache_dir: str = "data_cache",\n) -> None:' in new


# ---------------------------------------------------------------- UNIVERSE grouping + API


def test_universe_section_comments_still_label_exactly_their_tickers():
    src = (_ROOT / "src/quantforge/data/loader.py").read_text()
    start = src.index("UNIVERSE: list[str] = [") + len("UNIVERSE: list[str] = [")
    block = src[start : src.index("\n]\n", start)]
    groups: list[tuple[str, list[str]]] = []
    for line in block.splitlines():
        if not line.strip():
            continue
        if line.strip().startswith("#"):
            groups.append((line.strip(), []))
            continue
        m = re.fullmatch(r'    "([A-Z]+)",', line)  # exactly one ticker, 4-space indent, comma
        assert m, f"unexpected UNIVERSE line: {line!r}"
        assert groups, f"ticker {m.group(1)} appears before any section comment"
        groups[-1][1].append(m.group(1))
    assert groups == _M6_UNIVERSE_GROUPS
    flat = [t for _, tickers in groups for t in tickers]
    assert flat == loader.UNIVERSE and len(flat) == 30


def test_load_prices_signature_and_metric_constants_survived_the_wrap():
    sig = inspect.signature(loader.load_prices)
    assert list(sig.parameters) == ["tickers", "start", "end", "cache_dir"]
    assert [p.default for p in sig.parameters.values()] == [
        None,
        loader.START,
        loader.END,
        "data_cache",
    ]
    # Wrapping the signature must not have turned anything keyword-only.
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in sig.parameters.values())
    assert (loader.START, loader.END) == ("2010-01-01", "2026-06-30")
    assert performance.ANN == 252 and type(performance.ANN) is int
    assert performance.RISK_FREE == 0.0 and type(performance.RISK_FREE) is float
    # Hand computation, independent of the module constants (252 and 0 typed here on purpose).
    r = [0.01, -0.01, 0.02]
    mean = sum(r) / 3
    std = math.sqrt(sum((x - mean) ** 2 for x in r) / 3)  # population std (ddof=0)
    out = performance.compute_metrics(r)
    assert out["total_return"] == pytest.approx(1.01 * 0.99 * 1.02 - 1.0)
    assert out["cagr"] == pytest.approx((1.01 * 0.99 * 1.02) ** (252 / 3) - 1.0)
    assert out["ann_vol"] == pytest.approx(std * math.sqrt(252))
    assert out["sharpe"] == pytest.approx((mean - 0.0 / 252) / std * math.sqrt(252))
    assert out["max_drawdown"] == pytest.approx(-0.01)  # 1.01 -> 1.01*0.99: exactly -1%
    assert out["hit_rate"] == pytest.approx(2 / 3)


# ===========================================================================
# Milestone 7: closeout — handoff entry, 16-tests row, closeout-verifier pin advances
# ===========================================================================

_M7_HANDOFF_PATH = _ROOT / "handoff.md"
_M7_HANDOFF = _M7_HANDOFF_PATH.read_text()
_M7_TESTS_DOC = (_ROOT / "docs" / "components" / "16-tests.md").read_text()
_M7_NEW_HEADING = "## 2026-09-12 — Week 7/8 open-items fix run (build-verified workflow)\n"
_M7_HEAD_HEADING = "## 2026-09-11 — Week 8 complete"
_M7_PIN_PHRASE = "Pin advanced at the 2026-09-12 fix run"
_M7_VERIFIER_FILES = [f"tests/test_week{w}_closeout_verifier.py" for w in (3, 4, 5, 6, 7, 8)]
_M7_RENAME = (
    "test_handoff_newest_entry_is_week8_dated_2026_09_11_and_dates_descend",
    "test_handoff_week8_entry_is_second_below_the_2026_09_12_fix_entry_and_dates_descend",
)
# Every test function whose source may differ from HEAD in the closeout-verifier files, and
# which of them the milestone-7 spec listed by name (those must carry the dated phrase).
_M7_ALLOWED_CHANGES = {
    "tests/test_week6_closeout_verifier.py": {
        "test_handoff_week6_entry_preserved_below_week7_and_dates_descend": True,
    },
    "tests/test_week7_closeout_verifier.py": {
        _M7_RENAME[1]: True,
        "test_handoff_week8_entry_gate_counts_modules_suites_open_items_and_next_target": True,
        "test_handoff_week7_week6_and_week5_entries_preserved_below": True,
        # Advanced at milestone 1 (.env.example 5/25 -> 2/10), dated comment but not the phrase.
        "test_env_example_documents_dev_defaults_and_public_mode_fail_closed": False,
    },
    "tests/test_week8_closeout_verifier.py": {
        "test_handoff_first_entry_is_week8_2026_09_11_and_names_the_spec_items": True,
        "test_handoff_prepend_left_every_earlier_entry_byte_identical_to_head": True,
        "test_handoff_per_suite_growth_numbers_match_a_real_collection": True,
        "test_agent_reads_no_env_var_and_env_example_is_unchanged": False,
    },
}


def _m7_entries(text: str) -> list[tuple[str, str]]:
    """The week-8 verifier's split, repeated (tests/ is not a package)."""
    matches = list(re.finditer(r"^## (\d{4}-\d{2}-\d{2}) — ", text, flags=re.M))
    return [
        (m.group(1), text[m.start() : (matches[i + 1].start() if i + 1 < len(matches) else None)])
        for i, m in enumerate(matches)
    ]


def _m7_new_entry() -> str:
    entries = _m7_entries(_M7_HANDOFF)
    assert entries[0][0] == "2026-09-12"
    return entries[0][1]


def _m7_collected(*rel: str) -> int:
    """Real ``pytest --collect-only`` count of the given paths (or the whole suite when empty)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--collect-only",
            "--color=no",  # FORCE_COLOR in a caller's shell would ANSI-wrap the summary line
            "-q",
            *rel,
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    m = re.search(r"^(\d+) tests? collected", proc.stdout, flags=re.M)
    assert m, proc.stdout[-500:]
    return int(m.group(1))


def _m7_load_test_module(rel: str):
    """Import a tests/ file by path so its pins can be run against a tampered handoff."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(Path(rel).stem, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _m7_test_functions(src: str) -> dict[str, str]:
    tree = ast.parse(src)
    return {
        node.name: ast.get_source_segment(src, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }


# --- whole-file reconstruction -------------------------------------------------------------


@pytest.mark.skipif(not _git_available(), reason="needs git + HEAD to diff against")
def test_handoff_is_head_preamble_plus_one_new_entry_plus_head_body_byte_for_byte():
    """Stronger than the entry-split pins: the preamble above the first entry and the ``---``
    separators are part of the file too, and a prepend must leave them untouched."""
    head = _head_source("handoff.md")
    cut = head.index(_M7_HEAD_HEADING)
    preamble, body = head[:cut], head[cut:]
    assert _M7_HANDOFF.startswith(preamble), "the preamble above the first entry changed"
    assert _M7_HANDOFF.endswith(body), "HEAD's entries are not the byte-identical tail"
    inserted = _M7_HANDOFF[len(preamble) : len(_M7_HANDOFF) - len(body)]
    assert inserted.startswith(_M7_NEW_HEADING)
    assert inserted.endswith("\n---\n\n"), "the new entry must end with the same separator"
    # Exactly one entry was inserted — no second heading hides inside it.
    assert len(re.findall(r"^## \d{4}-\d{2}-\d{2} — ", inserted, flags=re.M)) == 1
    # And the whole file is nothing but those three pieces.
    assert preamble + inserted + body == _M7_HANDOFF
    # The baseline's own newest entry is the week-8 one (the commit the entry names as 74af8d9).
    assert _m7_entries(head)[0][0] == "2026-09-11"
    assert (_baseline_ref() or "").startswith("74af8d9")


# --- the closeout numbers are real ---------------------------------------------------------


def test_closeout_count_is_arithmetically_tied_to_a_real_full_collection():
    """``N passed / 2 skipped`` cannot be checked against a live run from inside pytest, but it
    can be checked against a live COLLECTION: at closeout the suite collected N + 2, with this
    file at the 119 tests the entry records. Whatever this file has grown by since (these very
    tests) is the only allowed difference."""
    body = _m7_new_entry()
    n = int(re.search(r"\*\*(\d+) passed / 2 skipped\*\*", body).group(1))
    builder_n = int(
        re.search(r"test_handoff_open_items\.py`\*?\*? \((\d+) collected", body).group(1)
    )
    verifier_n = int(
        re.search(r"test_handoff_open_items_verify\.py` \((\d+) collected", body).group(1)
    )
    assert (builder_n, verifier_n) == (69, 119)
    assert _m7_collected("tests/test_handoff_open_items.py") == builder_n
    this_file_now = _m7_collected("tests/test_handoff_open_items_verify.py")
    assert this_file_now >= verifier_n, "this file only ever grows past the recorded count"
    full_now = _m7_collected()
    assert full_now - (this_file_now - verifier_n) == n + 2, (
        f"handoff says {n} passed + 2 skipped, but the suite collects {full_now} "
        f"(this file grew by {this_file_now - verifier_n})"
    )
    assert n >= 1208


@pytest.mark.skipif(not _git_available(), reason="needs git + HEAD")
def test_entry_def_test_count_claim_matches_head_for_every_closeout_verifier():
    """The entry says no verifier test was deleted (per-file ``def test_`` counts equal HEAD's)."""
    for rel in _M7_VERIFIER_FILES:
        now = len(re.findall(r"^def test_", (_ROOT / rel).read_text(), flags=re.M))
        head = len(re.findall(r"^def test_", _head_source(rel), flags=re.M))
        assert now == head, f"{rel}: {now} tests now vs {head} at HEAD"


# --- adversarial: the pins actually bite --------------------------------------------------


def test_loose_count_regex_is_satisfied_by_the_baseline_phrase_but_the_bold_form_is_not():
    """Why the builder tightened the pin: the entry ALSO quotes the un-bold baseline
    ``1207 passed / 2 skipped / 1 failed``, which a loose regex would happily accept."""
    body = _m7_new_entry()
    loose = re.findall(r"(\d+) passed / 2 skipped", body)
    bold = re.findall(r"\*\*(\d+) passed / 2 skipped\*\*", body)
    assert "1207" in loose and len(loose) >= 2
    assert bold == [b for b in loose if b != "1207"] and len(bold) == 1


def test_builder_newest_entry_pin_fails_on_a_placeholder_count(monkeypatch):
    """Run the builder's own pin against the entry with the count replaced by ``NNNN`` — the
    shape the builder used before substituting the real number — and it must fail."""
    builder = _m7_load_test_module("tests/test_handoff_open_items.py")
    tampered = re.sub(r"\*\*\d+ passed / 2 skipped\*\*", "**NNNN passed / 2 skipped**", _M7_HANDOFF)
    assert tampered != _M7_HANDOFF
    monkeypatch.setattr(builder, "_HANDOFF", tampered)
    with pytest.raises(AssertionError):
        builder.test_newest_handoff_entry_is_the_2026_09_12_fix_run_and_says_what_matters()
    # A number below the baseline is refused too (a stale count pasted from an older entry).
    stale = re.sub(r"\*\*\d+ passed / 2 skipped\*\*", "**1186 passed / 2 skipped**", _M7_HANDOFF)
    monkeypatch.setattr(builder, "_HANDOFF", stale)
    with pytest.raises(AssertionError):
        builder.test_newest_handoff_entry_is_the_2026_09_12_fix_run_and_says_what_matters()
    # Untampered: passes (so the failures above are the tampering, not a broken import).
    monkeypatch.setattr(builder, "_HANDOFF", _M7_HANDOFF)
    builder.test_newest_handoff_entry_is_the_2026_09_12_fix_run_and_says_what_matters()


@pytest.mark.skipif(not _git_available(), reason="needs git + HEAD")
@pytest.mark.parametrize("victim_date", ["2026-09-11", "2026-09-10", "oldest"])
def test_byte_identity_pins_catch_a_single_changed_byte_in_an_earlier_entry(
    monkeypatch, victim_date
):
    """One character flipped inside the Week-8, Week-7 or oldest entry must be caught by BOTH the
    week-8 verifier's advanced pin and the builder's self-check."""
    entries = _m7_entries(_M7_HANDOFF)
    if victim_date == "oldest":
        victim_date = entries[-1][0]
    victim = next(b for d, b in entries if d == victim_date)
    # Flip the first digit after the heading line — a change no re-flow would produce.
    heading_end = victim.index("\n") + 1
    pos = heading_end + re.search(r"\d", victim[heading_end:]).start()
    flipped = victim[:pos] + ("0" if victim[pos] != "0" else "1") + victim[pos + 1 :]
    tampered = _M7_HANDOFF.replace(victim, flipped, 1)
    assert tampered != _M7_HANDOFF and _m7_entries(tampered)[0][0] == "2026-09-12"

    week8 = _m7_load_test_module("tests/test_week8_closeout_verifier.py")
    monkeypatch.setattr(week8, "_HANDOFF", tampered)
    with pytest.raises(AssertionError, match="earlier handoff entry was edited"):
        week8.test_handoff_prepend_left_every_earlier_entry_byte_identical_to_head()

    builder = _m7_load_test_module("tests/test_handoff_open_items.py")
    monkeypatch.setattr(builder, "_HANDOFF", tampered)
    with pytest.raises(AssertionError, match="earlier handoff entry was edited"):
        builder.test_every_earlier_handoff_entry_is_byte_identical_to_head()

    # And the untampered file passes both, so the raises above are the flip.
    monkeypatch.setattr(week8, "_HANDOFF", _M7_HANDOFF)
    week8.test_handoff_prepend_left_every_earlier_entry_byte_identical_to_head()
    monkeypatch.setattr(builder, "_HANDOFF", _M7_HANDOFF)
    builder.test_every_earlier_handoff_entry_is_byte_identical_to_head()


def test_advanced_week7_pin_rejects_a_handoff_where_the_fix_entry_is_missing(monkeypatch):
    """The renamed week-7 pin must fail against HEAD's handoff (no fix entry on top) — otherwise
    the rename would be cosmetic and the pin would not actually depend on the new entry."""
    if not _git_available():
        pytest.skip("needs git + HEAD")
    week7 = _m7_load_test_module("tests/test_week7_closeout_verifier.py")
    head = _head_source("handoff.md")
    monkeypatch.setattr(week7, "_HANDOFF", head)
    with pytest.raises(AssertionError):
        getattr(week7, _M7_RENAME[1])()
    monkeypatch.setattr(week7, "_HANDOFF", _M7_HANDOFF)
    getattr(week7, _M7_RENAME[1])()


# --- renames, phrases, allowlist -----------------------------------------------------------


@pytest.mark.skipif(not _git_available(), reason="needs git + HEAD")
def test_only_the_documented_rename_differs_from_head_across_the_closeout_verifiers():
    old_name, new_name = _M7_RENAME
    for rel in _M7_VERIFIER_FILES:
        head_names = set(_m7_test_functions(_head_source(rel)))
        now_names = set(_m7_test_functions((_ROOT / rel).read_text()))
        if rel.endswith("test_week7_closeout_verifier.py"):
            assert head_names - now_names == {old_name}
            assert now_names - head_names == {new_name}
        else:
            assert head_names == now_names, rel
    # The old name is gone from the whole tests/ tree, and the entry records the new one.
    for path in (_ROOT / "tests").glob("test_*.py"):
        if path.name != "test_handoff_open_items_verify.py":
            assert f"def {old_name}(" not in path.read_text(), path.name
    assert new_name in _m7_new_entry()


@pytest.mark.skipif(not _git_available(), reason="needs git + HEAD")
def test_changed_verifier_tests_are_allowlisted_and_spec_listed_ones_carry_the_dated_phrase():
    for rel, allowed in _M7_ALLOWED_CHANGES.items():
        head_fns = _m7_test_functions(_head_source(rel))
        now_fns = _m7_test_functions((_ROOT / rel).read_text())
        changed = {n for n, s in now_fns.items() if head_fns.get(n) != s}
        assert changed == set(allowed), f"{rel}: changed tests {sorted(changed)}"
        for name, must_carry_phrase in allowed.items():
            flat = " ".join(now_fns[name].replace("#", " ").split())
            if must_carry_phrase:
                assert _M7_PIN_PHRASE in flat, f"{rel}::{name} lacks the dated pin comment"
            assert "2026-09-12" in flat, f"{rel}::{name} carries no date for the advance"
    # Every advanced pin keeps its content assertions: the needles the week-8 verifier checks
    # are still present in the function bodies (not deleted along the way).
    week8_src = (_ROOT / "tests/test_week8_closeout_verifier.py").read_text()
    for needle in (
        '"1186 passed / 2 skipped"',
        '"919 passed / 1 skipped"',
        '"879 passed / 1 skipped"',
        '"488 passed / 2 skipped"',
    ):
        assert needle in week8_src
    week7_src = (_ROOT / "tests/test_week7_closeout_verifier.py").read_text()
    assert "uncommitted" in week7_src, "historical needle dropped from the week-7 verifier"


# --- 16-tests.md: exactly one inserted row -------------------------------------------------


@pytest.mark.skipif(not _git_available(), reason="needs git + HEAD")
def test_tests_doc_differs_from_head_by_exactly_one_row_below_the_week8_verifier_row():
    head_lines = _head_source("docs/components/16-tests.md").splitlines()
    now_lines = _M7_TESTS_DOC.splitlines()
    ops = [
        (tag, i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, head_lines, now_lines, autojunk=False
        ).get_opcodes()
        if tag != "equal"
    ]
    assert len(ops) == 1 and ops[0][0] == "insert", ops
    _, i1, i2, j1, j2 = ops[0]
    assert j2 - j1 == 1
    row = now_lines[j1]
    assert row.startswith("| `test_handoff_open_items.py`")
    above = now_lines[j1 - 1]
    assert (
        above.startswith("| `test_agent_verify.py`") and "test_week8_closeout_verifier.py" in above
    )
    # Same column count as the row above (a well-formed table row), and the row names the
    # milestone's proving files and the requirement ids the spec fixed.
    assert row.count(" | ") == above.count(" | ")
    assert "SF-1/SF-3/AR-6" in row and "✅ green (2026-09-12)" in row
    assert "test_handoff_open_items_verify.py" in row


# --- the entry's forward-looking claims ----------------------------------------------------


def test_entry_names_week9_surfaces_quotes_the_live_smoke_command_and_claims_no_week9_work():
    body = _m7_new_entry()
    flat = " ".join(body.split())  # the markdown wraps mid-phrase
    for needle in (
        "**Next up (per docs/TEN_WEEK_PLAN.md): Week 9 — finish UI + deploy + harden**",
        "Research tab / Research mode",
        "`nl_interface.handle`",
        "`DEMO_PASSCODE`",
        "`QUANTFORGE_LIVE_AI=1 pytest tests/test_nl_interface.py tests/test_agent.py -k live_smoke`",
        "no conflict",
        "Bedrock",
        "### Agent failures and resolutions",
        "### Open items (carried forward)",
    ):
        assert needle in flat, needle
    assert "Week 9 complete" not in flat and "Week 9 —" in flat
    # "The app shell still imports no quantforge.ai.*": check the real import list, not prose.
    tree = ast.parse((_ROOT / "app" / "streamlit_app.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported |= {node.module} | {f"{node.module}.{a.name}" for a in node.names}
    assert not any(n == "quantforge.ai" or n.startswith("quantforge.ai.") for n in imported)
