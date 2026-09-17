"""Verifier tests pinning the Week-7 closeout state (docs/config milestone; offline, no network).

Pins advanced at the Week-8 closeout, per precedent: the plan count, the handoff position, the
16-tests status text and the guardrails-doc agent-loop wording now assert the week-8 state, and
week-8 rows/entries are pinned alongside; every week-7 assertion is retargeted, not deleted.

Independently proves the closeout's deliverables so none can silently regress:

- docs/TEN_WEEK_PLAN.md: all three Week-7 boxes ticked and named; exactly 24 ticked overall
  (4+3+3+2+3+3+3+3, Weeks 1-8); Weeks 9-10 + Stretch untouched.
- handoff.md: the newest entry is the 2026-09-12 open-items fix-run one; directly below it is
  the 2026-09-11 Week-8 entry (names ai/agent.py, run_research,
  the three proof suites, its exact gate counts, the 919/1 week-7 baseline, the uncommitted /
  approval note, and Week 9 as next); the 2026-09-10 Week-7 entry sits intact below it with its
  exact gate counts (879 passed / 1 skipped), the 488/2 baseline, ruff clean, "Agent failures",
  every open item the spec demanded, and Week 8 as next; the Week-6 (488/2) and Week-5 (428/2)
  entries survive below; dates stay newest-first.
- .env.example <-> docs/components/18-runtime-config.md: the variable inventories are *equal as
  sets* (not just table ⊆ file), every variable carries a comment, and every env key the
  `src/quantforge/ai/` modules actually read is in the table. `QUANTFORGE_LIVE_AI` is a
  test-only flag: absent from both inventories, documented in tests/test_nl_interface.py.
- Docs may not overclaim what the code does (the docs-milestone analog of a prescient signal):
  the defaults the table and `.env.example` advertise (2/10 USD, rate limit 20, ledger path)
  are the module constants; PUBLIC_MODE with unset caps really denies `allow(0.0)`; the
  rate-limit state file really sits beside `AI_LEDGER_PATH`; `load_data` really *rejects*
  (never truncates) a window touching the holdout and registers nothing; `score_holdout`
  really lazy-imports the engine registry; `_call` really meters cache tokens.
- docs/components/09/10/11/12/18: status lines off "stub", each with a "Decisions made in
  build" list naming the four spec'd decisions; 10 records the agent-loop clause as landed in
  week 8 (no longer pending); 13 is built/green (wk 8) with the built `run_research`
  signature, which must match `inspect.signature` (a doc may not overclaim a kwarg).
- docs/components/16-tests.md: the seven week-7 rows green, the verifier row names its
  suites, a `test_agent.py` row and a week-8 verifier row exist, the status line advanced to
  "wk 1-8" naming both opt-in live smokes, and every test file the inventory names exists (a
  phantom row fails).
- README: the "AI layer safety" paragraph in "Methodology and known limitations" points at
  budget/guardrails/nl_interface, the parameter-only rule, and the proving tests — and every
  link target in that section resolves to a real file (a dead link fails).
- Git hygiene: `.env` and the AI state files are ignored, so a closeout commit cannot leak a key
  or a spend ledger.

Adversarial cases: a stray "[x]" anywhere later in the plan fails the exact-count pin; a
variable added to `.env.example` but not to the table (or vice versa) fails the set-equality
check; a `.env.example` default that drifts from the module constant fails; a holdout window
that `load_data` starts clamping instead of rejecting fails; a 16-tests.md row naming a missing
file fails; a README link whose file moved fails; the handoff Week-7 entry demoted or its
counts edited fails.
"""

import inspect
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantforge import interchange
from quantforge.ai import agent, budget, guardrails, mcp_server, nl_interface
from quantforge.data import loader

_ROOT = Path(__file__).resolve().parents[1]
_PLAN = (_ROOT / "docs" / "TEN_WEEK_PLAN.md").read_text()
_HANDOFF = (_ROOT / "handoff.md").read_text()
_ENV_EXAMPLE = (_ROOT / ".env.example").read_text()
_README = (_ROOT / "README.md").read_text()
_DOC_TESTS = (_ROOT / "docs" / "components" / "16-tests.md").read_text()
_DOC_RUNTIME = (_ROOT / "docs" / "components" / "18-runtime-config.md").read_text()
_COMPONENT_DOCS = {
    name: (_ROOT / "docs" / "components" / f"{name}.md").read_text()
    for name in ("09-ai-budget", "10-ai-guardrails", "11-mcp-server", "12-nl-interface")
}
_DOC_AGENT = (_ROOT / "docs" / "components" / "13-ai-agent.md").read_text()

# The spec's table inventory, spelled out so the test does not trust the doc it is checking.
_SPEC_TABLE_VARS = {
    "ANTHROPIC_API_KEY",
    "AI_BUDGET_USD_DAILY",
    "AI_BUDGET_USD_TOTAL",
    "AI_DISABLED",
    "AI_RATE_LIMIT_PER_HOUR",
    "AI_LEDGER_PATH",
    "PUBLIC_MODE",
    "DEMO_PASSCODE",
}


def _plan_section(week: str) -> str:
    """Body of one '## Week N — ...' (or '### Stretch') section of the plan."""
    parts = re.split(r"^#{2,3} ", _PLAN, flags=re.M)
    for part in parts:
        if part.startswith(week):
            return part
    raise AssertionError(f"plan section {week!r} not found")


def _status_line(doc: str) -> str:
    return next(line for line in doc.splitlines() if line.startswith("**Week"))


def _doc_section(doc: str, heading: str) -> str:
    """Text from ``heading`` up to the next H2."""
    assert heading in doc, f"missing section {heading!r}"
    section = doc[doc.index(heading) + len(heading) :]
    return section.split("\n## ")[0]


def _handoff_entries() -> list[tuple[str, str]]:
    """(date, body) for each '## YYYY-MM-DD — ...' entry, in file order."""
    matches = list(re.finditer(r"^## (\d{4}-\d{2}-\d{2}) — ", _HANDOFF, flags=re.M))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(_HANDOFF)
        out.append((m.group(1), _HANDOFF[m.start() : end]))
    return out


def _table_vars() -> set[str]:
    """Variables named in the 18-runtime-config inventory table (rows start with '| `VAR` |')."""
    return set(re.findall(r"^\| `([A-Z_]+)` \|", _DOC_RUNTIME, flags=re.M))


def _env_example_vars() -> dict[str, str]:
    """{VAR: value} for every 'VAR=value' assignment line in .env.example."""
    return dict(re.findall(r"^([A-Z_]+)=(.*)$", _ENV_EXAMPLE, flags=re.M))


# ---------------------------------------------------------------------------
# docs/TEN_WEEK_PLAN.md
# ---------------------------------------------------------------------------


def test_plan_week7_all_three_items_ticked_and_named():
    body = _plan_section("Week 7")
    assert body.count("- [x]") == 3 and "- [ ]" not in body
    assert "ai/mcp_server.py" in body
    assert "ai/nl_interface.py" in body
    assert "ai/budget.py" in body and "ai/guardrails.py" in body


def test_plan_week8_all_three_items_ticked_and_named():
    body = _plan_section("Week 8")
    assert body.count("- [x]") == 3 and "- [ ]" not in body
    assert "ai/agent.py" in body
    assert "untouched holdout" in body
    assert "tests/test_holdout_isolation.py" in body


def test_plan_exactly_24_ticked_and_weeks_9_plus_untouched():
    # Adversarial exact count: 4+3+3+2+3+3+3+3 for Weeks 1-8 (pin advanced at the week-8
    # closeout) — a stray tick anywhere later in the file (or an untick earlier) fails here.
    # Mirrors the spec's `grep -c` check.
    assert _PLAN.count("- [x]") == 24
    assert len(re.findall(r"^- \[x\]", _PLAN, flags=re.M)) == 24
    for week, n in {
        "Week 1": 4,
        "Week 2": 3,
        "Week 3": 3,
        "Week 4": 2,
        "Week 5": 3,
        "Week 6": 3,
        "Week 7": 3,
        "Week 8": 3,
    }.items():
        body = _plan_section(week)
        assert body.count("- [x]") == n and "- [ ]" not in body, week
    for later in ("Week 9", "Week 10", "Stretch"):
        body = _plan_section(later)
        assert "- [x]" not in body, f"{later} has a prematurely ticked box"
        assert "- [ ]" in body, f"{later} lost its checklist"


# ---------------------------------------------------------------------------
# handoff.md — the Week-8 closeout entry on top, the Week-7 entry intact below it
# ---------------------------------------------------------------------------


def _week7_entry() -> str:
    """The 2026-09-10 Week-7 entry, looked up by date + content (not position) so later
    closeouts can prepend on top without rewriting history."""
    return next(b for d, b in _handoff_entries() if d == "2026-09-10" and "Week 7" in b)


def test_handoff_week8_entry_is_second_below_the_2026_09_12_fix_entry_and_dates_descend():
    # Pin advanced at the 2026-09-12 fix run: the newest entry is that run's open-items entry;
    # the Week-8 entry sits directly below it and the Week-7 entry directly below that, each
    # still starting with its original heading.
    entries = _handoff_entries()
    assert entries, "handoff.md has no dated entries"
    date, body = entries[0]
    assert date == "2026-09-12"
    assert "open-items" in body.splitlines()[0]
    assert entries[1][0] == "2026-09-11"
    assert "Week 8" in entries[1][1]
    assert entries[1][1].lstrip().startswith("## 2026-09-11 — Week 8 complete")
    assert entries[2][0] == "2026-09-10"
    # A stale "Week 8 next" line from the week-7 entry must not be mistaken for the new entry.
    assert entries[2][1].lstrip().startswith("## 2026-09-10 — Week 7 complete")
    dates = [d for d, _ in entries]
    assert dates == sorted(dates, reverse=True), "handoff entries are not newest-first"


def test_handoff_week8_entry_gate_counts_modules_suites_open_items_and_next_target():
    # Pin advanced at the 2026-09-12 fix run: the Week-8 entry is read by date, not as [0].
    # Every needle below is historical text of THAT entry ("uncommitted" included — Week 8 was
    # committed afterwards as 74af8d9, which the 2026-09-12 entry records).
    body = next(b for d, b in _handoff_entries() if d == "2026-09-11" and "Week 8" in b)
    # Exact counts the closeout run produced (the spec: recorded verbatim), and the week-7
    # baseline it started from.
    assert "1186 passed / 2 skipped" in body
    assert "919 passed / 1 skipped" in body
    assert "ruff" in body.lower() and "clean" in body.lower()
    # Both opt-in live smokes are named as the only skips.
    assert "test_nl_interface.py::test_live_smoke" in body
    assert "test_agent.py::test_live_smoke" in body
    assert "QUANTFORGE_LIVE_AI" in body
    # What was built, and the three proof suites.
    assert "ai/agent.py" in body and "run_research" in body
    assert "propose_experiment" in body and "declare_done" in body
    for suite in ("test_agent.py", "test_holdout_isolation.py", "test_public_mode_no_codegen.py"):
        assert suite in body, f"week-8 handoff entry does not name {suite}"
    assert "Agent failures" in body
    # Open items the spec demanded.
    assert "uncommitted" in body.lower()
    assert "pending" in body.lower() and "approv" in body.lower()
    assert "ANTHROPIC_API_KEY" in body and "placeholder" in body
    assert re.search(r"Research (tab|mode)", body) and re.search(r"[Ww]eek 9", body)
    # Next target.
    assert "Next up" in body and "Week 9" in body


def test_handoff_week7_entry_gate_counts_and_baseline():
    body = _week7_entry()
    # The spec's literal string, plus the exact number the closeout run produced.
    assert "passed / 1 skipped" in body
    assert "879 passed / 1 skipped" in body
    assert "488 passed / 2 skipped" in body  # the pre-run baseline, stated in the entry
    assert "ruff" in body.lower() and "clean" in body.lower()
    # The single skip is named, and it is the opt-in live smoke — not a silent R skip.
    assert "test_live_smoke" in body and "QUANTFORGE_LIVE_AI" in body


def test_handoff_week7_entry_names_modules_open_items_and_next_target():
    body = _week7_entry()
    for module in ("ai/budget.py", "ai/guardrails.py", "ai/mcp_server.py", "ai/nl_interface.py"):
        assert module in body, f"handoff entry does not name {module}"
    for suite in (
        "test_budget.py",
        "test_guardrails.py",
        "test_mcp_tools.py",
        "test_mcp_server.py",
        "test_nl_interface.py",
        "test_holdout_isolation.py",
        "test_public_mode_no_codegen.py",
    ):
        assert suite in body, f"handoff entry does not name {suite}"
    assert "Agent failures" in body
    # Open items the spec demanded, each stated explicitly.
    assert "pending" in body.lower() and "approv" in body.lower()
    assert "Week 6" in body and "Week 7" in body  # both commits pending
    assert "ANTHROPIC_API_KEY" in body and "placeholder" in body
    assert "cach" in body.lower() and "not a defect" in body
    assert re.search(r"[Ww]eek 9", body) and "nl_interface.handle" in body
    # Next target.
    assert "Week 8" in body
    assert "ai/agent.py" in body and "run_research" in body and "score_holdout" in body


def test_handoff_week7_week6_and_week5_entries_preserved_below():
    # Prepending must not rewrite history: the week-7, week-6 and week-5 records keep their
    # counts, in order, directly below the week-8 entry. Pin advanced at the 2026-09-12 fix
    # run: positions are relative to the week-8 entry's index, not absolute.
    entries = _handoff_entries()
    idx8 = next(i for i, (d, b) in enumerate(entries) if d == "2026-09-11" and "Week 8" in b)
    idx7, idx6 = idx8 + 1, idx8 + 2
    assert entries[idx7][0] == "2026-09-10" and "Week 7" in entries[idx7][1]
    assert "919 passed / 1 skipped" in entries[idx7][1]
    assert entries[idx6][0] == "2026-09-01" and "Week 6" in entries[idx6][1]
    assert "488 passed / 2 skipped" in entries[idx6][1]
    idx5 = next(i for i, (d, b) in enumerate(entries) if d == "2026-08-31" and "Week 5" in b)
    assert idx5 > idx6
    assert "428 passed / 2 skipped" in entries[idx5][1]


# ---------------------------------------------------------------------------
# .env.example <-> docs/components/18-runtime-config.md inventory
# ---------------------------------------------------------------------------


def test_runtime_config_table_matches_the_spec_inventory():
    assert _table_vars() == _SPEC_TABLE_VARS


def test_env_example_inventory_equals_table_inventory():
    # Set equality, both directions: a variable in the table but not in the file breaks
    # onboarding; a variable in the file but not in the table is an undocumented knob.
    assert set(_env_example_vars()) == _table_vars()
    for var in _table_vars():
        assert re.search(rf"^{var}=", _ENV_EXAMPLE, flags=re.M), f"{var} missing from .env.example"


def test_env_example_every_variable_has_a_comment_above_it():
    lines = _ENV_EXAMPLE.splitlines()
    for i, line in enumerate(lines):
        if not re.match(r"^[A-Z_]+=", line):
            continue
        # Walk up past other assignments in the same block; the nearest other line must be a
        # comment (blank lines end the block).
        j = i - 1
        while j >= 0 and re.match(r"^[A-Z_]+=", lines[j]):
            j -= 1
        assert j >= 0 and lines[j].startswith("#"), f"{line.split('=')[0]} has no comment above it"


def test_env_example_documents_dev_defaults_and_public_mode_fail_closed():
    assert re.search(r"2 \(daily\) / 10 \(total\)", _ENV_EXAMPLE)
    assert "PUBLIC_MODE=on" in _ENV_EXAMPLE and "denies EVERY call" in _ENV_EXAMPLE
    assert "AI_LEDGER_PATH" in _ENV_EXAMPLE and "ai_rate_limits.json" in _ENV_EXAMPLE
    # The example values now EQUAL the in-code dev defaults (aligned 2026-09-12 — the file used to
    # say 5 / 25 while the comment block, the docs table and budget.py all said 2 / 10).
    values = _env_example_vars()
    assert values["AI_BUDGET_USD_DAILY"] == str(int(budget._DEV_DEFAULT_DAILY_USD)) == "2"
    assert values["AI_BUDGET_USD_TOTAL"] == str(int(budget._DEV_DEFAULT_TOTAL_USD)) == "10"
    assert values["PUBLIC_MODE"] == "off" and values["AI_DISABLED"] == "off"
    assert values["DEMO_PASSCODE"] == ""  # blank placeholder, never a real passcode


def test_quantforge_live_ai_is_test_only_not_a_runtime_setting():
    assert "QUANTFORGE_LIVE_AI" not in _ENV_EXAMPLE
    assert "QUANTFORGE_LIVE_AI" not in _table_vars()
    suite = (_ROOT / "tests" / "test_nl_interface.py").read_text()
    comment = next(
        line for line in suite.splitlines() if line.startswith("#") and "QUANTFORGE_LIVE_AI" in line
    )
    assert "test-only" in comment.lower()
    assert ".env.example" in suite
    # The doc explains the omission too, so a reader of the table is not left guessing.
    assert "QUANTFORGE_LIVE_AI" in _DOC_RUNTIME and "not" in _DOC_RUNTIME


def test_every_env_key_read_under_ai_package_is_in_the_table():
    # Grep-based inventory over the real source: os.getenv / os.environ[...] /
    # os.environ.get(...) / _cap_from_env("...") — anything the AI layer reads must be documented.
    pattern = re.compile(
        r"(?:os\.getenv|os\.environ\.get|_cap_from_env)\(\s*\"([A-Z_]+)\"|os\.environ\[\s*\"([A-Z_]+)\"\]"
    )
    seen: set[str] = set()
    for path in sorted((_ROOT / "src" / "quantforge" / "ai").glob("*.py")):
        for m in pattern.finditer(path.read_text()):
            seen.add(m.group(1) or m.group(2))
    assert seen, "no env reads found under src/quantforge/ai/ — the grep pattern is broken"
    assert seen <= _table_vars(), f"undocumented env keys read by ai/: {seen - _table_vars()}"
    # Every AI_*/PUBLIC_MODE row in the table is actually consumed by the AI package (no phantom
    # rows); the API key is read by the SDK and the passcode by the app, so they are exempt.
    expected_in_code = _table_vars() - {"ANTHROPIC_API_KEY", "DEMO_PASSCODE"}
    assert expected_in_code <= seen, f"table rows no ai/ module reads: {expected_in_code - seen}"


def test_runtime_config_doc_status_complete_for_week7():
    status = _status_line(_DOC_RUNTIME)
    assert "stub" not in status.lower()
    assert ".env.example" in status and "complete for week 7" in status


# ---------------------------------------------------------------------------
# Docs may not overclaim: the advertised defaults and behaviours must be the code's
# ---------------------------------------------------------------------------


def test_advertised_defaults_are_the_module_constants():
    # Table + .env.example say 2/10 USD dev caps, rate limit 20, ledger at
    # data_cache/ai_ledger.json. If a constant moves, the docs become a lie and this fails.
    assert budget._DEV_DEFAULT_DAILY_USD == 2.0
    assert budget._DEV_DEFAULT_TOTAL_USD == 10.0
    assert budget._DEFAULT_LEDGER_PATH == "data_cache/ai_ledger.json"
    assert guardrails._DEFAULT_RATE_LIMIT_PER_HOUR == 20
    assert _env_example_vars()["AI_LEDGER_PATH"] == budget._DEFAULT_LEDGER_PATH
    assert _env_example_vars()["AI_RATE_LIMIT_PER_HOUR"] == "20"
    assert re.search(r"`AI_BUDGET_USD_DAILY`.*`2` \(dev\)", _DOC_RUNTIME)
    assert re.search(r"`AI_BUDGET_USD_TOTAL`.*`10` \(dev\)", _DOC_RUNTIME)
    assert re.search(r"`AI_RATE_LIMIT_PER_HOUR`.*`20`", _DOC_RUNTIME)
    assert re.search(r"`AI_LEDGER_PATH`.*`data_cache/ai_ledger\.json`", _DOC_RUNTIME)


def test_public_mode_with_unset_caps_denies_and_dev_mode_uses_2_and_10(monkeypatch, tmp_path):
    # Behavioural check of the exact sentence .env.example and the table make.
    monkeypatch.setenv("AI_LEDGER_PATH", str(tmp_path / "ledger.json"))
    for var in ("AI_BUDGET_USD_DAILY", "AI_BUDGET_USD_TOTAL", "AI_DISABLED"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PUBLIC_MODE", "on")
    assert budget.allow(0.0) is False  # even a free call is denied: no cap == no budget
    assert budget.remaining() == {"daily": 0.0, "total": 0.0}
    monkeypatch.setenv("PUBLIC_MODE", "off")
    assert budget.remaining() == {"daily": 2.0, "total": 10.0}
    assert budget.allow(2.0) is True and budget.allow(2.0 + 1e-6) is False


def test_rate_limit_state_file_is_derived_from_ai_ledger_path(monkeypatch, tmp_path):
    ledger = tmp_path / "nested" / "spend" / "ledger.json"
    monkeypatch.setenv("AI_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "3")
    assert guardrails._rate_state_path() == ledger.with_name("ai_rate_limits.json")
    assert [guardrails.rate_limit("wk7-key") for _ in range(4)] == [True, True, True, False]
    assert ledger.with_name("ai_rate_limits.json").is_file()
    # Only the state file and its flock sidecar landed in the (freshly created) ledger
    # directory — no stray temp files from the atomic writer, and no ledger written by mistake.
    assert sorted(p.name for p in ledger.parent.iterdir()) == [
        "ai_rate_limits.json",
        "ai_rate_limits.json.lock",
    ]


def test_score_holdout_lazy_imports_the_engine_registry():
    # The decision list says the import is function-local to avoid the mcp_server -> guardrails
    # cycle. Prove it from both sides: the source of the resolver contains the import, and the
    # guardrails module namespace holds neither `mcp_server` nor `ENGINES`.
    src = inspect.getsource(guardrails._resolve_engine)
    assert "from quantforge.ai.mcp_server import ENGINES" in src
    assert "ENGINES" not in vars(guardrails) and "mcp_server" not in vars(guardrails)
    module_src = (_ROOT / "src" / "quantforge" / "ai" / "guardrails.py").read_text()
    assert not re.search(r"^(from|import) quantforge\.ai\.mcp_server", module_src, flags=re.M)
    assert "score_holdout" in inspect.getsource(guardrails)
    assert "_resolve_engine(" in inspect.getsource(guardrails.score_holdout)


def _synthetic_long_through_holdout() -> pd.DataFrame:
    """A price panel that deliberately *has* holdout-period rows, so truncation would succeed."""
    rng = np.random.default_rng(7)
    tickers = ["AAPL", "MSFT"]
    dates = pd.bdate_range("2010-01-01", loader.SPLITS["holdout"][1], tz="UTC")
    rets = rng.normal(0.0003, 0.01, size=(len(dates), len(tickers)))
    wide = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)
    return interchange.to_long(wide, "prices")


@pytest.mark.parametrize(
    "start, end",
    [
        ("2022-06-01", "2023-06-30"),  # straddles validation/holdout boundary
        ("2022-01-01", "2023-01-01"),  # end exactly on the first holdout day
        ("2023-01-01", "2024-01-01"),  # fully inside the holdout
        ("2009-06-01", "2015-01-01"),  # before train.start — same rule, no clamp there either
    ],
)
def test_load_data_rejects_never_truncates_windows_outside_train_plus_validation(
    monkeypatch, start, end
):
    monkeypatch.setattr(mcp_server, "_price_source", _synthetic_long_through_holdout)
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    mcp_server.reset_registry()
    try:
        with pytest.raises(ValueError, match="holdout"):
            mcp_server.load_data(start, end, ["AAPL", "MSFT"])
        assert mcp_server._DATASETS == {}, "a rejected window must register nothing"
        # The error names both allowed bounds, as component 10's decision list promises.
        with pytest.raises(ValueError) as excinfo:
            mcp_server.load_data(start, end, ["AAPL", "MSFT"])
        lo, hi = loader.SPLITS["train"][0], loader.SPLITS["validation"][1]
        assert lo in str(excinfo.value) and hi in str(excinfo.value)
        # And a window inside the allowed range still works, so the rejection is not "everything".
        ok = mcp_server.load_data("2015-01-01", "2016-12-31", ["AAPL", "MSFT"])
        assert ok["dataset_id"] in mcp_server._DATASETS
        assert mcp_server._DATASETS[ok["dataset_id"]].index.max() <= pd.Timestamp(hi, tz="UTC")
    finally:
        mcp_server.reset_registry()


def test_nl_interface_call_meters_cache_tokens_conservatively():
    # Component 09/12 decision: uncached + cache-write + cache-read tokens all billed as input.
    src = inspect.getsource(nl_interface._call)
    assert "cache_creation_input_tokens" in src and "cache_read_input_tokens" in src
    assert "budget.charge(" in src
    # `_call` is the only `messages.create` site.
    module_src = (_ROOT / "src" / "quantforge" / "ai" / "nl_interface.py").read_text()
    assert module_src.count("messages.create(") == 1


# ---------------------------------------------------------------------------
# docs/components/09, 10, 11, 12 — status + decisions lists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_COMPONENT_DOCS))
def test_component_doc_status_built_green_wk7_not_stub(name):
    status = _status_line(_COMPONENT_DOCS[name])
    assert "stub" not in status.lower(), f"{name} status still says stub"
    assert "built" in status.lower() and "green" in status.lower() and "wk 7" in status
    assert "## Decisions made in build" in _COMPONENT_DOCS[name]


def test_component_docs_record_the_four_spec_decisions():
    # Whitespace-normalised so the markdown's hard line wraps cannot split a phrase.
    decisions = {
        name: " ".join(_doc_section(doc, "## Decisions made in build").split())
        for name, doc in _COMPONENT_DOCS.items()
    }
    # Reject-not-truncate holdout dates (guardrails + mcp_server docs).
    for name in ("10-ai-guardrails", "11-mcp-server"):
        assert re.search(r"[Rr]eject", decisions[name]) and "truncate" in decisions[name], name
    # Conservative cache-token billing (budget + nl_interface docs).
    for name in ("09-ai-budget", "12-nl-interface"):
        assert "cache" in decisions[name].lower() and "full input rate" in decisions[name], name
    # Rate-limit file derived from AI_LEDGER_PATH.
    assert "AI_LEDGER_PATH" in decisions["10-ai-guardrails"]
    assert "ai_rate_limits.json" in decisions["10-ai-guardrails"]
    # score_holdout lazy-imports ENGINES (guardrails + mcp_server docs).
    for name in ("10-ai-guardrails", "11-mcp-server"):
        # "lazy-imports" (10) or "imports it lazily" (11): "lazily" does not contain "lazy".
        assert "ENGINES" in decisions[name] and re.search(r"laz(y|ily)", decisions[name]), name


def test_guardrails_doc_records_agent_loop_clause_landed_in_week8():
    # Pin advanced at the week-8 closeout: the doc used to defer the agent-loop clause to
    # week 8; now it must say the clause landed there (past tense), and nowhere still pending.
    doc = _COMPONENT_DOCS["10-ai-guardrails"]
    status = _status_line(doc)
    assert "test_holdout_isolation.py" in status and "week 8" in status
    assert "agent-loop" in doc and "week-8" in doc.lower().replace(" ", "-")
    assert "landed in week 8" in status
    assert "landed in week 8" in " ".join(doc.split())
    for stale in ("lands in week 8", "is added in week 8", "is a **week-8** item", "pending"):
        assert stale not in doc, f"10-ai-guardrails.md still says {stale!r}"
    # No doc in the set still describes the old clamp behaviour.
    for name, text in _COMPONENT_DOCS.items():
        assert "load_data clamps" not in text, f"{name} still says load_data clamps"
    assert "clamps" not in _DOC_AGENT


def test_agent_doc_status_built_green_wk8_with_decisions_and_the_built_signature():
    status = _status_line(_DOC_AGENT)
    assert "stub" not in status.lower()
    assert "built / green (wk 8)" in status
    for suite in ("test_agent.py", "test_holdout_isolation.py", "test_public_mode_no_codegen.py"):
        assert suite in status
    assert "## Decisions made in build" in _DOC_AGENT
    decisions = " ".join(_doc_section(_DOC_AGENT, "## Decisions made in build").split())
    assert "propose_experiment" in decisions and "declare_done" in decisions
    assert "strict" in decisions and "validate_params" in decisions
    assert "claude-sonnet-5" in decisions and "public_mode" in decisions
    # Stop vocabulary and result/record keys the doc advertises are the module's.
    assert "api_error" in _DOC_AGENT
    assert set(re.findall(r'"(converged|max_iters|budget|api_error)"', _DOC_AGENT)) == set(
        agent.STOP_REASONS
    )
    # The Interface block's signature is the built one: a doc may not overclaim a kwarg.
    interface = _doc_section(_DOC_AGENT, "## Interface")
    params = inspect.signature(agent.run_research).parameters
    assert list(params) == ["goal", "max_iters", "tickers", "engine", "cost_bps", "client"]
    for name in ("tickers", "engine", "client", "cost_bps", "max_iters"):
        assert re.search(rf"\b{name}\b", interface), f"interface block does not show {name}"
    assert "public_mode" not in interface
    assert params["max_iters"].default == guardrails.MAX_AGENT_ITERS
    assert "MAX_AGENT_ITERS" in interface
    # The live-run item is honestly marked pending the key, as in week 7.
    done = _doc_section(_DOC_AGENT, "## Done when")
    assert "pending" in done.lower() and "ANTHROPIC_API_KEY" in done


# ---------------------------------------------------------------------------
# docs/components/16-tests.md — inventory sync, no phantom suites
# ---------------------------------------------------------------------------


def _row(row_file: str) -> str:
    return next(line for line in _DOC_TESTS.splitlines() if line.startswith(f"| `{row_file}`"))


@pytest.mark.parametrize(
    "row_file",
    [
        "test_budget.py",
        "test_guardrails.py",
        "test_mcp_tools.py",
        "test_mcp_server.py",
        "test_nl_interface.py",
        "test_public_mode_no_codegen.py",
    ],
)
def test_tests_doc_week7_rows_green(row_file):
    row = _row(row_file)
    assert "green (wk 7" in row, f"{row_file} row not marked green wk 7"
    assert "(new)" not in row and "skipped →" not in row


def test_tests_doc_holdout_isolation_row_guard_level_with_agent_clause_wk8():
    # Pin advanced at the week-8 closeout: the clause is no longer forward-looking.
    row = _row("test_holdout_isolation.py")
    assert "green (wk 7 guard-level" in row and "wk 8" in row
    assert "agent-loop clause landed" in row
    assert "run_research" in row and "score_holdout" in row


def test_tests_doc_public_mode_row_mentions_the_agent_clause():
    row = _row("test_public_mode_no_codegen.py")
    assert "green (wk 7" in row and "wk 8" in row and "agent clause" in row


def test_tests_doc_agent_row_present_green_wk8():
    row = _row("test_agent.py")
    assert "green (wk 8" in row
    assert "FakeClient" in row
    for req in ("FR-9", "SF-1", "SF-7", "SF-8", "RG-4"):
        assert req in row, f"test_agent.py row lacks {req}"
    for claim in ("holdout", "converged", "max_iters", "budget", "api_error", "test_live_smoke"):
        assert claim in row, f"test_agent.py row does not mention {claim}"


def test_tests_doc_week8_verifier_row_names_its_suites():
    row = next(line for line in _DOC_TESTS.splitlines() if "test_agent_verify.py" in line)
    for name in ("test_agent_run_research_verify.py", "test_week8_proof_suites_verify.py"):
        assert name in row, f"week-8 verifier row does not name {name}"
    assert "green (wk 8)" in row


def test_tests_doc_week7_verifier_row_names_its_suites():
    row = next(line for line in _DOC_TESTS.splitlines() if "test_budget_verify.py" in line)
    for name in (
        "test_guardrails_verifier.py",
        "test_mcp_tools_verify.py",
        "test_mcp_server_verify.py",
        "test_nl_interface_verify.py",
        "test_week7_closeout_verifier.py",
    ):
        assert name in row, f"week-7 verifier row does not name {name}"
    assert "green (wk 7)" in row


def test_tests_doc_every_named_file_exists():
    # Phantom sweep over the WHOLE inventory (the week-6 precedent only swept test_r_*): a row
    # claiming a green suite whose file does not exist in tests/ fails here.
    names = set(re.findall(r"`(test_\w+\.py)`", _DOC_TESTS))
    assert len(names) >= 36
    for name in sorted(names):
        assert (_ROOT / "tests" / name).is_file(), f"16-tests.md names phantom suite {name}"


def test_tests_doc_status_line_advanced_to_week8():
    # Pin advanced at the week-8 closeout ("wk 1–7" -> "wk 1–8"; both live smokes named).
    head = "\n".join(_DOC_TESTS.splitlines()[:6])
    assert "**Status:**" in head
    assert "wk 1–8" in head or "wk 1-8" in head
    assert "only the" in head and "live-AI smoke tests skip" in head
    assert "test_nl_interface.py::test_live_smoke" in head
    assert "test_agent.py::test_live_smoke" in head
    assert "QUANTFORGE_LIVE_AI" in head
    assert "skipif" in head  # the machine-local R caveat survives


# ---------------------------------------------------------------------------
# README — the "AI layer safety" paragraph
# ---------------------------------------------------------------------------


def test_readme_ai_layer_safety_paragraph_present_with_links_that_resolve():
    section = _doc_section(_README, "## Methodology and known limitations")
    assert "AI layer safety" in section
    para = section[section.index("AI layer safety") :]
    para = para.split("\n- **")[0]  # up to the next bullet, if any
    assert "src/quantforge/ai/budget.py" in para
    assert "src/quantforge/ai/guardrails.py" in para
    assert "ai/nl_interface.py" in para
    assert "parameter-only" in para and "PUBLIC_MODE=on" in para
    assert "never executed" in para
    for suite in (
        "tests/test_public_mode_no_codegen.py",
        "tests/test_holdout_isolation.py",
        "tests/test_budget.py",
    ):
        assert suite in para, f"README safety paragraph does not cite {suite}"
    assert "docs/components/18-runtime-config.md" in para
    # Week-8 extension: the research agent's caps and the one-shot holdout, with its proofs.
    assert "research agent" in para and "run_research" in para
    assert "10-iteration cap" in para and "MAX_AGENT_ITERS" in para
    assert re.search(r"\bonce\b", para) and "side by side" in para
    assert "tests/test_agent.py" in para
    assert guardrails.MAX_AGENT_ITERS == 10  # the README number is the constant
    # Every relative link in the whole section resolves to a real file (dead links fail).
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", section):
        if target.startswith(("http://", "https://")):
            continue
        assert (_ROOT / target).exists(), f"README links to missing file {target}"
    # The spec's own grep must find the paragraph.
    assert re.search(r"nl_interface|ai/guardrails", _README)


# ---------------------------------------------------------------------------
# Git hygiene — no key or spend ledger can ride along in the closeout commit
# ---------------------------------------------------------------------------


def test_env_and_ai_state_files_are_gitignored():
    if shutil.which("git") is None or not (_ROOT / ".git").exists():
        pytest.skip("git or the repo metadata is unavailable")
    # Both state files AND their flock sidecars: the first rate_limit() call at the default
    # path creates ai_rate_limits.json.lock, which would otherwise ride along in `git add -A`.
    for rel in (
        ".env",
        "data_cache/ai_ledger.json",
        "data_cache/ai_ledger.json.lock",
        "data_cache/ai_rate_limits.json",
        "data_cache/ai_rate_limits.json.lock",
    ):
        proc = subprocess.run(
            ["git", "check-ignore", "-q", rel],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"{rel} is not gitignored"
    # And .env.example itself must not carry a real-looking key.
    assert _env_example_vars()["ANTHROPIC_API_KEY"] == "..."
    assert not re.search(r"sk-ant-[A-Za-z0-9_-]{10,}", _ENV_EXAMPLE)
