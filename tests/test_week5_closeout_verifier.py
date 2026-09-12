"""Verifier tests for the Week-5 docs-closeout milestone (docs-only; offline, no heavy deps).

Independently proves the closeout's four deliverables so none can silently regress:

- docs/TEN_WEEK_PLAN.md: all three Week-5 boxes ticked and named, exactly 21 ticked overall
  (pins advanced at the Week-7 closeout, per precedent), Weeks 8-10 + Stretch untouched.
- docs/components/07-portfolio.md: status advanced off "stub" AND the RG-6 R cross-check state
  is explicit — as of week 6 it must be stated as *landed* (with the machine-local caveat),
  never silently omitted and never regressed back to the pre-week-6 "NOT done" wording.
- docs/components/14-streamlit-app.md: status "shell built", with the live tabs (Backtest,
  Portfolio, Methodology) and the placeholders (AI Chat, Research mode) each named as such.
- docs/components/16-tests.md: the week-5 suites listed green — and, adversarially, every test
  file the inventory claims exists must actually exist in tests/ (no phantom suites).
- handoff.md: the 2026-08-31 Week-5 closeout entry is preserved in the history (looked up by
  content, not position, so later closeouts can prepend on top — the week-3/4 precedent),
  sitting *above* the 2026-08-30 entry, carrying the exact gate counts recorded at closeout,
  the two week-6 open items (R-side Parquet read check; R cross-check of optimizer results),
  the commit-pending item, and "Week 6 — R analytics layer" as next. Entry dates must be
  strictly newest-first.

Adversarial cases: a single stray "[x]" anywhere later in the plan fails the exact-count pin;
a doc row claiming a green suite whose file is missing fails the inventory sweep; a handoff
entry prepended out of date order (or the Week-5 entry deleted from the history) fails; and
the 07 doc regressing to the pre-week-6 "NOT done" caveat fails.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PLAN = (_ROOT / "docs" / "TEN_WEEK_PLAN.md").read_text()
_DOC_PORTFOLIO = (_ROOT / "docs" / "components" / "07-portfolio.md").read_text()
_DOC_APP = (_ROOT / "docs" / "components" / "14-streamlit-app.md").read_text()
_DOC_TESTS = (_ROOT / "docs" / "components" / "16-tests.md").read_text()
_HANDOFF = (_ROOT / "handoff.md").read_text()
_TESTS_DIR = _ROOT / "tests"


def _plan_section(week_prefix: str) -> str:
    """Body of the plan section whose '## '/'### ' heading starts with week_prefix.

    'Stretch' is a '### ' sub-heading at the end of the file, so split on both levels — a
    ticked Stretch box must be attributable to Stretch, not blamed on Week 10.
    """
    parts = re.split(r"^#{2,3} ", _PLAN, flags=re.MULTILINE)[1:]
    for part in parts:
        heading, _, body = part.partition("\n")
        if heading.startswith(week_prefix):
            return body
    raise AssertionError(f"no plan section starting with {week_prefix!r}")


# ---------------------------------------------------------------- (1) plan checkboxes


def test_plan_week5_all_three_items_ticked_and_named():
    body = _plan_section("Week 5")
    assert "- [ ]" not in body, "Week 5 still has an unchecked box"
    assert body.count("- [x]") == 3
    assert "PyPortfolioOpt" in body and "frontier" in body
    assert "Combine strategies" in body
    assert "Streamlit shell" in body


def test_plan_exactly_21_ticked_and_weeks_8_plus_untouched():
    # Adversarial exact count: 4+3+3+2+3+3+3 for Weeks 1-7 (pin advanced at the week-7
    # closeout) — a stray tick anywhere later in the file (or an untick earlier) fails here.
    assert _PLAN.count("- [x]") == 21
    weeks = {
        "Week 1": 4,
        "Week 2": 3,
        "Week 3": 3,
        "Week 4": 2,
        "Week 5": 3,
        "Week 6": 3,
        "Week 7": 3,
    }
    for week, n in weeks.items():
        body = _plan_section(week)
        assert body.count("- [x]") == n and "- [ ]" not in body
    for later in ("Week 8", "Week 9", "Week 10", "Stretch"):
        body = _plan_section(later)
        assert "- [x]" not in body, f"{later} has a prematurely ticked box"
        assert "- [ ]" in body, f"{later} lost its checklist"


# ---------------------------------------------------------------- (2) component docs


def _status_line(doc: str) -> str:
    return next(line for line in doc.splitlines() if "**Status:**" in line)


def test_portfolio_doc_status_working_not_stub():
    status = _status_line(_DOC_PORTFOLIO)
    assert "stub" not in status.lower(), "07-portfolio.md status line still says stub"
    assert "working" in status.lower()


def test_portfolio_doc_r_cross_check_state_explicit():
    # Week-5 state: the doc explicitly said the RG-6 R cross-check was NOT done. Week-6 pin
    # advance: the cross-check landed, so the doc must now say so explicitly — status line
    # names the landing, the body names the proving suite — and the stale pre-week-6 "NOT
    # done"/"single-implementation" caveat must be gone (not merely contradicted elsewhere).
    assert "landed wk 6" in _status_line(_DOC_PORTFOLIO).lower()
    assert "cross-check is done" in _DOC_PORTFOLIO
    assert "test_r_cross_check.py" in _DOC_PORTFOLIO
    assert not re.search(r"NOT done", _DOC_PORTFOLIO), (
        "07-portfolio.md still carries the pre-week-6 'NOT done' RG-6 caveat"
    )
    assert "single-implementation" not in _DOC_PORTFOLIO, (
        "stale single-implementation-verified caveat survived the week-6 flip"
    )


def test_app_doc_status_shell_built_with_tab_inventory():
    status = _status_line(_DOC_APP)
    assert "stub" not in status.lower(), "14-streamlit-app.md status line still says stub"
    assert "shell built" in status.lower()
    # Live tabs and placeholders each named, and the placeholder label applied to the AI tabs.
    slice_para = _DOC_APP[_DOC_APP.index("Week-5 slice") : _DOC_APP.index("## Function")]
    for live in ("Backtest", "Portfolio", "Methodology"):
        assert live in slice_para, f"live tab {live!r} not listed in the week-5 slice"
    assert "AI Chat" in slice_para and "Research mode" in slice_para
    assert "placeholder" in slice_para.lower()


def test_tests_doc_lists_week5_suites_green():
    for name in (
        "test_portfolio_optimize.py",
        "test_portfolio_combination.py",
        "test_app_shell.py",
    ):
        row = next(
            (line for line in _DOC_TESTS.splitlines() if line.startswith(f"| `{name}`")),
            None,
        )
        assert row is not None, f"16-tests.md has no inventory row for {name}"
        assert "green" in row and "wk 5" in row, f"{name} row not marked green for week 5"
    # The old planned placeholder row must be gone — it was superseded, not duplicated.
    assert "`test_portfolio.py`" not in _DOC_TESTS


def test_tests_doc_names_no_phantom_files():
    # Adversarial sweep: every tests/test_*.py file named anywhere in the inventory that is
    # marked green in its row must exist on disk. Docs may plan future suites ("(new)" rows),
    # but a green claim for a missing file is a lie the doc cannot tell.
    for line in _DOC_TESTS.splitlines():
        if not line.startswith("|") or "green" not in line:
            continue
        for name in re.findall(r"`(test_[A-Za-z0-9_]+\.py)`", line):
            assert (_TESTS_DIR / name).is_file(), f"16-tests.md claims green suite {name}: missing"


# ---------------------------------------------------------------- (3) handoff entry


def _handoff_entries() -> list[tuple[str, str]]:
    """(date, full text) per '## YYYY-MM-DD — ...' entry, in file order (newest first)."""
    entries = re.split(r"^## ", _HANDOFF, flags=re.MULTILINE)[1:]
    out = []
    for entry in entries:
        heading, _, body = entry.partition("\n")
        match = re.match(r"(\d{4}-\d{2}-\d{2})", heading)
        assert match, f"handoff entry heading has no leading date: {heading!r}"
        out.append((match.group(1), heading + "\n" + body))
    return out


def test_handoff_week5_entry_preserved_and_dates_descend():
    entries = _handoff_entries()
    # Looked up by content, not position (week-3/4 precedent): later closeouts prepend on top.
    assert any(d == "2026-08-31" and "Week 5" in b for d, b in entries), (
        "the 2026-08-31 Week-5 closeout entry was deleted from the handoff history"
    )
    # Adversarial ordering pin: an entry prepended out of order (or the Week-5 entry demoted
    # below the 2026-08-30 one) breaks the newest-first contract AGENTS.md requires.
    dates = [d for d, _ in entries]
    assert dates == sorted(dates, reverse=True), f"handoff entries out of date order: {dates}"
    assert "2026-08-30" in dates, "the Week-4 entry was deleted, not preserved"


def test_handoff_week5_entry_contents():
    body = next(b for d, b in _handoff_entries() if d == "2026-08-31" and "Week 5" in b)
    # Gate results recorded at closeout, by exact count, plus the ruff status.
    assert "428 passed / 2 skipped" in body
    assert "ruff" in body.lower() and "clean" in body.lower()
    # The deliverables, by load-bearing name.
    for needle in (
        "optimize.py",
        "streamlit_app.py",
        "test_portfolio_optimize.py",
        "test_portfolio_combination.py",
        "test_app_shell.py",
    ):
        assert needle in body, f"handoff entry missing deliverable {needle!r}"
    # Agent-failures section present (AGENTS.md requires it even when empty).
    assert "failures" in body.lower()
    # Open items: both week-6 rigor items and the commit-pending item.
    assert "Parquet read check" in body
    assert "cross-check" in body and "RG-6" in body
    assert "approval" in body, "commit-pending-approval open item missing"
    # Next target.
    assert "Week 6" in body and "R analytics layer" in body
