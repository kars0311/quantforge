"""Verifier tests pinning the Week-6 closeout state (docs-only milestone; offline, no R).

Independently proves the closeout's deliverables so none can silently regress:

- docs/TEN_WEEK_PLAN.md: all three Week-6 boxes ticked, the third explicitly annotated as
  satisfied by the written methodology notes; plan-count and handoff-position pins advanced at
  the week-7 closeout per precedent (21 ticked overall, Weeks 8-10 + Stretch untouched; the
  Week-6 handoff entry sits directly below the Week-7 one).
- docs/components/08-r-tearsheet.md: status advanced off "stub"; the "Cross-language
  conventions and caveats" section names every convention the spec demanded (population vs
  sample std with the sd()*sqrt((n-1)/n) correction, ANN=252 observation-count CAGR,
  risk-free 0, the min-variance cross-check objective, the ~1% vs ~1e-9 tolerance rationale,
  and the AR-3 display-tables note).
- Docs may not overclaim what the code does: the dual tolerance bounds the doc advertises must
  actually be asserted in tests/test_r_cross_check.py, and the population-std correction the
  doc describes must actually appear in analytics_r/tearsheet.R. A doc paragraph describing
  rigor that the code dropped fails here — the docs-milestone analog of a prescient signal.
- README's "Methodology and known limitations" carries a pointer to the notes, and the anchor
  it links to resolves to a real heading in the target doc (no dead fragment).
- docs/components/07-portfolio.md: the pre-week-6 "NOT done"/"single-implementation" RG-6
  caveat is gone and the landed cross-check is named with its proving suite.
- docs/components/16-tests.md: the R rows are green wk 6, and every week-6 test file the
  inventory claims exists actually exists (no phantom suites).
- handoff.md: the dated 2026-09-01 Week-6 closeout entry (now directly below the Week-7 one) with the exact gate
  counts (488 passed / 2 skipped), the long-carried R-side Parquet read check explicitly
  closed, the commit-pending and machine-local-R notes, and Week 7 as next. Dates stay
  newest-first.

Adversarial cases: a single stray "[x]" anywhere later in the plan fails the exact-count pin;
the doc claiming a 1e-9 transcription bound that test_r_cross_check.py stops asserting fails
the consistency sweep; a README pointer whose anchor no longer matches the doc heading fails;
a 16-tests.md row naming a missing file fails; the handoff Week-6 entry demoted below Week 5
(or its counts edited) fails.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PLAN = (_ROOT / "docs" / "TEN_WEEK_PLAN.md").read_text()
_DOC_TEARSHEET = (_ROOT / "docs" / "components" / "08-r-tearsheet.md").read_text()
_DOC_PORTFOLIO = (_ROOT / "docs" / "components" / "07-portfolio.md").read_text()
_DOC_TESTS = (_ROOT / "docs" / "components" / "16-tests.md").read_text()
_README = (_ROOT / "README.md").read_text()
_HANDOFF = (_ROOT / "handoff.md").read_text()


def _plan_section(week: str) -> str:
    """Body of one '## Week N — ...' (or '### Stretch') section of the plan."""
    parts = re.split(r"^#{2,3} ", _PLAN, flags=re.M)
    for part in parts:
        if part.startswith(week):
            return part
    raise AssertionError(f"plan section {week!r} not found")


def _status_line(doc: str) -> str:
    return next(line for line in doc.splitlines() if line.startswith("**Week"))


# ---------------------------------------------------------------------------
# docs/TEN_WEEK_PLAN.md
# ---------------------------------------------------------------------------


def test_plan_week6_all_three_items_ticked_and_named():
    body = _plan_section("Week 6")
    assert body.count("- [x]") == 3 and "- [ ]" not in body
    assert "tearsheet.R" in body
    assert "PortfolioAnalytics" in body
    assert "Pressure-test the stats methodology" in body


def test_plan_week6_pressure_test_annotated_as_satisfied_by_notes():
    # The spec: the third box is satisfied by the written rigor notes, and the plan must say
    # so where the box is ticked — a bare tick with no provenance fails.
    body = _plan_section("Week 6")
    third = body[body.index("Pressure-test") :]
    assert "satisfied by" in third
    assert "Cross-language conventions and caveats" in third
    assert "08-r-tearsheet.md" in third


def test_plan_exactly_24_ticked_and_weeks_9_plus_untouched():
    # Adversarial exact count: 4+3+3+2+3+3+3+3 for Weeks 1-8 (pin advanced at the week-8
    # closeout, per precedent) — a stray tick anywhere later in the file (or an untick
    # earlier) fails here.
    assert _PLAN.count("- [x]") == 24
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
        assert body.count("- [x]") == n and "- [ ]" not in body
    for later in ("Week 9", "Week 10", "Stretch"):
        body = _plan_section(later)
        assert "- [x]" not in body, f"{later} has a prematurely ticked box"
        assert "- [ ]" in body, f"{later} lost its checklist"


# ---------------------------------------------------------------------------
# docs/components/08-r-tearsheet.md — the methodology notes themselves
# ---------------------------------------------------------------------------


def test_tearsheet_doc_status_built_not_stub():
    status = _status_line(_DOC_TEARSHEET)
    assert "stub" not in status.lower()
    assert "built" in status.lower()


def test_tearsheet_doc_conventions_section_names_every_required_convention():
    assert "## Cross-language conventions and caveats" in _DOC_TEARSHEET
    section = _DOC_TEARSHEET[_DOC_TEARSHEET.index("## Cross-language conventions") :]
    section = section.split("\n## ")[0]  # up to the next H2
    # Population vs sample std, with the exact correction formula.
    assert "opulation" in section and "sample" in section
    compact = section.replace(" ", "").replace("`", "")
    assert "sd(r)*sqrt((n-1)/n)" in compact or "sd()*sqrt((n-1)/n)" in compact, (
        "the sd()*sqrt((n-1)/n) population-std correction is not spelled out"
    )
    # ANN=252 and observation-count (not calendar-time) CAGR.
    assert "252" in section
    assert "observation" in section.lower() and "calendar" in section.lower()
    # Risk-free 0.
    assert re.search(r"[Rr]isk-free (rate )?(=|of )?\s*(0|zero)", section)
    # Min-variance (not max-Sharpe) as the cross-check objective, with the uniqueness reason.
    assert "min-variance" in section.lower() or "minimum variance" in section.lower()
    assert "max-Sharpe" in section or "max-sharpe" in section.lower()
    assert "convex" in section.lower() and "unique" in section.lower()
    # Tolerance rationale: both the ~1% contract and the ~1e-9 observed agreement.
    assert "1%" in section and "1e-9" in section
    # AR-3: PerformanceAnalytics display tables keep their own conventions, display-only.
    assert "AR-3" in section
    assert "display-only" in section or "display only" in section


def test_tearsheet_doc_cites_product_decision_5_explained_not_hidden():
    # The section must anchor its rule to the recorded decision: differences explained, not
    # hidden — and that decision must still exist in docs/product.md saying the same thing.
    section = _DOC_TEARSHEET[_DOC_TEARSHEET.index("## Cross-language conventions") :]
    assert "product.md" in section and "explained, not hidden" in section
    product = (_ROOT / "docs" / "product.md").read_text()
    assert "differences explained, not" in product


# ---------------------------------------------------------------------------
# Docs may not overclaim: the cited code must actually do what the notes say
# ---------------------------------------------------------------------------


def test_doc_claimed_dual_bounds_are_actually_asserted_in_cross_check_suite():
    # The notes argue the ~1% contract alone would hide a dropped std correction, and that
    # test_r_cross_check.py therefore asserts BOTH bounds. Verify against the suite's source:
    # if someone later relaxes the 1e-9 transcription bound, the doc becomes a lie and this
    # test fails before the drift can hide.
    suite = (_ROOT / "tests" / "test_r_cross_check.py").read_text()
    assert re.search(r"_RG6_REL\s*=\s*0\.01", suite), "the ~1% RG-6 contract bound is gone"
    assert re.search(r"_TRANSCRIPTION_REL\s*=\s*1e-9", suite), (
        "the 1e-9 transcription bound the docs advertise is gone from test_r_cross_check.py"
    )
    for name in ("_RG6_REL", "_TRANSCRIPTION_REL"):
        assert suite.count(name) >= 2, f"{name} is defined but never used in an assertion"


def test_doc_claimed_population_std_correction_exists_in_r_source():
    # The notes describe the sd(r)*sqrt((n-1)/n) correction and the n=1 edge branch. Both must
    # exist in analytics_r/tearsheet.R itself, not just in prose.
    r_src = (_ROOT / "analytics_r" / "tearsheet.R").read_text()
    compact = r_src.replace(" ", "")
    assert "sd(r)*sqrt((n-1)/n)" in compact, (
        "tearsheet.R lost the population-std correction the methodology notes describe"
    )
    assert re.search(r"n\s*<\s*2", r_src), "the n<2 edge branch (sd()=NA vs numpy 0) is gone"


# ---------------------------------------------------------------------------
# README pointer
# ---------------------------------------------------------------------------


def test_readme_methodology_section_points_at_conventions_notes():
    assert "## Methodology and known limitations" in _README
    section = _README[_README.index("## Methodology and known limitations") :]
    section = section.split("\n## ")[0]
    assert "Cross-language" in section
    link = "docs/components/08-r-tearsheet.md#cross-language-conventions-and-caveats"
    assert link in section, "README's pointer to the conventions notes is missing"
    # Anchor validity: GitHub derives the fragment from the heading — if the heading is ever
    # reworded the link dies silently, so pin heading -> fragment agreement here.
    heading = "## Cross-language conventions and caveats"
    assert heading in _DOC_TEARSHEET
    fragment = heading[3:].lower().replace(" ", "-")
    assert link.endswith("#" + fragment)


# ---------------------------------------------------------------------------
# docs/components/07-portfolio.md — RG-6 caveat flipped
# ---------------------------------------------------------------------------


def test_portfolio_doc_rg6_caveat_flipped_to_done():
    assert not re.search(r"NOT (done|met)", _DOC_PORTFOLIO), (
        "07-portfolio.md still carries a pre-week-6 'NOT met/done' RG-6 caveat"
    )
    assert "single-implementation" not in _DOC_PORTFOLIO
    assert "cross-check is done" in _DOC_PORTFOLIO
    assert "test_r_cross_check.py" in _DOC_PORTFOLIO
    # The honest caveat that replaced it: the check needs a local R stack.
    assert "skipif" in _DOC_PORTFOLIO


# ---------------------------------------------------------------------------
# docs/components/16-tests.md — inventory sync, no phantom suites
# ---------------------------------------------------------------------------


def test_tests_doc_week6_rows_green_and_files_exist():
    for row_file in ("test_r_cross_check.py", "test_r_interchange.py"):
        row = next(line for line in _DOC_TESTS.splitlines() if line.startswith(f"| `{row_file}`"))
        assert "green (wk 6)" in row, f"{row_file} row not marked green wk 6"
        assert "(new)" not in row
    # Phantom sweep over every week-6 file the inventory names: a row claiming a green suite
    # whose file does not exist in tests/ fails here.
    for name in re.findall(r"`(test_r_\w+\.py)`", _DOC_TESTS):
        assert (_ROOT / "tests" / name).is_file(), f"16-tests.md names phantom suite {name}"


def test_tests_doc_status_line_advanced_to_week6():
    # The header status wraps across lines; check the doc's head, not one physical line.
    head = "\n".join(_DOC_TESTS.splitlines()[:6])
    assert "**Status:**" in head
    # Pin advanced at each closeout ("wk 1–6" -> "wk 1–7" -> "wk 1–8"): the R caveat must survive.
    assert "wk 1–8" in head or "wk 1-8" in head
    assert "skipif" in head  # the machine-local R caveat is stated up front


# ---------------------------------------------------------------------------
# handoff.md — the Week-6 closeout entry
# ---------------------------------------------------------------------------


def _handoff_entries() -> list[tuple[str, str]]:
    """(date, body) for each '## YYYY-MM-DD — ...' entry, in file order."""
    matches = list(re.finditer(r"^## (\d{4}-\d{2}-\d{2}) — ", _HANDOFF, flags=re.M))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(_HANDOFF)
        out.append((m.group(1), _HANDOFF[m.start() : end]))
    return out


def test_handoff_week6_entry_preserved_below_week7_and_dates_descend():
    # Pin advanced at the 2026-09-12 fix run: the newest entry is that run's open-items entry,
    # then the 2026-09-11 Week-8 one, then Week 7, and the Week-6 record must survive verbatim
    # enough to keep its counts, directly below that.
    entries = _handoff_entries()
    assert entries[0][0] == "2026-09-12" and "open-items" in entries[0][1]
    assert entries[1][0] == "2026-09-11" and "Week 8" in entries[1][1]
    assert entries[2][0] == "2026-09-10" and "Week 7" in entries[2][1]
    assert entries[3][0] == "2026-09-01" and "Week 6" in entries[3][1]
    dates = [d for d, _ in entries]
    assert dates == sorted(dates, reverse=True), "handoff entries are not newest-first"


def test_handoff_week6_entry_contents():
    body = next(b for d, b in _handoff_entries() if d == "2026-09-01" and "Week 6" in b)
    # Gate results recorded at closeout, by exact count, plus the ruff status.
    assert "488 passed / 2 skipped" in body
    assert "ruff" in body.lower() and "clean" in body.lower()
    # The long-carried week-1 open item is explicitly closed, not silently dropped.
    assert "R-side Parquet read check" in body
    assert "Resolved" in body
    assert "test_r_interchange.py" in body
    # Commit pending Kent's approval; machine-local R / bare-CI skip caveat; next target.
    assert "pending" in body.lower() and "approv" in body.lower()
    assert "machine-local" in body
    assert "Week 7" in body


def test_handoff_week5_entry_still_preserved_below():
    # Prepending week 6 must not have rewritten history: the week-5 entry survives verbatim
    # enough to keep its own closeout counts, below the new entry.
    entries = _handoff_entries()
    idx = next(i for i, (d, b) in enumerate(entries) if d == "2026-08-31" and "Week 5" in b)
    assert idx > 0
    assert "428 passed / 2 skipped" in entries[idx][1]
