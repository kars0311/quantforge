"""Verifier tests pinning the documented project state (docs-only; offline, no heavy deps).

Written at the Week-3 closeout and advanced at each subsequent closeout (currently Week 4) so
the docs cannot silently drift from reality:
- docs/TEN_WEEK_PLAN.md: exactly Weeks 1-4 ticked, Weeks 5-10 + Stretch untouched. The count is
  exact and adversarial — a single stray "[x]" anywhere later in the plan fails the suite, so
  nobody can quietly claim future work as done.
- handoff.md: the 2026-08-29 Week-3 closeout entry is preserved verbatim in the history (looked
  up by content, not position, so newer entries can be prepended on top), and entries stay
  newest-first (the AGENTS.md status-handoff contract a fresh session relies on).
- The survivorship caveat is genuinely documented (README section + authoritative note in
  data/loader.py), and the stale "~28 names" wording is gone from the loader design doc.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLAN = (REPO / "docs" / "TEN_WEEK_PLAN.md").read_text()
HANDOFF = (REPO / "handoff.md").read_text()
README = (REPO / "README.md").read_text()


def _plan_sections() -> dict[str, str]:
    """Split the plan into {heading: body} chunks keyed by the '## ...' line."""
    parts = re.split(r"^#{2,3} +(.+)$", PLAN, flags=re.MULTILINE)
    # parts[0] is the preamble; then alternating heading, body ('###' catches the Stretch section)
    headings = parts[1::2]
    bodies = parts[2::2]
    return dict(zip(headings, bodies, strict=True))


def test_plan_weeks_1_to_4_fully_checked():
    # Per-week minimum item counts guard against a week losing checklist items outright.
    for week, min_items in {"Week 1": 4, "Week 2": 3, "Week 3": 3, "Week 4": 2}.items():
        sections = _plan_sections()
        heading = next(h for h in sections if h.startswith(week))
        body = sections[heading]
        assert "- [ ]" not in body, f"{week} still has an unchecked box"
        assert body.count("- [x]") >= min_items, f"{week} lost checklist items"


def test_plan_weeks_5_plus_and_stretch_all_unchecked():
    # Adversarial exact count: the plan has precisely 12 ticked boxes (4+3+3+2 for Weeks 1-4).
    # Any extra "[x]" — in Week 5-10, Stretch, or a sneaky duplicate — fails here.
    assert PLAN.count("- [x]") == 12
    sections = _plan_sections()
    later = [h for h in sections if re.match(r"Week ([5-9]|10) ", h) or "Stretch" in h]
    assert len(later) == 7  # Weeks 5..10 + Stretch — all present, none deleted to game the count
    for heading in later:
        assert "- [x]" not in sections[heading], f"'{heading}' has a prematurely ticked box"
        assert "- [ ]" in sections[heading], f"'{heading}' lost its checklist"


def _handoff_entries() -> list[tuple[str, str]]:
    """(date, body) per '## YYYY-MM-DD — ...' entry, in file order (newest first)."""
    parts = re.split(r"^## (\d{4}-\d{2}-\d{2})", HANDOFF, flags=re.MULTILINE)
    dates = parts[1::2]
    bodies = parts[2::2]
    return list(zip(dates, bodies, strict=True))


def test_handoff_week3_closeout_entry_preserved():
    # Looked up by date + content (not position) so later closeouts can prepend entries on top
    # without rewriting history; the Week-3 record itself must stay intact.
    entries = _handoff_entries()
    assert len(entries) >= 3  # closeout + manual backtesting session + week-1 run
    body = next(b for d, b in entries if d == "2026-08-29" and "Week 3 rigor" in b)

    # The five work items, by their load-bearing names.
    for needle in (
        "python_engine.py",  # engine hardening
        "test_no_lookahead.py",
        "test_cost_accounting.py",
        "test_metrics_reference.py",
        "02-data-loader.md",  # docs item (README limitations + ~28 -> 30 fix)
    ):
        assert needle in body, f"closeout entry is missing work item {needle!r}"

    # Exact recorded gate results (verified against a fresh run at review time), > 130 baseline.
    match = re.search(r"(\d+) passed / (\d+) skipped", body)
    assert match, "closeout entry does not record pytest counts"
    assert int(match.group(1)) > 130
    assert int(match.group(2)) == 2
    assert "ruff" in body

    # The deliberate non-touch of the temporary meta-test, on record.
    assert "test_tmp_validation_has_teeth.py" in body

    # Open items carried forward + next target.
    assert "is_unique" in body
    assert "read_parquet" in body
    assert "Kent" in body  # commit pending approval
    assert "Week 4" in body and "mean-reversion" in body


def test_handoff_entries_stay_newest_first():
    # Adversarial ordering check: appending (instead of prepending) an entry fails here.
    dates = [d for d, _ in _handoff_entries()]
    assert dates == sorted(dates, reverse=True), f"handoff entries out of order: {dates}"


def test_survivorship_caveat_documented_in_readme_and_loader():
    section = README.split("## Methodology and known limitations", 1)
    assert len(section) == 2, "README lost the 'Methodology and known limitations' section"
    body = section[1].split("\n## ", 1)[0]
    assert "urvivorship" in body
    assert "look-ahead" in body.lower()
    assert "data/loader.py" in body  # points at the authoritative note

    loader_src = (REPO / "src" / "quantforge" / "data" / "loader.py").read_text()
    assert "urvivorship" in loader_src, "authoritative survivorship note missing from loader.py"


def test_loader_doc_universe_count_fixed():
    doc = (REPO / "docs" / "components" / "02-data-loader.md").read_text()
    assert "~28" not in doc, "stale '~28 names' wording is back"
    assert "30 names" in doc
