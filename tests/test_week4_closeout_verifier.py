"""Verifier tests for the Week-4 closeout milestone (offline, synthetic data only).

Independently proves the closeout's four deliverables so none can silently regress:

- docs/TEN_WEEK_PLAN.md: both Week-4 boxes ticked, exactly 12 ticked overall, Weeks 5-10 +
  Stretch untouched.
- The three component docs reflect the new statuses: no stale "stub" in 05-strategies.md, no
  stale "t-1" timing convention in 03-engine-base.md, the freeze pinned by name, and
  16-tests.md's inventory listing both week-4 suites as green.
- handoff.md's topmost entry is the dated 2026-08-30 closeout naming the five deliverables,
  the exact final gate counts, the carried R-side open item, and Week 5 as next.
- The new `to_long` duplicate-column guard: a hand-built wide frame with duplicated column
  labels raises SchemaError naming every offending label — never the bare pandas
  AttributeError it used to surface — while the good-path wide<->long round trip is untouched
  (hand-computed expected long frame, exact equality).

Adversarial cases: duplicated labels hiding non-numeric values (guard must fire before the
dtype loop), multiple distinct duplicated labels (all must be named), and a NaN cell inside
the duplicated pair (must still be rejected, not half-melted).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantforge.interchange import SchemaError, to_long, to_wide

_ROOT = Path(__file__).resolve().parent.parent
_PLAN = (_ROOT / "docs" / "TEN_WEEK_PLAN.md").read_text()
_DOC_STRATEGIES = (_ROOT / "docs" / "components" / "05-strategies.md").read_text()
_DOC_ENGINE_BASE = (_ROOT / "docs" / "components" / "03-engine-base.md").read_text()
_DOC_TESTS = (_ROOT / "docs" / "components" / "16-tests.md").read_text()
_HANDOFF = (_ROOT / "handoff.md").read_text()


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


def test_plan_week4_both_items_ticked_and_named():
    body = _plan_section("Week 4")
    assert "- [ ]" not in body, "Week 4 still has an unchecked box"
    assert body.count("- [x]") == 2
    assert "Mean-reversion" in body
    assert "Freeze the interface" in body


def test_plan_no_other_checkbox_changed():
    # Weeks 1-3 stay fully ticked (4+3+3), Week 4 adds 2 — exactly 12 in the whole file, so a
    # stray tick anywhere later (or an untick earlier) fails here.
    assert _PLAN.count("- [x]") == 12
    for week, n in {"Week 1": 4, "Week 2": 3, "Week 3": 3}.items():
        body = _plan_section(week)
        assert body.count("- [x]") == n and "- [ ]" not in body
    for later in ("Week 5", "Week 6", "Week 7", "Week 8", "Week 9", "Week 10", "Stretch"):
        body = _plan_section(later)
        assert "- [x]" not in body, f"{later} has a prematurely ticked box"
        assert "- [ ]" in body, f"{later} lost its checklist"


# ---------------------------------------------------------------- (2) component docs


def test_strategies_doc_reflects_both_strategies_working():
    assert "stub" not in _DOC_STRATEGIES.lower(), (
        "stale 'stub' wording survives in 05-strategies.md"
    )
    assert "both strategies working" in _DOC_STRATEGIES
    # The Done-when section must record how the criteria were met, by test-suite name.
    assert "test_strategies.py" in _DOC_STRATEGIES
    assert "test_interface_freeze.py" in _DOC_STRATEGIES


def test_engine_base_doc_freeze_in_effect_and_no_stale_timing():
    # The old draft documented a "close of t-1" signal convention; the frozen contract is
    # close-of-t decision + engine-applied shift. Neither ASCII nor Unicode-minus forms may remain.
    assert "t-1" not in _DOC_ENGINE_BASE and "t−1" not in _DOC_ENGINE_BASE
    assert "FROZEN as of week 4" in _DOC_ENGINE_BASE
    assert "test_interface_freeze.py" in _DOC_ENGINE_BASE


def test_tests_doc_inventory_lists_week4_suites_green():
    strategies_row = next(
        line for line in _DOC_TESTS.splitlines() if line.startswith("| `test_strategies.py`")
    )
    assert "(new)" not in strategies_row, "test_strategies.py still marked as not-yet-written"
    assert "green" in strategies_row and "wk 4" in strategies_row
    freeze_row = next(
        line for line in _DOC_TESTS.splitlines() if line.startswith("| `test_interface_freeze.py`")
    )
    assert "green" in freeze_row and "wk 4" in freeze_row


# ---------------------------------------------------------------- (3) handoff entry


def _handoff_top_entry() -> tuple[str, str]:
    entries = re.split(r"^## ", _HANDOFF, flags=re.MULTILINE)[1:]
    heading, _, body = entries[0].partition("\n")
    date = re.match(r"(\d{4}-\d{2}-\d{2})", heading).group(1)
    return date, heading + "\n" + body


def test_handoff_top_entry_is_the_week4_closeout():
    date, body = _handoff_top_entry()
    assert date == "2026-08-30"
    assert "Week 4" in body
    # The five deliverables, by load-bearing name.
    for needle in (
        "mean_reversion",
        "PARAM_WHITELIST",
        "test_strategies.py",
        "test_interface_freeze.py",
        "to_long",
    ):
        assert needle in body, f"handoff entry missing deliverable {needle!r}"
    # Exact final gate results, the carried open item, and the next target.
    assert "304 passed / 2 skipped" in body
    assert "ruff" in body
    assert "week 6" in body.lower()  # R-side Parquet check carried forward
    assert "Week 5" in body and "ortfolio" in body


# ---------------------------------------------------------------- (4) to_long guard


def _wide(columns: list[str], rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=columns,
        index=pd.date_range("2024-01-02", periods=len(rows), tz="UTC", name="date"),
    )


def test_duplicated_column_label_raises_schema_error_naming_it():
    wide = _wide(["AAPL", "AAPL", "GOOG"], [[10.0, 11.0, 20.0], [12.0, 13.0, 21.0]])
    with pytest.raises(SchemaError, match=r"AAPL"):
        to_long(wide, "prices")


def test_duplicated_label_never_surfaces_a_bare_pandas_error():
    # The pre-guard failure mode: df[col] on a duplicated label returns a DataFrame, which has
    # no .dtype, so the caller saw AttributeError. The boundary must raise its one exception
    # type (SchemaError is a ValueError) — prove the old bug can't come back.
    wide = _wide(["AAPL", "AAPL"], [[10.0, 11.0], [12.0, 13.0]])
    try:
        to_long(wide, "prices")
    except SchemaError:
        pass  # the contractually-required outcome
    except (AttributeError, KeyError) as err:  # pragma: no cover - only on regression
        pytest.fail(f"duplicated label leaked a bare pandas error: {type(err).__name__}: {err}")
    else:  # pragma: no cover - only on regression
        pytest.fail("duplicated column labels were silently accepted")


def test_all_distinct_duplicated_labels_are_named_once():
    wide = _wide(
        ["AAPL", "GOOG", "AAPL", "GOOG", "MSFT"],
        [[1.0, 2.0, 3.0, 4.0, 5.0]],
    )
    with pytest.raises(SchemaError) as excinfo:
        to_long(wide, "prices")
    msg = str(excinfo.value)
    assert "AAPL" in msg and "GOOG" in msg
    assert "MSFT" not in msg, "a non-duplicated label was wrongly reported"


def test_guard_fires_before_dtype_loop_even_with_non_numeric_duplicates():
    # Adversarial: the duplicated columns hold strings. Without the guard running first, the
    # dtype loop's df[col].dtype would raise AttributeError before any dtype message could form.
    wide = pd.DataFrame(
        [["a", "b", 1.0]],
        columns=["AAPL", "AAPL", "GOOG"],
        index=pd.date_range("2024-01-02", periods=1, tz="UTC"),
    )
    with pytest.raises(SchemaError, match=r"duplicated column label"):
        to_long(wide, "prices")


def test_duplicated_label_with_nan_cell_still_rejected():
    # A NaN inside the duplicated pair must not let the frame slip through half-melted:
    # rejection happens on labels alone, before any value handling.
    wide = _wide(["AAPL", "AAPL"], [[10.0, np.nan], [np.nan, 13.0]])
    with pytest.raises(SchemaError, match=r"AAPL"):
        to_long(wide, "positions")


def test_guard_applies_to_every_panel_kind_and_names_the_kind():
    for kind in ("prices", "positions", "asset_returns"):
        wide = _wide(["X", "X"], [[0.1, 0.2]])
        with pytest.raises(SchemaError, match=kind):
            to_long(wide, kind)


def test_good_path_round_trip_unchanged_by_the_guard():
    # Behavior preservation: a ragged panel with unique labels still round-trips exactly, and
    # the long output matches a hand-built expectation row for row.
    d1, d2 = pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")
    long = pd.DataFrame(
        {
            "date": [d1, d1, d2, d2, d2],
            "ticker": ["AAPL", "GOOG", "AAPL", "GOOG", "MSFT"],
            "close": [10.0, 20.0, 11.0, 21.0, 30.0],
        }
    )
    round_tripped = to_long(to_wide(long, "prices"), "prices")
    pd.testing.assert_frame_equal(round_tripped, long)
