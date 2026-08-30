"""Verifier for the RG-3 documentation milestone: README methodology section + doc/code consistency.

Docs rot silently — a README claim nobody re-checks is worse than no claim. These tests make the
Week-3 rigor documentation *executable*: if the universe changes size, a cited proving test is
renamed, or the engine's cost model drifts from what the README promises, the suite goes red.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.loader import UNIVERSE
from quantforge.engine.python_engine import PythonEngine

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text()
LOADER_DOC = (REPO / "docs" / "components" / "02-data-loader.md").read_text()


def _methodology_section() -> str:
    """The '## Methodology...' section body, up to the next '## ' heading."""
    m = re.search(r"^## .*(?:methodology|limitations).*?$", README, re.IGNORECASE | re.MULTILINE)
    assert m, "README has no 'Methodology / known limitations' section heading"
    rest = README[m.end() :]
    nxt = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def test_readme_survivorship_caveat_is_present_and_honest():
    sec = _methodology_section().lower()
    assert "survivorship" in sec
    # The caveat must name the actual shape of the bias, not just the word.
    assert "30" in sec, "must state the fixed universe size"
    assert "point-in-time" in sec, "must name the proper (non-goal) fix"
    assert "optimistic" in sec, "must admit the direction of the bias"


def test_readme_cites_each_proving_test_and_all_linked_files_exist():
    sec = _methodology_section()
    assert "tests/test_no_lookahead.py" in sec
    assert "tests/test_cost_accounting.py" in sec
    assert "tests/test_engine_vs_backtestingpy.py" in sec
    # Every relative markdown link in the section must resolve — a dangling doc link is a defect.
    for _, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", sec):
        if target.startswith(("http://", "https://", "#")):
            continue
        assert (REPO / target.split("#")[0]).exists(), f"README links to missing file: {target}"


def test_readme_states_the_conventions_the_code_actually_implements():
    sec = _methodology_section()
    assert "shift(1)" in sec, "must state the mechanical no-look-ahead rule"
    assert re.search(r"10\s*bps", sec), "must state the 10 bps default cost assumption"
    assert "turnover" in sec.lower(), "costs are charged on turnover, README must say so"
    assert "borrow" in sec.lower(), "unmodeled short borrow fees must be admitted"
    assert "cost_bps" in sec, "must name the override parameter"


def test_universe_is_30_and_component_doc_matches_the_frozen_code_list():
    assert len(UNIVERSE) == 30
    assert len(set(UNIVERSE)) == 30, "frozen universe must not contain duplicates"
    assert "~28" not in LOADER_DOC, "stale '~28' wording must be gone from 02-data-loader.md"
    assert "30 names" in LOADER_DOC
    # Adversarial: the tickers *listed* in the doc must be exactly the code's frozen UNIVERSE —
    # a doc that says "30" but lists different names would still be wrong.
    doc_tickers = set()
    for line in LOADER_DOC.splitlines():
        if re.match(r"\s*- (Tech|ADRs|Financials|Energy)", line):
            doc_tickers.update(re.findall(r"\b[A-Z]{1,5}\b", line.split(":", 1)[1]))
    doc_tickers -= {"US", "USD"}  # prose in the ADR line, not tickers
    assert doc_tickers == set(UNIVERSE), (
        f"doc/code universe mismatch: doc-only={doc_tickers - set(UNIVERSE)}, "
        f"code-only={set(UNIVERSE) - doc_tickers}"
    )


def test_loader_docstring_is_the_authoritative_caveat_the_readme_points_at():
    # README cross-links loader.py instead of duplicating the wording; the target must hold it.
    from quantforge.data import loader

    assert "SURVIVORSHIP" in loader.__doc__.upper()
    assert "src/quantforge/data/loader.py" in _methodology_section()


def test_readme_cost_and_lookahead_claims_are_executable_truth():
    """Adversarial: run the engine and check it does exactly what the README promises.

    A same-day-peeking signal (long iff *today* was an up day — information unknowable at the
    decision close under the README's convention) must earn ~nothing once shift(1) delays it onto
    day t+1's independent return. And the 10 bps documented default must charge exactly
    turnover * 10/10_000 on a hand-built case.
    """
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=250)
    rets = pd.Series(rng.normal(0.0, 0.01, size=len(dates)), index=dates)
    prices = pd.DataFrame({"A": 100.0 * (1.0 + rets).cumprod()}, index=dates)

    # Cheat: position on day t = sign of day t's own return. If the engine ever paid same-day
    # returns (no shift), this would earn |r| every up day; under honest shift(1) it earns
    # day t+1's return, which is i.i.d. of the peek — expectation zero.
    prescient = pd.DataFrame(
        {"A": np.sign(prices["A"].pct_change()).fillna(0.0).clip(lower=0.0)},
        index=dates,
    )
    net = PythonEngine().run_backtest(prices, prescient, {"cost_bps": 0}).returns
    assert abs(net.mean()) < 3.0 * rets.abs().mean() / np.sqrt(len(dates)), (
        "prescient signal earned a systematic profit — look-ahead leak contradicts the README"
    )

    # Hand-computed 10 bps case: flat then all-in on day 1 (held from day 2).
    hold = pd.DataFrame({"A": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=dates[:5])
    res = PythonEngine().run_backtest(prices.iloc[:5], hold, {"cost_bps": 10})
    day_rets = prices["A"].pct_change().iloc[:5].fillna(0.0)
    expected = day_rets.copy()
    expected.iloc[:2] = 0.0  # flat day 0; day-1 weight is *held* on day 2
    expected.iloc[2] -= 10.0 / 10_000.0  # entry turnover of 1.0 charged at 10 bps
    pd.testing.assert_series_equal(res.returns, expected, check_names=False)
