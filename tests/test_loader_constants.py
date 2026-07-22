"""Verifier tests: loader constants (UNIVERSE / START / END / SPLITS / get_split_bounds).

Rigor these lock in (docs/components/02-data-loader.md):
- RG-3: the survivorship-bias caveat stays in the module docstring — deleting it fails CI.
- RG-4: SPLITS is the single source of truth; this test imports it, never redefines dates,
  and pins the exact boundary strings so a silent edit (which would invalidate every past
  result and un-freeze the holdout) is caught.
- Reproducibility: END is a fixed cutoff, never "today" — a moving end date would silently
  change every metric between runs.

All offline: constants only, no network. load_prices is implemented (later milestone), but its
yfinance import is lazy inside the cache-miss branch — the test below proves that with no cache
and yfinance unimportable it fails fast instead of silently reaching for the network.
"""

import datetime as dt
import sys

import pytest

import quantforge.data.loader as loader
from quantforge.data.loader import END, SPLITS, START, UNIVERSE, get_split_bounds

# The frozen list exactly as enumerated in docs/components/02-data-loader.md ("Fixed
# universe & dates"). Restated here on purpose: the test must fail if the module list
# drifts from the doc, so it cannot just re-import the thing it checks.
_DOC_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "CRM", "ADBE", "ORCL", "AVGO",
    "TSM", "ASML", "SAP", "TM", "NVO", "SONY",
    "JPM", "GS", "V",
    "JNJ", "UNH", "PFE",
    "PG", "KO", "MCD", "WMT", "HD",
    "XOM", "CVX", "CAT",
]


# ---------------------------------------------------------------- universe

def test_universe_matches_doc_list_exactly():
    assert UNIVERSE == _DOC_UNIVERSE


def test_universe_tickers_unique_uppercase_strings():
    assert len(UNIVERSE) == len(set(UNIVERSE))
    for t in UNIVERSE:
        assert isinstance(t, str)
        assert t == t.upper() and t.strip() == t and t != ""


def test_universe_excludes_direct_foreign_listings():
    # ADR policy (doc + docstring): US-listed ADRs only — no exchange-suffixed listings.
    assert not any("." in t for t in UNIVERSE)
    assert "2330.TW" not in UNIVERSE and "TSM" in UNIVERSE


# ---------------------------------------------------------------- dates & splits

def test_start_end_exact_fixed_strings():
    assert START == "2010-01-01"
    assert END == "2026-06-30"
    # END must be a frozen cutoff, not computed from the clock.
    assert END != dt.date.today().isoformat()


def test_splits_exact_values():
    assert SPLITS == {
        "train": ("2010-01-01", "2019-12-31"),
        "validation": ("2020-01-01", "2022-12-31"),
        "holdout": ("2023-01-01", "2026-06-30"),
    }
    assert list(SPLITS) == ["train", "validation", "holdout"]  # chronological key order
    for bounds in SPLITS.values():
        assert isinstance(bounds, tuple) and len(bounds) == 2


def test_splits_chronological_contiguous_and_span_start_end():
    """Adversarial: hand-check the calendar math, don't trust string ordering alone.

    Each split must be internally ordered, consecutive splits must be adjacent calendar
    days (no gap a trade could fall into, no overlap that would leak train data into the
    holdout), and together they must cover exactly START..END.
    """
    d = dt.date.fromisoformat
    train, val, hold = SPLITS["train"], SPLITS["validation"], SPLITS["holdout"]

    for lo, hi in (train, val, hold):
        assert d(lo) <= d(hi)

    one_day = dt.timedelta(days=1)
    assert d(val[0]) - d(train[1]) == one_day  # 2019-12-31 -> 2020-01-01
    assert d(hold[0]) - d(val[1]) == one_day   # 2022-12-31 -> 2023-01-01

    assert train[0] == START
    assert hold[1] == END


def test_get_split_bounds_is_the_single_source_of_truth():
    assert get_split_bounds() == SPLITS
    # RG-4: callers get the module constant itself, not a diverging copy.
    assert get_split_bounds() is SPLITS


# ---------------------------------------------------------------- docstring rigor (RG-3)

def test_module_docstring_keeps_bias_caveats():
    doc = loader.__doc__ or ""
    assert "SURVIVORSHIP" in doc.upper()
    assert "META" in doc          # post-2010 IPOs enter when data begins
    assert "ADR" in doc.upper()   # direct foreign listings excluded in favor of ADRs


# ---------------------------------------------------------------- loader stays offline

def test_load_prices_cache_miss_fails_fast_without_yfinance(monkeypatch, tmp_path):
    """Offline guarantee (successor to the week-1 stub check): the cache-miss branch is the
    ONLY place yfinance is imported, so with no cache file and yfinance made unimportable,
    load_prices must raise ImportError immediately — never silently hit the network."""
    monkeypatch.setitem(sys.modules, "yfinance", None)  # None => `import yfinance` raises
    with pytest.raises(ImportError):
        loader.load_prices(cache_dir=str(tmp_path))
