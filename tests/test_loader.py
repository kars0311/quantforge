"""Loader suite (docs/components/16-tests.md row FR-1/RG-4): cache = no network; splits sane.

Runs fully offline in CI — yfinance is either poisoned (import raises) or replaced by a canned
fake module, and every cache lives in ``tmp_path``.

Rigor these lock in (docs/components/02-data-loader.md, docs/components/16-tests.md):
- FR-1: one canonical cache file (``{cache_dir}/prices.parquet``) holds the whole universe;
  every request — any ticker subset, any sub-range — is sliced from it.
- Offline guarantee: a cache HIT never imports yfinance (proved by making the import raise),
  so after the first run the pipeline works on machines with no network and no yfinance.
- Cache MISS: exactly one full-universe, full-range download (auto_adjust=True), normalized
  (MultiIndex flattened, tz-naive dates localized to UTC, NaN closes dropped so missing data
  is an absent row) and written once; the very next call is a pure hit.
- No silent refetch: the cache is authoritative even when its contents are "wrong" (sentinel
  price comes back verbatim), and a malformed cache fails loudly instead of being re-downloaded.
- Determinism: identical frames on repeated calls, given the cache.
- RG-4: get_split_bounds() == SPLITS, three chronological non-overlapping ranges spanning
  START..END — plus UNIVERSE size/uniqueness sanity. (Deeper constants pinning — the exact
  doc ticker list, boundary strings, docstring caveats — lives in test_loader_constants.py;
  the checks here keep this FR-1/RG-4 suite self-contained per the test inventory.)
"""

import sys
import types

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantforge import interchange
from quantforge.data.loader import (
    END,
    SPLITS,
    START,
    UNIVERSE,
    get_split_bounds,
    load_prices,
)

# ---------------------------------------------------------------- helpers / fixtures

# Hand-built long prices frame spanning several years so sub-range slices are meaningful.
# 2015-12-31 and 2016-01-04 are both present to prove the [start, end] slice is inclusive
# of its endpoint but excludes the next trading day.
_SEED_ROWS = [
    ("2014-06-02", "AAPL", 90.0),
    ("2014-06-02", "MSFT", 40.0),
    ("2015-01-02", "AAPL", 100.0),
    ("2015-01-02", "MSFT", 45.0),
    ("2015-06-15", "AAPL", 120.0),
    ("2015-12-31", "AAPL", 105.0),
    ("2015-12-31", "MSFT", 55.0),
    ("2016-01-04", "AAPL", 101.0),
    ("2016-01-04", "MSFT", 54.0),
]


def _seed_frame() -> pd.DataFrame:
    df = pd.DataFrame(_SEED_ROWS, columns=["date", "ticker", "close"])
    # Pin ns explicitly: pandas 3 defaults to 'us', but read_frame normalizes the file to
    # timestamp[ns], and this frame doubles as the expected round-trip result.
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC").astype("datetime64[ns, UTC]")
    df["close"] = df["close"].astype("float64")
    return df


@pytest.fixture
def seeded_cache(tmp_path):
    """A cache_dir pre-seeded with a valid prices.parquet (the offline fixture, no network)."""
    interchange.write_frame(_seed_frame(), str(tmp_path / "prices.parquet"), "prices")
    return tmp_path


@pytest.fixture
def no_yfinance(monkeypatch):
    """Make ``import yfinance`` raise: any code path that reaches for the network dies loudly."""
    monkeypatch.setitem(sys.modules, "yfinance", None)  # None => import raises ImportError


def _fake_yfinance(monkeypatch, raw: pd.DataFrame):
    """Install a fake yfinance module whose download() records calls and returns ``raw``."""
    calls: list[dict] = []

    def download(tickers, **kwargs):
        calls.append({"tickers": tickers, **kwargs})
        return raw.copy()

    fake = types.ModuleType("yfinance")
    fake.download = download
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return calls


def _canned_download() -> pd.DataFrame:
    """A canned multi-ticker, multi-field yfinance-shaped frame with the real quirks.

    - MultiIndex columns, level 0 = price field, level 1 = ticker (group_by="column" layout);
    - tz-NAIVE daily index (yfinance returns naive dates);
    - META NaN on the first two days (pre-IPO — META listed 2012-05-18);
    - MSFT NaN on 2012-05-18 (a hole standing in for a holiday/halt).
    """
    dates = pd.to_datetime(["2012-05-16", "2012-05-17", "2012-05-18", "2012-05-21"])
    tickers = ["AAPL", "META", "MSFT"]
    close = pd.DataFrame(
        {
            "AAPL": [20.0, 21.0, 22.0, 23.0],
            "META": [float("nan"), float("nan"), 38.23, 34.03],
            "MSFT": [29.0, 29.5, float("nan"), 29.75],
        },
        index=dates,
    )
    fields = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        for t in tickers:
            # Non-Close fields get junk values: the loader must keep ONLY the adjusted close.
            fields[(field, t)] = close[t] if field == "Close" else close[t] * 1000.0
    raw = pd.DataFrame(fields, index=dates)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns, names=["Price", "Ticker"])
    return raw


# ---------------------------------------------------------------- cache HIT: pure offline


def test_cache_hit_never_imports_yfinance_and_validates(seeded_cache, no_yfinance):
    """The done-when core: with a seeded cache, load_prices works with yfinance unimportable."""
    out = load_prices(cache_dir=str(seeded_cache))
    interchange.validate_frame(out, "prices")
    # Full default request returns the entire seeded panel, RangeIndex from 0.
    pd.testing.assert_frame_equal(out, _seed_frame())


def test_cache_hit_subset_sliced_from_single_file(seeded_cache, no_yfinance):
    out = load_prices(["AAPL"], start="2015-01-01", end="2015-12-31", cache_dir=str(seeded_cache))
    interchange.validate_frame(out, "prices")
    assert set(out["ticker"]) == {"AAPL"}
    assert out["date"].min() >= pd.Timestamp("2015-01-01", tz="UTC")
    assert out["date"].max() <= pd.Timestamp("2015-12-31", tz="UTC")
    # Inclusive endpoint: 2015-12-31 is in range; 2016-01-04 and 2014 rows are not.
    assert pd.Timestamp("2015-12-31", tz="UTC") in set(out["date"])
    # Hand-computed expectation: exactly the three 2015 AAPL rows, in date order.
    assert list(out["close"]) == [100.0, 120.0, 105.0]
    assert list(out.index) == [0, 1, 2]  # canonical RangeIndex regardless of slice position
    # Still exactly ONE cache file — subsetting must not spawn per-request caches.
    assert [p.name for p in seeded_cache.iterdir()] == ["prices.parquet"]


def test_cache_hit_deterministic_across_calls(seeded_cache, no_yfinance):
    a = load_prices(["MSFT"], cache_dir=str(seeded_cache))
    b = load_prices(["MSFT"], cache_dir=str(seeded_cache))
    pd.testing.assert_frame_equal(a, b)


def test_no_silent_refetch_cache_is_authoritative(tmp_path, no_yfinance):
    """Adversarial: plant a sentinel price no real feed would serve. If load_prices ever
    refetched behind our back, the sentinel would vanish; it must come back verbatim."""
    df = _seed_frame()
    df.loc[df.index[0], "close"] = 123456789.0
    interchange.write_frame(df, str(tmp_path / "prices.parquet"), "prices")

    out = load_prices(["AAPL"], start="2014-06-02", end="2014-06-02", cache_dir=str(tmp_path))
    assert list(out["close"]) == [123456789.0]


def test_malformed_cache_fails_loudly_not_refetched(tmp_path, monkeypatch):
    """A wrong-schema cache file must raise SchemaError at the boundary — never be silently
    'repaired' by a new download (which would change data under every previous result)."""
    calls = _fake_yfinance(monkeypatch, _canned_download())
    table = pa.table({"foo": [1.0, 2.0], "bar": ["a", "b"]})
    pq.write_table(table, tmp_path / "prices.parquet")

    with pytest.raises(interchange.SchemaError):
        load_prices(cache_dir=str(tmp_path))
    assert calls == []  # the network was never consulted


# ---------------------------------------------------------------- cache MISS: one download


def test_cache_miss_downloads_full_universe_once_and_writes_valid_parquet(tmp_path, monkeypatch):
    calls = _fake_yfinance(monkeypatch, _canned_download())

    out = load_prices(cache_dir=str(tmp_path))

    # Exactly one download, for the FULL universe and FULL range regardless of the request,
    # with auto_adjust so "Close" is the adjusted close. end is exclusive in yfinance, so the
    # loader must pass END + 1 day to keep the promised range *through* END.
    assert len(calls) == 1
    call = calls[0]
    assert call["tickers"] == UNIVERSE
    assert call["auto_adjust"] is True
    assert pd.Timestamp(call["start"]) == pd.Timestamp(START)
    assert pd.Timestamp(call["end"]) == pd.Timestamp(END) + pd.Timedelta(days=1)

    # Exactly one schema-valid file on disk, and the returned frame equals its contents.
    assert [p.name for p in tmp_path.iterdir()] == ["prices.parquet"]
    on_disk = interchange.read_frame(str(tmp_path / "prices.parquet"), "prices")
    pd.testing.assert_frame_equal(out, on_disk)

    # Normalization contract: long shape, tz-aware UTC ns dates, float64, no NaN closes.
    interchange.validate_frame(out, "prices")
    assert str(out["date"].dtype) == "datetime64[ns, UTC]"
    assert not out["close"].isna().any()
    # Sorted (date-major, then ticker) and canonical.
    assert out.equals(out.sort_values(["date", "ticker"]).reset_index(drop=True))


def test_cache_miss_normalization_hand_computed(tmp_path, monkeypatch):
    """Hand-computed expectation for the canned download: 4 dates x 3 tickers = 12 cells,
    minus 2 pre-IPO META NaNs and 1 MSFT hole = 9 rows; missing data is an ABSENT row."""
    _fake_yfinance(monkeypatch, _canned_download())
    out = load_prices(cache_dir=str(tmp_path))

    assert len(out) == 9
    # Pre-IPO META rows must not exist (neither as NaN rows nor zeros).
    meta = out[out["ticker"] == "META"]
    assert meta["date"].min() == pd.Timestamp("2012-05-18", tz="UTC")
    assert list(meta["close"]) == [38.23, 34.03]
    # The holiday-style hole is an absent row too.
    msft_dates = set(out.loc[out["ticker"] == "MSFT", "date"])
    assert pd.Timestamp("2012-05-18", tz="UTC") not in msft_dates
    # Only the Close field survived — the junk (x1000) values from other fields never leak in.
    assert out["close"].max() < 1000.0
    # Spot-check one exact value against the canned input.
    aapl_first = out[
        (out["ticker"] == "AAPL") & (out["date"] == pd.Timestamp("2012-05-16", tz="UTC"))
    ]
    assert list(aapl_first["close"]) == [20.0]


def test_after_miss_next_call_is_pure_hit(tmp_path, monkeypatch):
    """Adversarial offline check: populate via one mocked download, then make yfinance
    unimportable — the second call must succeed anyway and return the identical frame."""
    calls = _fake_yfinance(monkeypatch, _canned_download())
    first = load_prices(cache_dir=str(tmp_path))
    assert len(calls) == 1

    monkeypatch.setitem(sys.modules, "yfinance", None)  # now importing yfinance raises
    second = load_prices(cache_dir=str(tmp_path))
    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 1  # and the fake was never called again


def test_miss_request_subset_still_caches_whole_download(tmp_path, monkeypatch):
    """Asking for one ticker on a cold cache must still download and cache the FULL panel —
    a partial cache would silently poison every later request."""
    calls = _fake_yfinance(monkeypatch, _canned_download())
    out = load_prices(["META"], cache_dir=str(tmp_path))

    assert calls[0]["tickers"] == UNIVERSE  # not ["META"]
    assert set(out["ticker"]) == {"META"}
    # The file still holds all downloaded tickers, ready for other callers offline.
    on_disk = interchange.read_frame(str(tmp_path / "prices.parquet"), "prices")
    assert set(on_disk["ticker"]) == {"AAPL", "META", "MSFT"}


# ---------------------------------------------------------------- split bounds sane (RG-4)


def test_split_bounds_sane():
    """RG-4: three named splits, chronological, non-overlapping, jointly spanning START..END.

    Guardrails and later tests import these bounds rather than redefining them; this check
    guarantees the imported values are internally coherent (an overlap would leak train data
    into the holdout, a gap would drop trading days on the floor).
    """
    bounds = get_split_bounds()
    assert bounds == SPLITS
    assert list(bounds) == ["train", "validation", "holdout"]

    d = pd.Timestamp  # ISO strings -> comparable timestamps; no tz needed for pure ordering
    for lo, hi in bounds.values():
        assert d(lo) <= d(hi)  # each range internally ordered

    # Chronological and non-overlapping: each split starts the calendar day after the
    # previous one ends (adjacent, so there is also no gap between them).
    pairs = list(bounds.values())
    for (_, prev_hi), (nxt_lo, _) in zip(pairs, pairs[1:]):
        assert d(nxt_lo) - d(prev_hi) == pd.Timedelta(days=1)

    # Together the splits span exactly START..END.
    assert pairs[0][0] == START
    assert pairs[-1][1] == END


# ---------------------------------------------------------------- constants sanity


def test_universe_constants_sanity():
    """The frozen universe is ~28-30 unique upper-case US-listed tickers (FR-1)."""
    assert len(UNIVERSE) == 30
    assert len(set(UNIVERSE)) == len(UNIVERSE)  # no duplicate names
    for t in UNIVERSE:
        assert isinstance(t, str) and t and t == t.upper()
