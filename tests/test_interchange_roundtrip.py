"""Interchange contract suite (AR-2, docs/components/16-tests.md): the Parquet boundary holds.

Three coverage groups, all offline on hand-built synthetic frames:
(a) write -> read identity for every one of the 7 kinds (dtypes included, date tz-aware UTC ns);
(b) SchemaError on every class of bad hand-off — unknown kind, missing column, wrong dtype,
    tz-naive dates — with rejected writes leaving *nothing* on disk and wrong-kind/foreign files
    failing loudly on read (a bad file must die at the boundary, not become NaNs downstream);
(c) to_wide/to_long are exact inverses on a ragged panel, and non-panel kinds are rejected.
"""

import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantforge.interchange import SCHEMAS, SchemaError, read_frame, to_long, to_wide, write_frame

_KINDS = sorted(SCHEMAS)

#: The per-ticker panel kinds (the only ones with a wide form) and their value column.
_PANEL_KINDS = {"prices": "close", "positions": "weight", "asset_returns": "ret"}
_NON_PANEL_KINDS = ["returns", "metrics", "weights", "frontier"]


def _utc_ns_dates(n: int = 3) -> pd.Series:
    # Pin to ns explicitly: the contract's canonical date dtype is timestamp[ns] UTC, and under
    # pandas 3.x a naturally-built date column defaults to 'us' resolution.
    return pd.Series(pd.date_range("2024-01-02", periods=n, tz="UTC")).astype("datetime64[ns, UTC]")


def _good_frame(kind: str) -> pd.DataFrame:
    """Hand-build a canonical conforming frame for each kind (independent of the module)."""
    cols = {}
    for field in SCHEMAS[kind]:
        if isinstance(field.type, pa.TimestampType):
            cols[field.name] = _utc_ns_dates()
        elif field.type == pa.float64():
            cols[field.name] = [0.1, -0.2, 0.3]
        else:
            cols[field.name] = ["AAPL", "MSFT", "GOOG"]
    return pd.DataFrame(cols)


# ---------------------------------------------------------------- round-trip (all 7 kinds)


@pytest.mark.parametrize("kind", _KINDS)
def test_round_trip_exact_for_every_kind(kind, tmp_path):
    df = _good_frame(kind)
    path = str(tmp_path / f"{kind}.parquet")
    write_frame(df, path, kind)
    out = read_frame(path, kind)
    pd.testing.assert_frame_equal(out, df)  # dtypes included
    assert isinstance(out.index, pd.RangeIndex)
    if "date" in out.columns:
        assert str(out["date"].dtype) == "datetime64[ns, UTC]"


def test_round_trip_empty_typed_frame(tmp_path):
    df = pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns, UTC]"),
            "ret": pd.Series([], dtype="float64"),
        }
    )
    path = str(tmp_path / "empty.parquet")
    write_frame(df, path, "returns")
    pd.testing.assert_frame_equal(read_frame(path, "returns"), df)


def test_write_frame_creates_parent_directories(tmp_path):
    path = str(tmp_path / "a" / "b" / "prices.parquet")
    write_frame(_good_frame("prices"), path, "prices")
    assert os.path.exists(path)


# ---------------------------------------------------------------- rejected writes (adversarial)


def test_invalid_write_raises_and_leaves_no_file(tmp_path):
    bad = pd.DataFrame({"date": [1, 2], "ret": [0.1, 0.2]})  # date is int64, not timestamp
    path = str(tmp_path / "bad.parquet")
    with pytest.raises(SchemaError):
        write_frame(bad, path, "returns")
    assert not os.path.exists(path)


def test_invalid_write_creates_no_parent_directories(tmp_path):
    # A rejected frame must touch nothing on disk — not even the mkdir -p side effect.
    bad = _good_frame("prices").drop(columns=["close"])
    nested = tmp_path / "never" / "created"
    with pytest.raises(SchemaError):
        write_frame(bad, str(nested / "prices.parquet"), "prices")
    assert not nested.exists()
    assert not (tmp_path / "never").exists()


def test_write_unknown_kind_raises_schema_error(tmp_path):
    path = str(tmp_path / "x.parquet")
    with pytest.raises(SchemaError):
        write_frame(_good_frame("prices"), path, "quotes")
    assert not os.path.exists(path)


def test_write_missing_column_rejected_naming_the_column(tmp_path):
    path = str(tmp_path / "no_ticker.parquet")
    with pytest.raises(SchemaError, match="ticker"):
        write_frame(_good_frame("prices").drop(columns=["ticker"]), path, "prices")
    assert not os.path.exists(path)


def test_write_wrong_value_dtype_rejected_leaves_no_file(tmp_path):
    # int64 close must not be silently upcast on the way to disk: the caller's frame is wrong
    # and the contract's fail-loudly rule applies at write time too.
    df = _good_frame("prices")
    df["close"] = [100, 101, 102]  # int64, not float64
    path = str(tmp_path / "int_close.parquet")
    with pytest.raises(SchemaError, match="close"):
        write_frame(df, path, "prices")
    assert not os.path.exists(path)


def test_write_tz_naive_dates_rejected_leaves_no_file(tmp_path):
    # Localizing on the caller's behalf would guess a timezone; the writer must refuse instead.
    df = _good_frame("returns")
    df["date"] = df["date"].dt.tz_localize(None)
    path = str(tmp_path / "naive_write.parquet")
    with pytest.raises(SchemaError, match="date"):
        write_frame(df, path, "returns")
    assert not os.path.exists(path)


# ---------------------------------------------------------------- rejected reads (adversarial)


def test_read_frame_cross_kind_raises_schema_error(tmp_path):
    # A file written under one kind's schema must not be readable as another kind.
    path = str(tmp_path / "weights.parquet")
    write_frame(_good_frame("weights"), path, "weights")
    with pytest.raises(SchemaError):
        read_frame(path, "prices")


def test_read_frame_wrong_value_dtype_in_file_rejected(tmp_path):
    # Foreign writer stored `close` as int64: must fail loudly, not silently coerce.
    table = pa.table(
        {
            "date": pa.array([pd.Timestamp("2024-01-02", tz="UTC")], pa.timestamp("ns", tz="UTC")),
            "ticker": pa.array(["AAPL"], pa.string()),
            "close": pa.array([100], pa.int64()),
        }
    )
    path = str(tmp_path / "int_close.parquet")
    pq.write_table(table, path)
    with pytest.raises(SchemaError, match="close"):
        read_frame(path, "prices")


def test_read_frame_extra_column_in_file_rejected(tmp_path):
    table = pa.table(
        {
            "date": pa.array([pd.Timestamp("2024-01-02", tz="UTC")], pa.timestamp("ns", tz="UTC")),
            "ticker": pa.array(["AAPL"], pa.string()),
            "close": pa.array([100.0], pa.float64()),
            "volume": pa.array([1.0], pa.float64()),
        }
    )
    path = str(tmp_path / "extra.parquet")
    pq.write_table(table, path)
    with pytest.raises(SchemaError, match="volume"):
        read_frame(path, "prices")


def test_read_pandas_file_with_named_index_rejected(tmp_path):
    # A named index materializes as a physical extra column in Parquet — the contract is
    # columns-only, so this must be rejected at the boundary, not silently absorbed.
    df = pd.DataFrame({"ticker": ["AAPL"], "weight": [1.0]}, index=pd.Index([7], name="rowid"))
    path = str(tmp_path / "named_idx.parquet")
    df.to_parquet(path)
    with pytest.raises(SchemaError):
        read_frame(path, "weights")


# ---------------------------------------------------------------- date normalization on read


def test_read_tz_naive_file_localized_to_utc_same_wall_clock(tmp_path):
    # Adversarial hand-computed expectation: naive 2024-01-02 must become exactly
    # 2024-01-02 00:00:00+00:00 (localized, i.e. same wall clock — never shifted).
    table = pa.table(
        {
            "date": pa.array(
                [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], pa.timestamp("ns")
            ),
            "ret": pa.array([0.01, -0.02], pa.float64()),
        }
    )
    path = str(tmp_path / "naive.parquet")
    pq.write_table(table, path)
    out = read_frame(path, "returns")
    assert str(out["date"].dtype) == "datetime64[ns, UTC]"
    assert list(out["date"]) == [
        pd.Timestamp("2024-01-02", tz="UTC"),
        pd.Timestamp("2024-01-03", tz="UTC"),
    ]


def test_read_non_utc_aware_file_converted_losslessly(tmp_path):
    # 19:00 US/Eastern on Jan 2 is exactly midnight UTC on Jan 3 — conversion, not relabeling.
    table = pa.table(
        {
            "date": pa.array(
                [pd.Timestamp("2024-01-02 19:00", tz="US/Eastern")],
                pa.timestamp("ns", tz="US/Eastern"),
            ),
            "ret": pa.array([0.01], pa.float64()),
        }
    )
    path = str(tmp_path / "eastern.parquet")
    pq.write_table(table, path)
    out = read_frame(path, "returns")
    assert str(out["date"].dtype) == "datetime64[ns, UTC]"
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-03 00:00", tz="UTC")


def test_read_us_precision_file_normalized_to_ns(tmp_path):
    # Foreign writers (and pandas 3.x defaults) may store microsecond timestamps; the contract
    # pins the in-memory result to ns.
    table = pa.table(
        {
            "date": pa.array([pd.Timestamp("2024-01-02", tz="UTC")], pa.timestamp("us", tz="UTC")),
            "ret": pa.array([0.01], pa.float64()),
        }
    )
    path = str(tmp_path / "usprec.parquet")
    pq.write_table(table, path)
    out = read_frame(path, "returns")
    assert str(out["date"].dtype) == "datetime64[ns, UTC]"
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-02", tz="UTC")


def test_read_plain_pandas_written_file_gets_range_index(tmp_path):
    # A conforming file written by pandas directly (not write_frame) must read back with a
    # default RangeIndex — stored pandas index metadata is ignored.
    df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "weight": [0.6, 0.4]})
    path = str(tmp_path / "plain.parquet")
    df.to_parquet(path)
    out = read_frame(path, "weights")
    assert isinstance(out.index, pd.RangeIndex)
    assert out["weight"].tolist() == [0.6, 0.4]


# ---------------------------------------------------------------- on-disk contract


@pytest.mark.parametrize("kind", _KINDS)
def test_written_file_physical_schema_matches_contract(kind, tmp_path):
    # The Parquet file itself (what R/KNIME see) must carry exactly SCHEMAS[kind]'s types,
    # regardless of which pandas dtypes produced it.
    path = str(tmp_path / f"{kind}.parquet")
    write_frame(_good_frame(kind), path, kind)
    stored = pq.read_table(path).schema
    assert [(f.name, f.type) for f in stored] == [(f.name, f.type) for f in SCHEMAS[kind]]


# ---------------------------------------------------------------- wide <-> long inverse


def _ragged_panel(kind: str) -> pd.DataFrame:
    """Hand-built 3-ticker long panel, ragged at both ends: CCC starts a day late (no row on
    the first date) and BBB stops a day early (no row on the last date). Rows are already in
    canonical (date, ticker) order so the frame can be compared directly after a round-trip.
    """
    d1, d2, d3 = (pd.Timestamp(d, tz="UTC") for d in ("2024-01-02", "2024-01-03", "2024-01-04"))
    return pd.DataFrame(
        {
            "date": [d1, d1, d2, d2, d2, d3, d3],
            "ticker": ["AAA", "BBB", "AAA", "BBB", "CCC", "AAA", "CCC"],
            _PANEL_KINDS[kind]: [1.0, -2.0, 1.1, -2.1, 3.0, 1.2, 3.1],
        }
    )


@pytest.mark.parametrize("kind", sorted(_PANEL_KINDS))
def test_to_long_inverts_to_wide_on_ragged_panel(kind):
    df = _ragged_panel(kind)
    wide = to_wide(df, kind)
    # The two missing (date, ticker) pairs surface as exactly two NaN cells in wide form...
    assert wide.shape == (3, 3)
    assert int(wide.isna().sum().sum()) == 2
    # ...and vanish again on the way back: absent rows, not NaN rows, and nothing else changes.
    back = to_long(wide, kind)
    pd.testing.assert_frame_equal(back, df)  # dtypes included
    assert not back[_PANEL_KINDS[kind]].isna().any()


@pytest.mark.parametrize("kind", _NON_PANEL_KINDS + ["quotes"])
def test_to_wide_rejects_non_pivotable_kind(kind):
    with pytest.raises(SchemaError, match=repr(kind)):
        to_wide(_ragged_panel("prices"), kind)


@pytest.mark.parametrize("kind", _NON_PANEL_KINDS + ["quotes"])
def test_to_long_rejects_non_pivotable_kind(kind):
    wide = to_wide(_ragged_panel("prices"), "prices")
    with pytest.raises(SchemaError, match=repr(kind)):
        to_long(wide, kind)
