"""Verification tests for interchange to_wide/to_long (long <-> wide panel pivot).

Proves the milestone: to_wide pivots a ragged long panel to a date x ticker matrix (UTC
DatetimeIndex named 'date', NaN where a late-starting ticker has no data); to_long inverts it
exactly (absent rows — not NaN rows — represent missing data, output re-validates); both raise
SchemaError for every non-panel kind; malformed inputs on either side fail loudly. Offline,
synthetic data only.
"""

import numpy as np
import pandas as pd
import pytest

from quantforge.interchange import (
    SchemaError,
    read_frame,
    to_long,
    to_wide,
    validate_frame,
    write_frame,
)

#: Only these kinds are date x ticker panels; everything else must be rejected.
_PANEL_KINDS = {"prices": "close", "positions": "weight", "asset_returns": "ret"}
_NON_PANEL_KINDS = ["returns", "metrics", "weights", "frontier"]

_D1, _D2, _D3 = (pd.Timestamp(d, tz="UTC") for d in ("2024-01-02", "2024-01-03", "2024-01-04"))


def _ragged_prices() -> pd.DataFrame:
    """3-ticker long prices panel where MSFT starts one day late (no 2024-01-02 row)."""
    return pd.DataFrame(
        {
            "date": [_D1, _D1, _D2, _D2, _D2, _D3, _D3, _D3],
            "ticker": ["AAPL", "GOOG", "AAPL", "GOOG", "MSFT", "AAPL", "GOOG", "MSFT"],
            "close": [10.0, 20.0, 11.0, 21.0, 30.0, 12.0, 22.0, 31.0],
        }
    )


def _ragged_panel(kind: str) -> pd.DataFrame:
    """The same ragged panel re-labeled for any per-ticker kind."""
    return _ragged_prices().rename(columns={"close": _PANEL_KINDS[kind]})


# ---------------------------------------------------------------- to_wide shape and values


def test_to_wide_ragged_prices_shape_and_hand_computed_cells():
    wide = to_wide(_ragged_prices(), "prices")

    # Index: tz-aware UTC DatetimeIndex named 'date'; columns: the tickers, axis named 'ticker'.
    assert isinstance(wide.index, pd.DatetimeIndex)
    assert str(wide.index.tz) == "UTC"
    assert wide.index.name == "date"
    assert list(wide.index) == [_D1, _D2, _D3]
    assert list(wide.columns) == ["AAPL", "GOOG", "MSFT"]
    assert wide.columns.name == "ticker"

    # Hand-computed cells: every value in its (date, ticker) slot.
    assert wide.loc[_D1, "AAPL"] == 10.0
    assert wide.loc[_D2, "GOOG"] == 21.0
    assert wide.loc[_D3, "MSFT"] == 31.0

    # The late starter is NaN exactly where it has no data — and nowhere else.
    assert np.isnan(wide.loc[_D1, "MSFT"])
    assert int(wide.isna().sum().sum()) == 1


@pytest.mark.parametrize("kind", sorted(_PANEL_KINDS))
def test_to_wide_works_for_every_panel_kind(kind):
    wide = to_wide(_ragged_panel(kind), kind)
    assert wide.shape == (3, 3)
    assert np.isnan(wide.loc[_D1, "MSFT"])


# ---------------------------------------------------------------- round-trip property


@pytest.mark.parametrize("kind", sorted(_PANEL_KINDS))
def test_to_long_to_wide_round_trip_identical_on_ragged_panel(kind):
    df = _ragged_panel(kind)
    back = to_long(to_wide(df, kind), kind)
    expected = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(back, expected)  # dtypes included


def test_to_long_output_validates_and_represents_missing_as_absent_rows():
    wide = to_wide(_ragged_prices(), "prices")
    long = to_long(wide, "prices")
    validate_frame(long, "prices")  # must pass silently
    # 8 rows, not 9: the MSFT NaN cell became an *absent* row, and no NaN survives anywhere.
    assert len(long) == 8
    assert not long["close"].isna().any()
    assert long[(long["ticker"] == "MSFT") & (long["date"] == _D1)].empty


def test_to_long_output_is_writable_via_write_frame(tmp_path):
    # The stated purpose of to_long: run it before every write_frame of wide data. Prove the
    # full path wide -> long -> Parquet -> read_frame keeps the contract intact.
    long = to_long(to_wide(_ragged_prices(), "prices"), "prices")
    path = str(tmp_path / "prices.parquet")
    write_frame(long, path, "prices")
    out = read_frame(path, "prices")
    assert out["close"].tolist() == long["close"].tolist()
    assert out["ticker"].tolist() == long["ticker"].tolist()


# ---------------------------------------------------------------- non-panel kinds rejected


@pytest.mark.parametrize("kind", _NON_PANEL_KINDS + ["quotes"])
def test_to_wide_rejects_non_panel_kind(kind):
    with pytest.raises(SchemaError, match=repr(kind)):
        to_wide(_ragged_prices(), kind)


@pytest.mark.parametrize("kind", _NON_PANEL_KINDS + ["quotes"])
def test_to_long_rejects_non_panel_kind(kind):
    wide = to_wide(_ragged_prices(), "prices")
    with pytest.raises(SchemaError, match=repr(kind)):
        to_long(wide, kind)


# ---------------------------------------------------------------- malformed inputs (adversarial)


def test_to_wide_validates_input_tz_naive_dates_rejected():
    df = _ragged_prices()
    df["date"] = df["date"].dt.tz_localize(None)
    with pytest.raises(SchemaError, match="date"):
        to_wide(df, "prices")


def test_to_wide_validates_input_misordered_columns_rejected():
    df = _ragged_prices()[["ticker", "date", "close"]]
    with pytest.raises(SchemaError):
        to_wide(df, "prices")


def test_to_wide_duplicate_date_ticker_rejected_not_silently_collapsed():
    # Two AAPL rows on the same date: a pivot cannot represent this; keeping either value
    # would hide an upstream data bug, so it must fail loudly.
    df = _ragged_prices()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="AAPL"):
        to_wide(dup, "prices")


def test_to_long_rejects_non_datetime_index():
    wide = pd.DataFrame({"AAPL": [1.0, 2.0]})  # default RangeIndex
    with pytest.raises(SchemaError, match="DatetimeIndex"):
        to_long(wide, "prices")


def test_to_long_rejects_tz_naive_index():
    wide = pd.DataFrame(
        {"AAPL": [1.0, 2.0]}, index=pd.date_range("2024-01-02", periods=2)  # naive
    )
    with pytest.raises(SchemaError, match="naive"):
        to_long(wide, "prices")


def test_to_long_rejects_non_numeric_column():
    wide = pd.DataFrame(
        {"AAPL": ["a", "b"]}, index=pd.date_range("2024-01-02", periods=2, tz="UTC")
    )
    with pytest.raises(SchemaError, match="AAPL"):
        to_long(wide, "prices")


def test_to_long_rejects_non_string_column_labels_with_schema_error():
    # Integer column labels would produce a non-string ticker column — must surface as
    # SchemaError (the boundary's one exception type), not a bare pandas error.
    wide = pd.DataFrame(
        {1: [10.0], 2: [20.0]}, index=pd.date_range("2024-01-02", periods=1, tz="UTC")
    )
    with pytest.raises(SchemaError):
        to_long(wide, "prices")


# ---------------------------------------------------------------- canonicalization


def test_to_long_canonicalizes_messy_but_equivalent_wide_frame():
    # Same panel, adversarially scrambled: unnamed axes, shuffled column order, reversed rows,
    # int dtype, and a US/Eastern (aware, non-UTC) index. 19:00 Eastern on Jan 1 IS midnight
    # UTC on Jan 2 — hand-computed conversion, not a relabel.
    eastern = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 19:00", tz="US/Eastern"),
         pd.Timestamp("2024-01-01 19:00", tz="US/Eastern")]
    )
    messy = pd.DataFrame({"GOOG": [21, 20], "AAPL": [11, 10]}, index=eastern)
    assert messy.index.name is None and messy.columns.name is None

    long = to_long(messy, "prices")
    expected = pd.DataFrame(
        {
            "date": [_D1, _D1, _D2, _D2],
            "ticker": ["AAPL", "GOOG", "AAPL", "GOOG"],
            "close": [10.0, 20.0, 11.0, 21.0],
        }
    )
    pd.testing.assert_frame_equal(long, expected, check_dtype=False)
    assert long["close"].dtype == "float64"  # int cells cast to the contract dtype
    assert isinstance(long.index, pd.RangeIndex)


def test_to_long_on_empty_wide_frame_returns_empty_valid_long_frame():
    empty = to_wide(_ragged_prices(), "prices").iloc[0:0]
    out = to_long(empty, "prices")
    assert len(out) == 0
    validate_frame(out, "prices")
