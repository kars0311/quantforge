"""Independent verifier tests for the interchange contract (AR-2 milestone lock-in).

Written adversarially, separate from the builder's suites: an end-to-end wide -> long ->
Parquet -> long -> wide loop with hand-computed cell values (the exact path engine output
takes to disk and back), plus write-side rejections proving the no-partial-artifact rule
holds for dtype variants the other suites don't push through ``write_frame``.
"""

import os

import pandas as pd
import pytest

from quantforge.interchange import SchemaError, read_frame, to_long, to_wide, write_frame


def _ragged_wide_prices() -> pd.DataFrame:
    """Hand-built wide prices: 3 dates x 3 tickers with two deliberate holes.

    CCC has no price on the first date (starts late) and BBB none on the last
    (stops early) — the raggedness a real fixed universe produces.
    """
    # Pin ns explicitly: pandas 3.x builds a fresh DatetimeIndex at 'us' resolution, but the
    # contract's canonical in-memory date dtype (what read_frame hands back) is timestamp[ns].
    dates = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="UTC") for d in ("2024-01-02", "2024-01-03", "2024-01-04")],
        name="date",
    ).astype("datetime64[ns, UTC]")
    return pd.DataFrame(
        {
            "AAA": [10.0, 11.0, 12.0],
            "BBB": [20.0, 21.0, float("nan")],
            "CCC": [float("nan"), 30.0, 31.0],
        },
        index=dates,
    ).rename_axis(columns="ticker")


def test_wide_long_parquet_full_loop_hand_computed(tmp_path):
    # The full hand-off an engine result actually makes: wide -> to_long -> write_frame ->
    # read_frame -> to_wide must reproduce the original matrix cell-for-cell, holes included.
    wide = _ragged_wide_prices()
    long = to_long(wide, "prices")

    # Hand-computed: 9 cells minus 2 holes = 7 rows, and the holes are *absent*, not NaN.
    assert len(long) == 7
    assert not long["close"].isna().any()
    jan4 = pd.Timestamp("2024-01-04", tz="UTC")
    assert set(long.loc[long["date"] == jan4, "ticker"]) == {"AAA", "CCC"}

    path = str(tmp_path / "prices.parquet")
    write_frame(long, path, "prices")
    back = to_wide(read_frame(path, "prices"), "prices")

    pd.testing.assert_frame_equal(back, wide)
    # Spot-check two cells against the hand-built values (not just frame equality machinery).
    assert back.loc[jan4, "CCC"] == 31.0
    assert pd.isna(back.loc[jan4, "BBB"])


def test_write_float32_rejected_leaves_no_file(tmp_path):
    # float32 is numeric and would upcast silently — the contract demands exact float64,
    # and a rejected write must leave the filesystem untouched.
    df = pd.DataFrame(
        {
            "date": pd.Series(pd.date_range("2024-01-02", periods=2, tz="UTC")).astype(
                "datetime64[ns, UTC]"
            ),
            "ret": pd.Series([0.01, -0.02], dtype="float32"),
        }
    )
    path = str(tmp_path / "f32.parquet")
    with pytest.raises(SchemaError, match="ret"):
        write_frame(df, path, "returns")
    assert not os.path.exists(path)


def test_write_non_utc_aware_dates_rejected_leaves_no_file(tmp_path):
    # tz-aware but wrong zone: conversion would be lossless, but the writer must not do the
    # caller's normalization — the contract is UTC-in, and rejection keeps the boundary strict.
    df = pd.DataFrame(
        {
            "date": pd.Series(
                pd.date_range("2024-01-02", periods=2, tz="US/Eastern")
            ).astype("datetime64[ns, US/Eastern]"),
            "ret": [0.01, -0.02],
        }
    )
    path = str(tmp_path / "eastern.parquet")
    with pytest.raises(SchemaError, match="date"):
        write_frame(df, path, "returns")
    assert not os.path.exists(path)


def test_write_misordered_columns_rejected_leaves_no_file(tmp_path):
    # Same columns, wrong physical order: order is part of the on-disk contract (R/KNIME
    # read positionally-typed files), so this must die at write time.
    df = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "date": pd.Series([pd.Timestamp("2024-01-02", tz="UTC")]).astype(
                "datetime64[ns, UTC]"
            ),
            "close": [100.0],
        }
    )
    path = str(tmp_path / "misordered.parquet")
    with pytest.raises(SchemaError):
        write_frame(df, path, "prices")
    assert not os.path.exists(path)
