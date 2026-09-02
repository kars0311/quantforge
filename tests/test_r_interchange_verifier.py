"""Independent verifier tests for the R Parquet round-trip milestone (week-6 preflight).

Written adversarially, separate from the builder's suite, and deliberately importing the
builder's *actual* R script text (not a copy): the point is to prove that the R-side
``stopifnot`` guard the round-trip tests rely on really fails on off-contract files, not that
some similar-looking guard would. Three corrupted fixtures are built with raw pyarrow —
bypassing ``write_frame``, which would (correctly) refuse to write them — each violating the
date convention a different way:

- tz-naive timestamp: R reads POSIXct but with a NULL tzone attr, so ``identical(tzone, "UTC")``
  must fail — this is the exact ambiguity the UTC rule exists to catch.
- tz-aware non-UTC (America/New_York): POSIXct with the wrong tzone; a consumer that "helpfully"
  converted displays would silently shift every date by hours.
- date32: arrow hands R a ``Date``, not POSIXct at all, so the class assertion must fire first.

Plus the return leg's teeth (``read_frame`` rejecting an off-contract value dtype, since the R
script intentionally checks only the date semantics and defers value dtypes to Python), and a
bit-exact fidelity check with hand-picked awkward doubles — stricter than the builder's 1e-12
tolerance, justified because Parquet float64 -> R double -> Parquet float64 involves no rounding.

Same availability gate as the builder's suite: skips module-wide without Rscript + R arrow.
"""

import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_r_interchange as builder_suite

from quantforge.interchange import SchemaError, read_frame, write_frame

pytestmark = pytest.mark.skipif(
    not builder_suite._r_arrow_available(),
    reason="requires Rscript with the R 'arrow' package on PATH",
)


def _run_builder_r_script(tmp_path, in_path, out_path, cols: str, rows: int):
    """Run the builder's throwaway R contract script exactly as the round-trip tests do."""
    script = tmp_path / "roundtrip.R"
    script.write_text(builder_suite._R_ROUNDTRIP_SCRIPT)
    return subprocess.run(
        ["Rscript", str(script), str(in_path), str(out_path), cols, str(rows)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _write_corrupt_prices(tmp_path, date_array: pa.Array):
    """Write a prices-shaped Parquet whose date column is deliberately off-contract.

    Built with raw pyarrow because ``write_frame`` refuses invalid frames — the corruption has
    to be injected below the boundary, the way a buggy foreign writer would produce it.
    """
    table = pa.table(
        {
            "date": date_array,
            "ticker": pa.array(["AAPL", "MSFT"], pa.string()),
            "close": pa.array([185.64, 370.87], pa.float64()),
        }
    )
    path = tmp_path / "corrupt_prices.parquet"
    pq.write_table(table, path)
    return path


_MIDNIGHTS = pd.to_datetime(["2024-01-02", "2024-01-03"])

# (corruption id, date array builder, R error message the guard must emit).
# Messages match the named stopifnot conditions in the builder's R script; asserting on them
# proves the *right* assertion fired, not just that R exploded somewhere.
_CORRUPTIONS = [
    (
        "tz_naive",
        lambda: pa.array(_MIDNIGHTS, pa.timestamp("ns")),
        "date tzone must be UTC",
    ),
    (
        "non_utc_tz",
        lambda: pa.array(
            _MIDNIGHTS.tz_localize("America/New_York"),
            pa.timestamp("ns", tz="America/New_York"),
        ),
        "date tzone must be UTC",
    ),
    (
        "date32_not_timestamp",
        lambda: pa.array(_MIDNIGHTS.date, pa.date32()),
        "date must arrive in R as POSIXct",
    ),
]


@pytest.mark.parametrize(
    ("dates", "expected_error"),
    [pytest.param(build, err, id=name) for name, build, err in _CORRUPTIONS],
)
def test_r_guard_rejects_corrupted_date_dtype(dates, expected_error, tmp_path):
    """Corrupt the date dtype in a fixture and the R-side contract check must fail, by name.

    This is the done-when criterion verbatim: the timezone/class assertions in the R script are
    only worth shipping if an off-contract file actually trips them. Each corruption must
    produce a nonzero Rscript exit, the specific named stopifnot message, and no output file
    (a failed contract check may never leave a plausible-looking artifact behind).
    """
    in_path = _write_corrupt_prices(tmp_path, dates())
    out_path = tmp_path / "never_written.parquet"
    proc = _run_builder_r_script(tmp_path, in_path, out_path, "date,ticker,close", 2)

    assert proc.returncode != 0, (
        f"R script accepted an off-contract date column; stdout/stderr:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert expected_error in proc.stderr
    assert not out_path.exists()


def test_r_guard_rejects_wrong_row_count(tmp_path):
    """Lying about the expected row count must also fail — the nrow assertion executes too."""
    df = builder_suite._prices_frame()
    py_path = tmp_path / "prices.parquet"
    write_frame(df, str(py_path), "prices")

    proc = _run_builder_r_script(
        tmp_path, py_path, tmp_path / "never_written.parquet", "date,ticker,close", len(df) + 1
    )
    assert proc.returncode != 0
    assert "row count mismatch" in proc.stderr
    assert not (tmp_path / "never_written.parquet").exists()


def test_python_return_leg_rejects_wrong_value_dtype(tmp_path):
    """The Python read leg — not R — is what guards value dtypes; prove it has teeth.

    The R script checks only the date semantics (the genuinely cross-language ambiguity), so if
    a foreign writer produced float32 closes, catching it falls entirely to ``read_frame``'s
    re-validation. Feed it such a file directly and require SchemaError naming the column —
    this is the assertion the round-trip test leans on when it calls read_frame on R's output.
    """
    table = pa.table(
        {
            "date": pa.array(_MIDNIGHTS.tz_localize("UTC"), pa.timestamp("ns", tz="UTC")),
            "ticker": pa.array(["AAPL", "MSFT"], pa.string()),
            "close": pa.array([185.0, 370.0], pa.float32()),  # off-contract: float32
        }
    )
    path = tmp_path / "float32_close.parquet"
    pq.write_table(table, path)

    with pytest.raises(SchemaError, match="close"):
        read_frame(str(path), "prices")


def test_r_round_trip_is_bit_exact_hand_computed(tmp_path):
    """Round-trip through R preserves every bit: hand-picked awkward doubles and exact ns dates.

    Stricter than the builder's atol=1e-12: Parquet float64 -> R double -> Parquet float64 is
    a chain of identical 64-bit representations with no arithmetic, so any difference at all —
    even one ulp — is a real interop bug. Values include 0.1 + 0.2 (the classic
    0.30000000000000004) and a subnormal-adjacent tiny return, chosen so a float32 detour or a
    decimal re-parse anywhere in the chain could not go unnoticed. Dates are compared on their
    raw int64 nanosecond values, which also re-proves the POSIXct (double seconds) leg is exact
    for midnight-aligned daily timestamps.
    """
    awkward = [0.1 + 0.2, -1.0 / 3.0, 5e-324 * 2**40, 0.0007]
    original = pd.DataFrame(
        {
            "date": pd.Series(
                pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
            )
            .dt.tz_localize("UTC")
            .astype("datetime64[ns, UTC]"),
            "ret": awkward,
        }
    )
    returned = builder_suite._round_trip_through_r(original, "returns", tmp_path)

    # Hand-computed expectations, independent of the original frame object.
    assert list(returned.columns) == ["date", "ret"]
    assert str(returned["date"].dtype) == "datetime64[ns, UTC]"
    expected_ns = np.array(
        [1704240000000000000, 1704326400000000000, 1704412800000000000, 1704672000000000000]
    )
    assert (returned["date"].astype("int64").to_numpy() == expected_ns).all()
    ret_bits = returned["ret"].to_numpy().view(np.int64)
    expected_bits = np.array(awkward, dtype=np.float64).view(np.int64)
    assert (ret_bits == expected_bits).all(), "float64 values changed at the bit level through R"
