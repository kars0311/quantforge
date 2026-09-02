"""R-side Parquet interchange check — certifies SCHEMAS as the frozen cross-language contract.

Week 1 shipped the Arrow/Parquet interchange with a documented promise: "Python writes; R reads
the same files natively." Until now that promise was only tested Python-to-Python. This suite
closes the loop with a full Python -> R -> Python round trip:

1. Python writes a ``prices`` and a ``returns`` frame via :func:`quantforge.interchange.write_frame`
   (so the file carries the exact physical schema of ``SCHEMAS[kind]``).
2. An Rscript reads each file with ``arrow::read_parquet`` and asserts the contract *on the R
   side* — row count, column names in order, and that ``date`` arrives as POSIXct with tzone
   "UTC". The timezone assertion is the load-bearing one: a tz-naive or non-UTC date column is
   exactly the cross-language ambiguity the contract exists to eliminate, and only an in-R check
   proves R's reader agrees with our convention rather than guessing.
3. R writes the frame back out; Python reads it with :func:`read_frame` (which re-validates
   against SCHEMAS — R is a "foreign writer" and gets no trust) and asserts values match the
   originals: dates equal at ns-UTC, floats to 1e-12.

If this passes, any future change that breaks R interop must first break this test, which is what
"frozen contract" means in practice. The throwaway R script lives in pytest's tmp_path, never in
the repo, so the R layer proper (week 6's tearsheet) stays the only .R code we ship.

Requires a local ``Rscript`` with the ``arrow`` package; skipped (module-wide) when either is
absent so CI without R stays green without weakening local runs.
"""

import shutil
import subprocess

import pandas as pd
import pytest

from quantforge.interchange import read_frame, write_frame


def _r_arrow_available() -> bool:
    """True iff Rscript exists and can load the `arrow` package.

    The probe exits nonzero when arrow is missing (a bare ``requireNamespace`` call always exits
    0 — it *returns* FALSE rather than failing — so we must translate the FALSE into an exit
    status ourselves). Any OS-level failure to launch R counts as "not available", not an error:
    a machine without R should skip this suite, never break the build.
    """
    if shutil.which("Rscript") is None:
        return False
    try:
        proc = subprocess.run(
            ["Rscript", "-e", 'if (!requireNamespace("arrow", quietly=TRUE)) quit(status=1)'],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


pytestmark = pytest.mark.skipif(
    not _r_arrow_available(),
    reason="requires Rscript with the R 'arrow' package on PATH",
)


#: Throwaway R script, written into tmp_path at test time (never into the repo).
#: It takes (in_path, out_path, comma-joined expected columns, expected row count) as CLI args,
#: asserts the contract on the R side, then re-writes the frame for the Python return leg.
#: `stopifnot` with named conditions makes a failed assertion name itself in stderr, so a broken
#: contract is diagnosable straight from the pytest failure message.
_R_ROUNDTRIP_SCRIPT = """\
suppressPackageStartupMessages(library(arrow))
args <- commandArgs(trailingOnly = TRUE)
in_path <- args[[1]]
out_path <- args[[2]]
expected_cols <- strsplit(args[[3]], ",")[[1]]
expected_rows <- as.integer(args[[4]])

df <- arrow::read_parquet(in_path)
stopifnot(
  "row count mismatch" = nrow(df) == expected_rows,
  "column names/order mismatch" = identical(names(df), expected_cols),
  "date must arrive in R as POSIXct" = inherits(df[["date"]], "POSIXct"),
  "date tzone must be UTC" = identical(attr(df[["date"]], "tzone"), "UTC")
)
arrow::write_parquet(df, out_path)
"""


def _utc_ns(dates: list[str]) -> pd.Series:
    """Timestamps pinned to the contract's canonical datetime64[ns, UTC] dtype."""
    return pd.Series(pd.to_datetime(dates)).dt.tz_localize("UTC").astype("datetime64[ns, UTC]")


def _round_trip_through_r(df: pd.DataFrame, kind: str, tmp_path) -> pd.DataFrame:
    """Write ``df`` for R, run the R contract check + re-write, and read R's file back.

    The R process's exit status carries the R-side assertions: any `stopifnot` failure makes
    Rscript exit nonzero, and we surface its stderr in the pytest failure so the R message
    (e.g. "date tzone must be UTC") is visible without re-running anything by hand.
    """
    py_path = tmp_path / f"{kind}_from_python.parquet"
    r_path = tmp_path / f"{kind}_from_r.parquet"
    write_frame(df, str(py_path), kind)

    script = tmp_path / "roundtrip.R"
    script.write_text(_R_ROUNDTRIP_SCRIPT)
    proc = subprocess.run(
        [
            "Rscript",
            str(script),
            str(py_path),
            str(r_path),
            ",".join(df.columns),
            str(len(df)),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"R-side contract check failed for kind {kind!r} "
        f"(exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
    # read_frame re-validates against SCHEMAS[kind]: this is the assertion that R produced a
    # conforming file, not merely *a* file.
    return read_frame(str(r_path), kind)


def _prices_frame() -> pd.DataFrame:
    # Deliberately non-round closes (repeating binary fractions) so the float64 comparison
    # actually exercises bit-level fidelity, not just "roughly the same number".
    return pd.DataFrame(
        {
            "date": _utc_ns(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04"]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "close": [185.64, 370.87, 184.25, 370.60, 181.91, 367.94],
        }
    )


def _returns_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": _utc_ns(["2024-01-03", "2024-01-04", "2024-01-05"]),
            "ret": [0.0123456789012345, -0.0210987654321098, 0.0007],
        }
    )


@pytest.mark.parametrize(
    ("kind", "builder"),
    [("prices", _prices_frame), ("returns", _returns_frame)],
)
def test_r_round_trip_preserves_values(kind, builder, tmp_path):
    """Python -> R -> Python leaves every value intact and every dtype on-contract.

    Dates must come back equal at ns-UTC precision (R hands arrow a POSIXct — double seconds —
    so midnight-aligned daily dates are exactly representable and any drift would be a real
    interop bug, not rounding). Floats are compared to 1e-12 absolute, effectively bit-exact
    for prices/returns magnitudes, per the contract's "lossless round-trip" promise.
    """
    original = builder()
    returned = _round_trip_through_r(original, kind, tmp_path)

    assert list(returned.columns) == list(original.columns)
    assert str(returned["date"].dtype) == "datetime64[ns, UTC]"
    pd.testing.assert_series_equal(returned["date"], original["date"], check_exact=True)
    pd.testing.assert_frame_equal(returned, original, rtol=0.0, atol=1e-12)


def test_r_side_assertion_rejects_wrong_contract(tmp_path):
    """The R script's checks have teeth: lie about the expected columns and R must fail.

    This guards the guard — if the R-side `stopifnot` were accidentally vacuous (say, a typo
    that made every condition TRUE), the round-trip tests above would keep passing while
    certifying nothing on the R side. Feeding R a deliberately wrong expectation and requiring
    a nonzero exit proves the assertions actually execute and can fail.
    """
    df = _returns_frame()
    py_path = tmp_path / "returns.parquet"
    write_frame(df, str(py_path), "returns")

    script = tmp_path / "roundtrip.R"
    script.write_text(_R_ROUNDTRIP_SCRIPT)
    proc = subprocess.run(
        [
            "Rscript",
            str(script),
            str(py_path),
            str(tmp_path / "never_written.parquet"),
            "date,wrong_column",  # contract violation R must catch
            str(len(df)),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "column names/order mismatch" in proc.stderr
    assert not (tmp_path / "never_written.parquet").exists()
