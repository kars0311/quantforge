"""Verifier suite for week-6 milestone 3: tearsheet.R part 2 (headless PNG + display tables).

Adversarial check of the tearsheet step added to ``analytics_r/tearsheet.R`` against its
contract (docs/components/08-r-tearsheet.md, internal step 2) and the milestone spec:

1. Headless PNG: with ``DISPLAY`` scrubbed from the environment (the CI/Docker reality the
   png() device must survive — quartz()/X11() would die here), the script exits 0 and writes
   ``out_dir/tearsheet.png`` that is non-empty, starts with the PNG magic bytes, and whose
   IHDR header carries the contract's exact 1200x900 dimensions — parsed from raw bytes, so
   a zero-byte or truncated device dump cannot pass.
2. Display tables: stdout carries both institutional-style tables (annualized returns and
   worst drawdowns), including a hand-computable drawdown depth, and does NOT carry the
   ``"null device"`` string a bare (non-invisible) ``dev.off()`` would auto-print — the
   easiest way for the chart step to silently corrupt the stdout contract.
3. Non-interference (the milestone's "unchanged" clause): on the same run, metrics_r.parquet
   still validates through ``read_frame(..., "metrics")`` and matches the Python reference
   within rel 1e-9 — the display tables use PerformanceAnalytics' own conventions (sample
   std, geometric annualization) and must stay display-only (AR-3: match Python's
   definitions across the Parquet boundary, don't invent variants). Success stderr stays
   breadcrumbs-only, so a real failure's stderr is pure signal.
4. Failure paths stay clean: a malformed hand-off (misnamed column) and an all-NA series
   must still exit non-zero with the milestone-2 diagnostics and must leave NO tearsheet.png
   (both fail before the graphics device opens) and no metrics_r.parquet.
5. NA robustness: interior-NA rows (legal per the contract, dropped by the metrics) must not
   crash the chart or tables.

Requires a local ``Rscript`` with the full stack the script now loads (arrow, tidyquant —
which attaches PerformanceAnalytics/xts — plus PortfolioAnalytics/ROI); skipped module-wide
when absent so CI without R stays green.
"""

import math
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantforge.interchange import read_frame, write_frame
from quantforge.metrics.performance import compute_metrics

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "analytics_r" / "tearsheet.R"

_KEYS = ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "hit_rate"]

# 8-byte PNG signature (\x89PNG\r\n\x1a\n) — the milestone's "is really a PNG" check.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _r_deps_available() -> bool:
    """True iff Rscript exists and can load every package tearsheet.R now attaches.

    Milestone 3 added tidyquant (which pulls in PerformanceAnalytics/xts/zoo), so the probe
    must cover it too — a machine with arrow+xts but no tidyquant must skip, not go red.
    """
    if shutil.which("Rscript") is None:
        return False
    probe = (
        'ok <- all(vapply(c("arrow", "xts", "tidyquant", "PerformanceAnalytics",'
        ' "PortfolioAnalytics", "ROI", "ROI.plugin.quadprog"),'
        " requireNamespace, logical(1), quietly=TRUE)); if (!ok) quit(status=1)"
    )
    try:
        proc = subprocess.run(["Rscript", "-e", probe], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


pytestmark = pytest.mark.skipif(
    not _r_deps_available(),
    reason="requires Rscript with the arrow/tidyquant/PortfolioAnalytics/ROI stacks on PATH",
)


def _utc_ns(n: int, start: str = "2024-01-02") -> pd.Series:
    """n business-day timestamps in the contract's canonical datetime64[ns, UTC] dtype."""
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(idx).dt.tz_localize("UTC").astype("datetime64[ns, UTC]")


def _write_asset_returns(data_dir: Path) -> None:
    """Write the fixed, valid ``asset_returns`` panel — REQUIRED input since milestone 4.

    Same rationale as in test_r_tearsheet_verifier.py: the optimizer cross-check made
    asset_returns.parquet mandatory, so every fixture whose run must get PAST the input
    boundary needs one; a deterministic dense 3-ticker panel keeps these PNG/table tests
    exercising exactly what they always did.
    """
    rng = np.random.default_rng(7)
    n, tickers = 60, ["AAA", "BBB", "CCC"]
    df = pd.DataFrame(
        {
            "date": _utc_ns(n).repeat(len(tickers)).reset_index(drop=True),
            "ticker": tickers * n,
            "ret": rng.normal(0.0005, 0.01, size=n * len(tickers)),
        }
    )
    write_frame(df, str(data_dir / "asset_returns.parquet"), "asset_returns")


def _write_returns(root: Path, rets: list[float]) -> Path:
    """Write valid interchange ``returns`` + ``asset_returns`` files; return the fixture dir."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"date": _utc_ns(len(rets)), "ret": pd.Series(rets, dtype="float64")})
    write_frame(df, str(data_dir / "returns.parquet"), "returns")
    _write_asset_returns(data_dir)
    return data_dir


def _run_headless(data_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    """Run the script with DISPLAY scrubbed — the spec's no-display environment.

    A png() device needs no display server; quartz()/X11() would. Removing DISPLAY (and
    doing nothing else) is exactly how a Linux CI box or Docker container looks, so a pass
    here is evidence the script is genuinely headless, not merely headless-on-a-Mac.
    """
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    return subprocess.run(
        ["Rscript", str(_SCRIPT), str(data_dir), str(out_dir)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_REPO_ROOT,
        env=env,
    )


@pytest.fixture(scope="module")
def happy_run(tmp_path_factory):
    """One shared headless run on the milestone-2 fixture (120-day random-ish series).

    Module-scoped: the PNG, stdout, stderr, and metrics assertions below all interrogate
    this single run, keeping the suite fast without weakening any check.
    """
    root = tmp_path_factory.mktemp("tearsheet_png")
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0004, 0.01, size=120).tolist()
    data_dir = _write_returns(root, rets)
    out_dir = root / "out"
    proc = _run_headless(data_dir, out_dir)
    return proc, data_dir, out_dir, rets


# --- the visual deliverable ---------------------------------------------------------------


def test_headless_run_exits_zero(happy_run):
    proc, _, _, _ = happy_run
    assert proc.returncode == 0, f"tearsheet.R failed headless:\n{proc.stdout}\n{proc.stderr}"


def test_png_exists_with_magic_bytes_and_contract_dimensions(happy_run):
    """tearsheet.png must be a real, finalized PNG at the spec's 1200x900 — not a stub file.

    The IHDR chunk directly follows the 8-byte signature: 4-byte length + b"IHDR", then
    big-endian width and height at byte offsets 16 and 20. Parsing them proves the device
    was opened with the contracted size AND that dev.off() ran (an unclosed device leaves
    a truncated file that still starts with the magic bytes).
    """
    proc, _, out_dir, _ = happy_run
    assert proc.returncode == 0, proc.stderr
    png_path = out_dir / "tearsheet.png"
    assert png_path.exists(), "successful run must write out_dir/tearsheet.png"
    blob = png_path.read_bytes()
    assert len(blob) > 1024, f"PNG suspiciously small ({len(blob)} bytes) — empty chart?"
    assert blob[:8] == _PNG_MAGIC, "file does not start with the PNG magic bytes"
    assert blob[12:16] == b"IHDR"
    width, height = struct.unpack(">II", blob[16:24])
    assert (width, height) == (1200, 900), f"device size {width}x{height} != contract 1200x900"
    # IEND is the final chunk of a well-formed PNG; its presence means the file was flushed
    # and closed, not abandoned mid-render.
    assert b"IEND" in blob[-16:], "PNG not finalized (missing IEND) — dev.off() didn't run?"


def test_stdout_carries_both_display_tables(happy_run):
    """A headless run must still show the institutional-style tables in the terminal/log."""
    proc, _, _, _ = happy_run
    assert proc.returncode == 0, proc.stderr
    # table.AnnualizedReturns rows
    assert "Annualized Return" in proc.stdout
    assert "Annualized Std Dev" in proc.stdout
    assert "Annualized Sharpe" in proc.stdout
    # table.Drawdowns columns
    for col in ("From", "Trough", "Depth", "Recovery"):
        assert col in proc.stdout, f"drawdowns table column {col!r} missing from stdout"


def test_stdout_has_no_null_device_leak(happy_run):
    """dev.off() returns visibly; without invisible() Rscript prints 'null device 1' into
    stdout, corrupting the table output a log-scraper would read. Assert the leak is gone."""
    proc, _, _, _ = happy_run
    assert "null device" not in proc.stdout


def test_success_stderr_is_breadcrumbs_only(happy_run):
    """The file's invariant: stderr is reserved for real errors (plus the two success
    breadcrumbs). Package startup chatter or stray warnings landing here would bury the
    diagnostic when a run actually fails."""
    proc, _, _, _ = happy_run
    for line in proc.stderr.splitlines():
        if line.strip():
            assert line.startswith("tearsheet.R:"), f"unexpected stderr noise: {line!r}"


# --- non-interference: the Parquet cross-check must be untouched by the display step ------


def test_metrics_parquet_still_matches_python_reference(happy_run):
    """AR-3 guard: the tables print PerformanceAnalytics' own conventions, but the numbers
    crossing the Parquet boundary must remain exact transcriptions of performance.py."""
    proc, data_dir, _, rets = happy_run
    assert proc.returncode == 0, proc.stderr
    metrics = read_frame(str(data_dir / "metrics_r.parquet"), "metrics")
    assert list(metrics["name"]) == _KEYS
    py = compute_metrics(pd.Series(rets, dtype="float64"))
    for name, r_val in zip(metrics["name"], metrics["value"]):
        py_val = py[name]
        if math.isnan(py_val):
            assert math.isnan(r_val), f"{name}: Python NaN but R gave {r_val}"
        else:
            assert math.isclose(r_val, py_val, rel_tol=1e-9, abs_tol=1e-12), (
                f"{name}: R {r_val!r} != Python {py_val!r}"
            )


def test_hand_computed_drawdown_appears_in_stdout_table(tmp_path):
    """Adversarial hand computation: r = [0.10, -0.05] has exactly one drawdown, whose depth
    is the -0.05 single-day loss (peak after day 1, trough at day 2). The drawdowns table
    must show that number — proving the table is computed from the actual series, not
    boilerplate that merely looks like a table. Two days < the 5 drawdowns the table wants,
    so this also exercises the suppressed 'Only N available' warning path: still exit 0,
    and the warning must not pollute stderr."""
    data_dir = _write_returns(tmp_path, [0.10, -0.05])
    proc = _run_headless(data_dir, tmp_path / "out")
    assert proc.returncode == 0, proc.stderr
    assert "-0.05" in proc.stdout, f"hand-computed depth missing from:\n{proc.stdout}"
    assert "Only" not in proc.stderr, "informational drawdown-count warning leaked to stderr"
    assert (tmp_path / "out" / "tearsheet.png").read_bytes()[:8] == _PNG_MAGIC


def test_interior_na_rows_still_render_chart_and_tables(tmp_path):
    """NA rows are legal in the hand-off (metrics drop them); the chart/table step must not
    turn a legal input into a crash. Metrics must still match Python's .dropna() result."""
    rets = [0.01, float("nan"), -0.02, float("nan"), 0.03]
    data_dir = _write_returns(tmp_path, rets)
    proc = _run_headless(data_dir, tmp_path / "out")
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "out" / "tearsheet.png").read_bytes()[:8] == _PNG_MAGIC
    metrics = read_frame(str(data_dir / "metrics_r.parquet"), "metrics")
    py = compute_metrics(pd.Series(rets, dtype="float64"))
    for name, r_val in zip(metrics["name"], metrics["value"]):
        assert math.isclose(r_val, py[name], rel_tol=1e-9, abs_tol=1e-12)


# --- failure paths: still fail loudly, and now must ALSO leave no chart artifact ----------


def test_malformed_columns_fail_before_any_artifact(tmp_path):
    """A misnamed column must be rejected at the boundary — before the graphics device ever
    opens — so a failed run leaves neither a tearsheet.png nor a metrics_r.parquet that a
    later stage (or a human) could mistake for real output."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    table = pa.table(
        {
            "date": pa.array(_utc_ns(3), type=pa.timestamp("ns", tz="UTC")),
            "return": pa.array([0.01, 0.02, 0.03], type=pa.float64()),
        }
    )
    pq.write_table(table, data_dir / "returns.parquet")
    out_dir = tmp_path / "out"
    proc = _run_headless(data_dir, out_dir)
    assert proc.returncode != 0
    assert "columns" in proc.stderr
    assert not (out_dir / "tearsheet.png").exists(), "failed run left a chart artifact"
    assert not (data_dir / "metrics_r.parquet").exists()


def test_all_na_fails_with_milestone2_diagnostic_and_no_png(tmp_path):
    """The empty/all-NA guard sits BEFORE the tearsheet step, so a broken hand-off fails
    with the milestone-2 message ('no non-NA'), not a PerformanceAnalytics error about
    charting nothing — and no PNG is left behind."""
    data_dir = _write_returns(tmp_path, [float("nan")] * 5)
    out_dir = tmp_path / "out"
    proc = _run_headless(data_dir, out_dir)
    assert proc.returncode != 0
    assert "no non-NA" in proc.stderr
    assert not (out_dir / "tearsheet.png").exists()
    assert not (data_dir / "metrics_r.parquet").exists()
