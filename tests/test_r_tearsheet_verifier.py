"""Verifier suite for week-6 milestone 2: tearsheet.R part 1 (CLI + read + metrics + write).

Adversarial cross-language check of ``analytics_r/tearsheet.R`` against its contract
(docs/components/08-r-tearsheet.md) and the Python reference implementation
(``quantforge.metrics.performance.compute_metrics`` — the single source of truth, AR-3):

1. Happy path: on a fixture dir holding a valid interchange ``returns`` file, the script
   exits 0 and writes ``metrics_r.parquet`` that Python's ``read_frame(..., "metrics")``
   validates, with the six ``_KEYS`` names in exact order and every value matching the
   Python reference within relative 1e-9 (the formulas are transcriptions, so agreement
   should be near machine precision — far inside the ~1% RG-6 tolerance).
2. Hand-computed values: a two-day series whose six metrics are derived by hand on paper,
   so R and Python can't "agree" merely by sharing a common bug — the numbers themselves
   are asserted, not just cross-language consistency.
3. Convention edges: constant returns must yield NaN sharpe on BOTH sides (std == 0 guard),
   and the strictly-greater hit_rate must exclude zero-return days.
4. Failure paths: missing returns.parquet, a misnamed column, a wrong-dtype ``ret``, and an
   all-NA series must each exit non-zero with a diagnostic naming the problem, and must
   never leave a metrics_r.parquet behind (a partial artifact would poison the next stage).
5. CLI defaults: with no arguments and cwd at the fixture root, the script reads
   ``data_cache/`` and creates ``analytics_r/output/`` per the component contract.

Requires a local ``Rscript`` with everything the script attaches (arrow, xts, tidyquant,
PerformanceAnalytics, PortfolioAnalytics, ROI, ROI.plugin.quadprog); skipped module-wide
when absent (same policy as tests/test_r_interchange.py) so CI without R stays green.
"""

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantforge.interchange import read_frame, validate_frame, write_frame
from quantforge.metrics.performance import compute_metrics

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "analytics_r" / "tearsheet.R"

_KEYS = ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "hit_rate"]


def _r_deps_available() -> bool:
    """True iff Rscript exists and can load every package tearsheet.R needs.

    ``requireNamespace`` returns FALSE instead of failing, so the probe translates FALSE
    into a non-zero exit itself. Any OS-level failure to launch R means "skip", never
    "break the build" — a machine without R must not go red. The list tracks what the
    script actually attaches (milestone 3 added the tidyquant stack, milestone 4 the
    PortfolioAnalytics/ROI stack), so a machine with only arrow+xts must skip, not fail.
    """
    if shutil.which("Rscript") is None:
        return False
    probe = (
        'ok <- all(vapply(c("arrow", "xts", "tidyquant", "PerformanceAnalytics",'
        ' "PortfolioAnalytics", "ROI", "ROI.plugin.quadprog"),'
        " requireNamespace, logical(1), quietly=TRUE)); if (!ok) quit(status=1)"
    )
    try:
        proc = subprocess.run(["Rscript", "-e", probe], capture_output=True, timeout=60)
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
    """Write a fixed, valid ``asset_returns`` panel — REQUIRED input since milestone 4.

    The optimizer cross-check made asset_returns.parquet mandatory, so every fixture that
    expects exit 0 (and every failure fixture whose diagnostic should come from a LATER
    step) needs one. A deterministic 3-ticker, 60-day dense panel; its values are unrelated
    to the portfolio series under test because the metrics and optimizer cross-checks are
    independent — the contract does not require the two files to share dates.
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


def _write_returns(tmp_path: Path, rets: list[float]) -> Path:
    """Write valid interchange ``returns`` + ``asset_returns`` files; return the fixture dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    df = pd.DataFrame({"date": _utc_ns(len(rets)), "ret": pd.Series(rets, dtype="float64")})
    write_frame(df, str(data_dir / "returns.parquet"), "returns")
    _write_asset_returns(data_dir)
    return data_dir


def _run_tearsheet(data_dir: Path, out_dir: Path, cwd: Path | None = None, args=None):
    """Shell out to the script exactly as the contract specifies; return CompletedProcess."""
    cmd = ["Rscript", str(_SCRIPT)]
    cmd += [str(data_dir), str(out_dir)] if args is None else list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=cwd or _REPO_ROOT)


def _read_metrics(data_dir: Path) -> pd.DataFrame:
    """Read R's output through the same trust boundary the pipeline would use."""
    df = read_frame(str(data_dir / "metrics_r.parquet"), "metrics")
    validate_frame(df, "metrics")  # explicit, though read_frame already validated
    return df


def _assert_matches_python(metrics_r: pd.DataFrame, rets: list[float]) -> None:
    """Names in exact _KEYS order; every value within rel 1e-9 of the Python reference."""
    assert list(metrics_r["name"]) == _KEYS
    py = compute_metrics(pd.Series(rets, dtype="float64"))
    for name, r_val in zip(metrics_r["name"], metrics_r["value"]):
        py_val = py[name]
        if math.isnan(py_val):
            assert math.isnan(r_val), f"{name}: Python NaN but R gave {r_val}"
        else:
            assert math.isclose(r_val, py_val, rel_tol=1e-9, abs_tol=1e-12), (
                f"{name}: R {r_val!r} != Python {py_val!r}"
            )


# --- happy path ---------------------------------------------------------------------------


def test_happy_path_matches_python_reference(tmp_path):
    """Realistic random-ish series: exit 0, valid metrics file, six values match Python."""
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0004, 0.01, size=120).tolist()
    data_dir = _write_returns(tmp_path, rets)
    out_dir = tmp_path / "out"

    proc = _run_tearsheet(data_dir, out_dir)
    assert proc.returncode == 0, f"tearsheet.R failed:\n{proc.stdout}\n{proc.stderr}"
    assert out_dir.is_dir()  # contract: out_dir exists after any successful run

    _assert_matches_python(_read_metrics(data_dir), rets)


def test_hand_computed_two_day_series(tmp_path):
    """Adversarial: values derived on paper, so a shared R/Python bug cannot hide.

    r = [0.10, -0.05]:
      total_return = 1.10 * 0.95 - 1              = 0.045
      cagr         = 1.045 ** (252/2) - 1
      pop. std     = 0.075  (mean 0.025, deviations exactly +/-0.075)
      ann_vol      = 0.075 * sqrt(252)
      sharpe       = (0.025 / 0.075) * sqrt(252)  = sqrt(252) / 3
      max_drawdown = 1.045/1.10 - 1               = -0.05  (peak day 1, trough day 2)
      hit_rate     = 0.5
    """
    rets = [0.10, -0.05]
    data_dir = _write_returns(tmp_path, rets)
    proc = _run_tearsheet(data_dir, tmp_path / "out")
    assert proc.returncode == 0, proc.stderr

    metrics = _read_metrics(data_dir)
    got = dict(zip(metrics["name"], metrics["value"]))
    expected = {
        "total_return": 0.045,
        "cagr": 1.045**126 - 1,
        "ann_vol": 0.075 * math.sqrt(252),
        "sharpe": math.sqrt(252) / 3,
        "max_drawdown": -0.05,
        "hit_rate": 0.5,
    }
    for name, exp in expected.items():
        assert math.isclose(got[name], exp, rel_tol=1e-12, abs_tol=1e-15), (
            f"{name}: R {got[name]!r} != hand-computed {exp!r}"
        )
    # And the same file must also agree with the Python reference (transitively certifies
    # compute_metrics against the hand computation too).
    _assert_matches_python(metrics, rets)


def test_constant_returns_nan_sharpe_and_strict_hit_rate(tmp_path):
    """std == 0 must yield NaN sharpe (not Inf/0), and zero days must not count as hits."""
    rets = [0.0, 0.0, 0.0, 0.0]
    data_dir = _write_returns(tmp_path, rets)
    proc = _run_tearsheet(data_dir, tmp_path / "out")
    assert proc.returncode == 0, proc.stderr

    metrics = _read_metrics(data_dir)
    got = dict(zip(metrics["name"], metrics["value"]))
    assert math.isnan(got["sharpe"])  # 0/0 guarded exactly as Python guards it
    assert got["hit_rate"] == 0.0  # strictly-greater convention: 0.0 is not a hit
    assert got["total_return"] == 0.0
    assert got["max_drawdown"] == 0.0
    _assert_matches_python(metrics, rets)


def test_nan_rows_are_dropped_like_python(tmp_path):
    """A file with some NA returns computes on the non-NA subset, matching .dropna()."""
    rets = [0.01, float("nan"), -0.02, float("nan"), 0.03]
    data_dir = _write_returns(tmp_path, rets)
    proc = _run_tearsheet(data_dir, tmp_path / "out")
    assert proc.returncode == 0, proc.stderr
    _assert_matches_python(_read_metrics(data_dir), rets)


def test_cli_defaults_data_cache_and_output_dir(tmp_path):
    """No args + cwd at a fixture root: reads data_cache/, creates analytics_r/output/."""
    root = tmp_path / "fixture_root"
    (root / "data_cache").mkdir(parents=True)
    rets = [0.01, -0.005, 0.02]
    df = pd.DataFrame({"date": _utc_ns(len(rets)), "ret": rets})
    write_frame(df, str(root / "data_cache" / "returns.parquet"), "returns")
    _write_asset_returns(root / "data_cache")

    proc = _run_tearsheet(None, None, cwd=root, args=[])
    assert proc.returncode == 0, proc.stderr
    assert (root / "analytics_r" / "output").is_dir()
    _assert_matches_python(_read_metrics(root / "data_cache"), rets)


# --- failure paths: every malformed hand-off must exit non-zero, name the problem, and
# --- leave no metrics_r.parquet behind ----------------------------------------------------


def _assert_failed_cleanly(proc, data_dir: Path, *needles: str) -> None:
    assert proc.returncode != 0, "script must exit non-zero on a broken hand-off"
    for needle in needles:
        assert needle in proc.stderr, (
            f"diagnostic should name the problem ({needle!r}); stderr was:\n{proc.stderr}"
        )
    assert not (data_dir / "metrics_r.parquet").exists(), (
        "a failed run must not leave a partial metrics artifact behind"
    )


def test_missing_returns_file_exits_nonzero(tmp_path):
    data_dir = tmp_path / "empty_dir"
    data_dir.mkdir()
    proc = _run_tearsheet(data_dir, tmp_path / "out")
    _assert_failed_cleanly(proc, data_dir, "returns.parquet")


def test_misnamed_column_exits_nonzero(tmp_path):
    """A column named 'return' instead of 'ret' must be rejected with both names visible."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # write_frame would (rightly) refuse this, so build the bad file with pyarrow directly —
    # exactly what a buggy foreign writer would produce.
    table = pa.table(
        {
            "date": pa.array(_utc_ns(3), type=pa.timestamp("ns", tz="UTC")),
            "return": pa.array([0.01, 0.02, 0.03], type=pa.float64()),
        }
    )
    pq.write_table(table, data_dir / "returns.parquet")
    proc = _run_tearsheet(data_dir, tmp_path / "out")
    _assert_failed_cleanly(proc, data_dir, "columns", "ret")


def test_extra_column_exits_nonzero(tmp_path):
    """Exactly-two-columns means an extra column is a contract violation, not a bonus."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    table = pa.table(
        {
            "date": pa.array(_utc_ns(3), type=pa.timestamp("ns", tz="UTC")),
            "ret": pa.array([0.01, 0.02, 0.03], type=pa.float64()),
            "ticker": pa.array(["A", "A", "A"], type=pa.string()),
        }
    )
    pq.write_table(table, data_dir / "returns.parquet")
    proc = _run_tearsheet(data_dir, tmp_path / "out")
    _assert_failed_cleanly(proc, data_dir, "columns")


def test_integer_ret_dtype_exits_nonzero(tmp_path):
    """ret stored as int64 violates float64; must fail at the boundary, not compute quietly."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    table = pa.table(
        {
            "date": pa.array(_utc_ns(3), type=pa.timestamp("ns", tz="UTC")),
            "ret": pa.array([1, 0, 2], type=pa.int64()),
        }
    )
    pq.write_table(table, data_dir / "returns.parquet")
    proc = _run_tearsheet(data_dir, tmp_path / "out")
    _assert_failed_cleanly(proc, data_dir, "ret")


def test_all_na_returns_exits_nonzero(tmp_path):
    """NA-only series: Python returns a NaN dict, but the batch hand-off must fail loudly —
    a file of NaNs would defeat any tolerance comparison downstream (per-milestone spec)."""
    data_dir = _write_returns(tmp_path, [float("nan")] * 5)
    proc = _run_tearsheet(data_dir, tmp_path / "out")
    _assert_failed_cleanly(proc, data_dir, "no non-NA")


def test_empty_returns_file_exits_nonzero(tmp_path):
    """Zero rows is a broken pipeline, not a valid input."""
    data_dir = _write_returns(tmp_path, [])
    proc = _run_tearsheet(data_dir, tmp_path / "out")
    _assert_failed_cleanly(proc, data_dir)
