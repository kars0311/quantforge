"""Verifier suite for week-6 milestone 4: tearsheet.R part 3 (second optimizer + weights).

Adversarial cross-language check of the PortfolioAnalytics step added to
``analytics_r/tearsheet.R`` against its contract (docs/components/08-r-tearsheet.md,
internal step 4) and the milestone spec:

1. Happy path: with both interchange inputs present the script exits 0 and produces ALL
   THREE outputs (tearsheet.png, metrics_r.parquet, weights_r.parquet); the weights file
   passes ``validate_frame(df, "weights")``, sums to 1 within 1e-6, keeps every weight in
   the long-only box ``[-1e-8, 1 + 1e-8]``, and carries one row per asset in sorted-ticker
   order even when the input file's row order is scrambled.
2. Objective identity (adversarial): on a panel whose min-volatility and max-Sharpe
   portfolios differ by >0.18 in some weight, the R weights must match Python's
   ``optimize_weights(returns, {"objective": "min_volatility"})`` within 1% absolute AND
   must NOT match the max-Sharpe portfolio — so agreement can't come from R quietly
   solving the wrong (but similar-looking) problem.
3. Closed form (adversarial): on a 2-asset panel the weights must match the paper formula
   ``w1 = (s2^2 - rho*s1*s2) / (s1^2 + s2^2 - 2*rho*s1*s2)`` evaluated at the REALIZED
   sample moments — a number neither library computed, so a shared setup bug cannot hide.
4. Binding box (adversarial): a near-collinear pair where the unconstrained min-variance
   solution is a large short (~ -0.96 in the leveraged asset) must come back clipped to
   the long-only corner (0, 1) — proving the box constraint is real, not decorative.
5. Failure paths: asset_returns.parquet missing, ragged (NA after pivot), single-ticker,
   duplicate (date, ticker), wrong dtype, and misordered columns must each exit non-zero
   with a diagnostic and leave NO artifact behind (no png, no metrics, no weights) — the
   input boundary sits before any output is written.
6. Determinism: two runs on the same fixture produce identical weights (ROI/quadprog is a
   deterministic QP; a stochastic search method would make the cross-check meaningless).

Requires a local ``Rscript`` with everything tearsheet.R attaches (arrow, tidyquant,
PortfolioAnalytics, ROI, ROI.plugin.quadprog); skipped module-wide when absent so CI
without R stays green.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantforge.interchange import read_frame, validate_frame, write_frame
from quantforge.portfolio.optimize import optimize_weights

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "analytics_r" / "tearsheet.R"

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _r_deps_available() -> bool:
    """True iff Rscript exists and can load every package tearsheet.R attaches."""
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


def _write_portfolio_returns(data_dir: Path, n: int = 90, seed: int = 5) -> None:
    """The (already-verified) portfolio-returns input every run needs to get to STEP 2."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "date": _utc_ns(n),
            "ret": pd.Series(rng.normal(0.0004, 0.01, size=n), dtype="float64"),
        }
    )
    write_frame(df, str(data_dir / "returns.parquet"), "returns")


def _long_from_wide(wide: pd.DataFrame, *, shuffle_seed: int | None = None) -> pd.DataFrame:
    """Melt a wide panel to the interchange long format, optionally scrambling row order.

    Scrambling matters: the contract says the OUTPUT is sorted by ticker; a script that
    merely preserves input order would pass an already-sorted fixture by accident.
    """
    n = len(wide)
    dates = _utc_ns(n)
    rows = [
        pd.DataFrame(
            {
                "date": dates,
                "ticker": [col] * n,
                "ret": wide[col].to_numpy(dtype="float64"),
            }
        )
        for col in wide.columns
    ]
    long = pd.concat(rows, ignore_index=True)
    if shuffle_seed is not None:
        long = long.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    return long


def _write_fixture(tmp_path: Path, wide: pd.DataFrame, *, shuffle_seed: int | None = None):
    """Write both inputs from a wide asset panel; return (data_dir, out_dir)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    _write_portfolio_returns(data_dir)
    long = _long_from_wide(wide, shuffle_seed=shuffle_seed)
    write_frame(long, str(data_dir / "asset_returns.parquet"), "asset_returns")
    return data_dir, tmp_path / "out"


def _run(data_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["Rscript", str(_SCRIPT), str(data_dir), str(out_dir)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_REPO_ROOT,
    )


def _four_asset_panel() -> pd.DataFrame:
    """250-day, 4-asset panel with distinct means/vols so min-vol != max-Sharpe (>0.18 gap)."""
    rng = np.random.default_rng(123)
    n = 250
    means = [0.0002, 0.0012, 0.0005, 0.0009]
    vols = [0.006, 0.02, 0.009, 0.013]
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {f"T{i}": rng.normal(m, v, size=n) for i, (m, v) in enumerate(zip(means, vols))},
        index=dates,
    )


def _read_weights(data_dir: Path) -> pd.DataFrame:
    """Read R's weights through the same trust boundary the pipeline would use."""
    df = read_frame(str(data_dir / "weights_r.parquet"), "weights")
    validate_frame(df, "weights")  # explicit, though read_frame already validated
    return df


# --- happy path: all three outputs, valid weights schema, budget + box + order ------------


def test_all_three_outputs_and_weights_contract(tmp_path):
    wide = _four_asset_panel()
    # Scramble the long file's row order: sorted output must be the script's doing.
    data_dir, out_dir = _write_fixture(tmp_path, wide, shuffle_seed=99)
    proc = _run(data_dir, out_dir)
    assert proc.returncode == 0, f"tearsheet.R failed:\n{proc.stdout}\n{proc.stderr}"

    # All three artifacts of the finished script.
    assert (out_dir / "tearsheet.png").read_bytes()[:8] == _PNG_MAGIC
    read_frame(str(data_dir / "metrics_r.parquet"), "metrics")
    weights = _read_weights(data_dir)

    # One row per asset — near-zero weights included — in sorted-ticker order.
    assert list(weights["ticker"]) == sorted(wide.columns)
    # Fully-invested budget and long-only box, to the milestone's exact tolerances.
    assert abs(float(weights["weight"].sum()) - 1.0) <= 1e-6
    assert (weights["weight"] >= -1e-8).all()
    assert (weights["weight"] <= 1 + 1e-8).all()


def test_weights_match_python_min_volatility_not_max_sharpe(tmp_path):
    """RG-6 cross-check with teeth: R must agree with Python's min-vol portfolio within 1%
    absolute AND disagree with the max-Sharpe one (>5% somewhere on this panel, where the
    two differ by ~0.19) — proving the R script solves the stated objective, not a
    different objective that happens to sum to 1."""
    wide = _four_asset_panel()
    data_dir, out_dir = _write_fixture(tmp_path, wide)
    proc = _run(data_dir, out_dir)
    assert proc.returncode == 0, proc.stderr

    weights = _read_weights(data_dir)
    w_r = pd.Series(weights["weight"].to_numpy(), index=list(weights["ticker"]))

    w_mv = optimize_weights(wide, {"objective": "min_volatility"}).reindex(w_r.index)
    assert (w_r - w_mv).abs().max() < 0.01, (
        f"R min-vol weights diverge from Python:\nR:      {w_r.to_dict()}\nPython: {w_mv.to_dict()}"
    )

    w_ms = optimize_weights(wide, {"objective": "max_sharpe"}).reindex(w_r.index)
    assert (w_r - w_ms).abs().max() > 0.05, (
        "R weights match the max-Sharpe portfolio — wrong objective is being solved"
    )


def test_two_asset_closed_form_minimum_variance(tmp_path):
    """The milestone's hand-checkable case: 2 assets, interior optimum, weights must match
    w1 = (s2^2 - rho*s1*s2) / (s1^2 + s2^2 - 2*rho*s1*s2) at the REALIZED sample moments
    (the formula is scale-invariant, so sample-vs-population std and daily-vs-annualized
    covariance all cancel). Neither library computed this number — both must land on it."""
    rng = np.random.default_rng(7)
    n = 120
    a = rng.normal(0.0004, 0.010, size=n)
    b = rng.normal(0.0006, 0.018, size=n)
    wide = pd.DataFrame({"AAA": a, "BBB": b}, index=pd.bdate_range("2023-01-02", periods=n))
    data_dir, out_dir = _write_fixture(tmp_path, wide)
    proc = _run(data_dir, out_dir)
    assert proc.returncode == 0, proc.stderr

    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    rho = float(np.corrcoef(a, b)[0, 1])
    w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    assert 0.05 < w1 < 0.95, "fixture must have an interior optimum for this test to bite"

    weights = _read_weights(data_dir)
    got = dict(zip(weights["ticker"], weights["weight"]))
    assert abs(got["AAA"] - w1) < 0.01, f"R w1 {got['AAA']} != closed form {w1}"
    assert abs(got["BBB"] - (1.0 - w1)) < 0.01
    assert abs(got["AAA"] + got["BBB"] - 1.0) <= 1e-6


def test_long_only_box_binds_on_leveraged_clone(tmp_path):
    """Adversarial: HI = 2*LO + small noise. Unconstrained min-variance wants HI short
    (w_HI ~ -0.96); the component-07 box must clip the solution to the corner (0, 1).
    A script that forgot the box (or applied (-1, 1)) fails here loudly, yet would have
    passed every interior-optimum fixture above."""
    rng = np.random.default_rng(11)
    n = 200
    lo = rng.normal(0.0005, 0.010, size=n)
    hi = 2 * lo + rng.normal(0.0, 0.002, size=n)
    wide = pd.DataFrame({"HI": hi, "LO": lo}, index=pd.bdate_range("2023-01-02", periods=n))
    data_dir, out_dir = _write_fixture(tmp_path, wide)
    proc = _run(data_dir, out_dir)
    assert proc.returncode == 0, proc.stderr

    weights = _read_weights(data_dir)
    got = dict(zip(weights["ticker"], weights["weight"]))
    assert got["HI"] >= -1e-8, "long-only box violated: negative weight crossed the boundary"
    assert got["HI"] < 0.01, f"box should pin the leveraged clone to ~0, got {got['HI']}"
    assert abs(got["LO"] - 1.0) < 0.01
    # Python must agree on the corner too (both under the same constraints).
    w_py = optimize_weights(wide, {"objective": "min_volatility"})
    assert abs(got["HI"] - w_py["HI"]) < 0.01
    assert abs(got["LO"] - w_py["LO"]) < 0.01


def test_weights_deterministic_across_runs(tmp_path):
    """ROI/quadprog is a deterministic QP solve: two runs on one fixture must produce
    identical weights (a stochastic method like DEoptim/random would differ and make the
    ~1% cross-check tolerance meaningless)."""
    wide = _four_asset_panel()
    results = []
    for tag in ("run1", "run2"):
        root = tmp_path / tag
        root.mkdir()
        data_dir, out_dir = _write_fixture(root, wide)
        proc = _run(data_dir, out_dir)
        assert proc.returncode == 0, proc.stderr
        results.append(_read_weights(data_dir))
    pd.testing.assert_frame_equal(results[0], results[1])


# --- failure paths: broken asset_returns hand-offs must exit non-zero with a diagnostic
# --- and leave NO artifact (png, metrics, weights) behind ---------------------------------


def _assert_failed_cleanly(proc, data_dir: Path, out_dir: Path, *needles: str) -> None:
    assert proc.returncode != 0, "script must exit non-zero on a broken hand-off"
    for needle in needles:
        assert needle in proc.stderr, (
            f"diagnostic should name the problem ({needle!r}); stderr was:\n{proc.stderr}"
        )
    for artifact in (
        out_dir / "tearsheet.png",
        data_dir / "metrics_r.parquet",
        data_dir / "weights_r.parquet",
    ):
        assert not artifact.exists(), f"failed run left an artifact behind: {artifact.name}"


def test_missing_asset_returns_exits_nonzero_no_artifacts(tmp_path):
    """asset_returns.parquet is REQUIRED now: absent -> non-zero, message names the file,
    and crucially NOT EVEN the tearsheet/metrics are produced (fail before any output)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_portfolio_returns(data_dir)
    out_dir = tmp_path / "out"
    proc = _run(data_dir, out_dir)
    _assert_failed_cleanly(proc, data_dir, out_dir, "asset_returns.parquet")


def test_ragged_panel_na_after_pivot_exits_nonzero(tmp_path):
    """One (date, ticker) pair removed -> an NA cell after pivot -> rejected, like the
    Python optimizer's NaN guard (a ragged panel is a data bug, not an optimizer input)."""
    wide = _four_asset_panel().iloc[:60]
    data_dir, out_dir = _write_fixture(tmp_path, wide)
    long = read_frame(str(data_dir / "asset_returns.parquet"), "asset_returns")
    write_frame(
        long.iloc[1:].reset_index(drop=True),  # drop one row -> hole in the panel
        str(data_dir / "asset_returns.parquet"),
        "asset_returns",
    )
    proc = _run(data_dir, out_dir)
    _assert_failed_cleanly(proc, data_dir, out_dir, "NA")


def test_single_ticker_exits_nonzero(tmp_path):
    """<2 assets: nothing to allocate across — same rule as _validate_returns_panel."""
    rng = np.random.default_rng(3)
    wide = pd.DataFrame(
        {"ONLY": rng.normal(0.0, 0.01, size=30)},
        index=pd.bdate_range("2024-01-02", periods=30),
    )
    data_dir, out_dir = _write_fixture(tmp_path, wide)
    proc = _run(data_dir, out_dir)
    _assert_failed_cleanly(proc, data_dir, out_dir, "2 tickers")


def test_duplicate_date_ticker_exits_nonzero(tmp_path):
    """A duplicated (date, ticker) row cannot be pivoted; keeping either value silently
    would hide an upstream bug (same reason interchange.to_wide raises)."""
    wide = _four_asset_panel().iloc[:40]
    data_dir, out_dir = _write_fixture(tmp_path, wide)
    long = read_frame(str(data_dir / "asset_returns.parquet"), "asset_returns")
    dup = pd.concat([long, long.iloc[[0]]], ignore_index=True)
    write_frame(dup, str(data_dir / "asset_returns.parquet"), "asset_returns")
    proc = _run(data_dir, out_dir)
    _assert_failed_cleanly(proc, data_dir, out_dir, "duplicate")


def test_integer_ret_dtype_exits_nonzero(tmp_path):
    """ret stored as int64 violates float64; must fail at the boundary. Built with raw
    pyarrow because write_frame would (rightly) refuse to produce this file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_portfolio_returns(data_dir)
    n = 6
    table = pa.table(
        {
            "date": pa.array(_utc_ns(n), type=pa.timestamp("ns", tz="UTC")),
            "ticker": pa.array(["A", "B"] * 3, type=pa.string()),
            "ret": pa.array([1, 0, 2, 1, 0, 1], type=pa.int64()),
        }
    )
    pq.write_table(table, data_dir / "asset_returns.parquet")
    out_dir = tmp_path / "out"
    proc = _run(data_dir, out_dir)
    _assert_failed_cleanly(proc, data_dir, out_dir, "ret")


def test_misordered_asset_columns_exit_nonzero(tmp_path):
    """Column ORDER is part of the contract (interchange validates it too): a file with
    (ticker, date, ret) must be rejected even though the column SET is correct."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_portfolio_returns(data_dir)
    n = 6
    table = pa.table(
        {
            "ticker": pa.array(["A", "B"] * 3, type=pa.string()),
            "date": pa.array(_utc_ns(n), type=pa.timestamp("ns", tz="UTC")),
            "ret": pa.array([0.01, 0.02, -0.01, 0.0, 0.005, -0.002], type=pa.float64()),
        }
    )
    pq.write_table(table, data_dir / "asset_returns.parquet")
    out_dir = tmp_path / "out"
    proc = _run(data_dir, out_dir)
    _assert_failed_cleanly(proc, data_dir, out_dir, "columns")
