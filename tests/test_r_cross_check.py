"""RG-6 Python-side cross-check: the REAL Python pipeline vs analytics_r/tearsheet.R (week 6).

The other R suites (test_r_tearsheet_*.py) feed tearsheet.R hand-built fixtures to probe its
boundary behavior. This suite is the actual RG-6 claim from docs/components/08-r-tearsheet.md:
run the GENUINE Python pipeline — STRATEGIES registry -> PythonEngine -> optimize_weights ->
combine_returns — write the interchange hand-off exactly as production would (AR-2), shell out to
``Rscript tearsheet.R``, and assert that R's independently-computed metrics and PortfolioAnalytics
optimum agree with Python's:

- metrics within relative **1%** — the RG-6 contract (docs/product.md decision 5) — AND within
  **1e-9**, because all six ``_KEYS`` in tearsheet.R are exact transcriptions of
  ``metrics/performance.py`` (same annualization, same population std), so anything beyond float
  noise means a formula drifted. The loose bound documents the contract; the tight bound has the
  teeth. NaN == NaN is honored for the degenerate zero-vol sharpe case.
- optimizer weights within **0.01 absolute per asset** (Python min-vol vs PortfolioAnalytics/ROI
  min-StdDev — a convex QP with one global optimum both libraries must find), and the achieved
  annualized vol ``sqrt(252 * w' Sigma w)`` of the two weight vectors within 1% relative, so a
  weight disagreement that happens to sit in a flat region of the objective is still caught.

Everything is offline and deterministic: seeded synthetic prices (the smoke-test generator
pattern), no network, all files written under pytest tmp paths — the repo's ``data_cache/`` is
never touched. Skipped module-wide (with reason) when Rscript or any R package tearsheet.R
attaches is unavailable, so CI without R skips cleanly instead of going red.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantforge.data.loader import get_split_bounds
from quantforge.engine.python_engine import PythonEngine
from quantforge.interchange import read_frame, to_long, write_frame
from quantforge.metrics.performance import _KEYS, compute_metrics
from quantforge.portfolio.optimize import combine_returns, optimize_weights
from quantforge.strategies import STRATEGIES, validate_params

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "analytics_r" / "tearsheet.R"

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: The two vetted strategies whose blend is "the portfolio" (FR-4), in panel column order.
_STRATEGY_NAMES = ["momentum", "mean_reversion"]

# Tolerances asserted below, named once so the docstring, the assertions, and the RG-6 writeup
# all refer to the same numbers.
_RG6_REL = 0.01  # the contract: ~1% relative (docs/product.md decision 5)
_TRANSCRIPTION_REL = 1e-9  # what exact transcriptions of performance.py actually achieve
_WEIGHT_ABS = 0.01  # per-asset absolute weight agreement
_VOL_REL = 0.01  # achieved annualized vol agreement between the two weight vectors


def _r_deps_available() -> bool:
    """One subprocess probe: Rscript on PATH AND every package tearsheet.R attaches loads.

    ``requireNamespace`` returns FALSE (it does not error) on a missing package, and a bare
    Rscript call always exits 0 — so the probe must translate FALSE into a nonzero exit itself.
    Probed once at import time (module-level skipif), so a CI box without R skips the whole
    module in one shot instead of timing out per test.
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
    reason=(
        "RG-6 cross-check requires Rscript with arrow, xts, tidyquant, PerformanceAnalytics, "
        "PortfolioAnalytics, ROI and ROI.plugin.quadprog installed"
    ),
)


def _synthetic_prices(n: int = 400, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Seeded synthetic wide prices — the same generator pattern as the smoke test.

    Anchored at the TRAIN split start (from get_split_bounds, the single source of truth) so
    the whole cross-check provably runs on train-only dates: even a test fixture respects the
    holdout discipline (RG-4). Deterministic seed, zero network.
    """
    train_start = get_split_bounds()["train"][0]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(train_start, periods=n)
    rets = rng.normal(0.0004, 0.01, size=(n, k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(k)])


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory):
    """Run the real pipeline once, write the hand-off, run tearsheet.R once; share artifacts.

    Session scope (why): the pipeline is deterministic end to end (seeded prices, deterministic
    engine and QP solvers) and the R subprocess is the slowest step in the whole suite —
    re-running it per test would multiply wall time without adding independence. Tests treat
    every artifact as read-only. All paths come from ``tmp_path_factory``: nothing under the
    repo (in particular ``data_cache/``) is written.
    """
    root = tmp_path_factory.mktemp("r_cross_check")
    data_dir = root / "data"
    out_dir = root / "out"
    data_dir.mkdir()

    # --- the real Python pipeline, through the frozen interfaces only (AR-1) ---------------
    prices = _synthetic_prices()
    engine = PythonEngine()
    results = {}
    for name in _STRATEGY_NAMES:
        params = validate_params(name, {})  # {} -> whitelisted defaults, the vetted path
        strategy = STRATEGIES[name]()  # registry-driven construction, as the AI layer uses
        positions = strategy.generate_signals(prices, params)
        results[name] = engine.run_backtest(prices, positions, {"cost_bps": 10})

    # Wide panel of the two strategy return streams. Inner join + dropna are identities for
    # engine streams (same index, NaN-free warmup zeros) but state the alignment rule.
    panel = pd.concat(
        {name: results[name].returns for name in _STRATEGY_NAMES}, axis=1, join="inner"
    )
    panel.columns = list(_STRATEGY_NAMES)
    panel = panel.dropna()
    # The engine works on tz-naive business days; the interchange contract requires tz-aware
    # UTC. Localizing (not converting) at the boundary is lossless and mirrors production.
    panel.index = panel.index.tz_localize("UTC")

    weights = optimize_weights(panel)  # default objective — the portfolio the UI would show
    portfolio = combine_returns(panel, weights)

    # --- the Parquet hand-off, exactly as production would write it (AR-2) -----------------
    returns_df = pd.DataFrame(
        {"date": pd.Series(portfolio.index), "ret": portfolio.to_numpy(dtype="float64")}
    )
    write_frame(returns_df, str(data_dir / "returns.parquet"), "returns")
    write_frame(
        to_long(panel, "asset_returns"), str(data_dir / "asset_returns.parquet"), "asset_returns"
    )

    proc = subprocess.run(
        ["Rscript", str(_SCRIPT), str(data_dir), str(out_dir)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_REPO_ROOT,
    )
    return {
        "proc": proc,
        "data_dir": data_dir,
        "out_dir": out_dir,
        "panel": panel,
        "portfolio": portfolio,
    }


def _require_r_success(run: dict) -> None:
    """Precondition for the agreement tests: name the R failure instead of a FileNotFoundError."""
    proc = run["proc"]
    assert proc.returncode == 0, (
        f"tearsheet.R failed (rc={proc.returncode}) — agreement tests cannot run.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def _rel_diff(py: float, r: float) -> float:
    """Relative difference |r - py| / |py|, with |py| floored at 1 so a metric that is
    legitimately ~0 (possible on another seed) doesn't blow the ratio up over float noise."""
    return abs(r - py) / max(abs(py), 1.0)


# ---------------------------------------------------------------------------
# (1) The subprocess contract: exit 0 on a production-shaped hand-off
# ---------------------------------------------------------------------------


def test_rscript_exits_zero_on_real_pipeline_handoff(pipeline_run):
    """tearsheet.R accepts the exact files the Python pipeline writes and exits 0 — the CLI
    contract of component 08 holds against real (not hand-built) inputs."""
    proc = pipeline_run["proc"]
    assert proc.returncode == 0, (
        f"tearsheet.R rejected the real pipeline's hand-off (rc={proc.returncode}).\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# (2) Metrics agreement — the heart of RG-6
# ---------------------------------------------------------------------------


def test_metrics_agree_with_python_reference(pipeline_run):
    """R's metrics_r.parquet carries exactly performance._KEYS, in order, and every value
    matches compute_metrics on the same in-memory series.

    Two bounds per metric, both asserted:
    - 1% relative: the RG-6 contract (docs/product.md decision 5) — what the writeup promises.
    - 1e-9 relative: all six formulas in tearsheet.R are line-for-line transcriptions of
      performance.py (ANN=252, risk-free 0, population std), so agreement must be float-exact;
      a drift inside 1% but outside 1e-9 (e.g. R's sample std sneaking back in, ~0.125% at
      n=400) means a transcription broke even though the headline contract still "passes".
    NaN == NaN: sharpe is NaN by definition when std == 0; a degenerate series must agree on
    that too, not fail an arithmetic comparison.
    """
    _require_r_success(pipeline_run)
    metrics_r = read_frame(str(pipeline_run["data_dir"] / "metrics_r.parquet"), "metrics")

    # Names AND order are the contract — a swapped pair of rows must fail here, not average out.
    assert list(metrics_r["name"]) == _KEYS, (
        f"metric names/order diverge from performance._KEYS:\n"
        f"expected {_KEYS}\ngot      {list(metrics_r['name'])}"
    )

    py = compute_metrics(pipeline_run["portfolio"])
    r_values = dict(zip(metrics_r["name"], metrics_r["value"]))
    for key in _KEYS:
        py_v, r_v = py[key], float(r_values[key])
        if np.isnan(py_v) or np.isnan(r_v):
            # Degenerate case (zero-vol sharpe): both sides must agree it is undefined.
            assert np.isnan(py_v) and np.isnan(r_v), (
                f"{key}: one side is NaN, the other is not (python={py_v}, r={r_v})"
            )
            continue
        rel = _rel_diff(py_v, r_v)
        assert rel <= _RG6_REL, (
            f"{key}: RG-6 contract broken — relative diff {rel:.3e} > {_RG6_REL} "
            f"(python={py_v!r}, r={r_v!r})"
        )
        assert rel <= _TRANSCRIPTION_REL, (
            f"{key}: transcription drift — relative diff {rel:.3e} > {_TRANSCRIPTION_REL}; "
            f"tearsheet.R no longer mirrors performance.py exactly "
            f"(python={py_v!r}, r={r_v!r})"
        )


# ---------------------------------------------------------------------------
# (3) Optimizer agreement — PortfolioAnalytics vs PyPortfolioOpt on one QP
# ---------------------------------------------------------------------------


def test_optimizer_weights_and_achieved_vol_agree(pipeline_run):
    """R's min-StdDev optimum matches Python's min_volatility optimum, two ways.

    Why min-vol is the comparable objective: it is a convex QP with a unique global optimum
    under the shared constraints (full investment, long-only box), so two correct solvers MUST
    land on the same point; max-Sharpe internals legitimately differ between libraries.

    Per-asset weights within 0.01 absolute catches a solver landing somewhere else; the
    achieved annualized vol sqrt(252 * w' Sigma w) of BOTH weight vectors within 1% relative
    catches the complementary failure — near-collinear streams can make the objective flat
    enough that 0.01-different weights hide a materially different portfolio, and vice versa.
    Sigma is the panel's daily sample covariance (ddof=1), the same estimator optimize.py uses
    (risk_models.sample_cov is exactly panel.cov() * 252), so the vol comparison judges both
    weight vectors on Python's own risk model rather than a third convention.
    """
    _require_r_success(pipeline_run)
    panel = pipeline_run["panel"]

    weights_r_df = read_frame(str(pipeline_run["data_dir"] / "weights_r.parquet"), "weights")
    w_r = pd.Series(
        weights_r_df["weight"].to_numpy(), index=list(weights_r_df["ticker"]), dtype=float
    )
    assert sorted(w_r.index) == sorted(panel.columns), (
        f"R weights cover tickers {sorted(w_r.index)}, panel has {sorted(panel.columns)}"
    )

    w_py = optimize_weights(panel, {"objective": "min_volatility"})
    w_py = w_py.reindex(w_r.index)

    diff = (w_r - w_py).abs()
    assert float(diff.max()) < _WEIGHT_ABS, (
        f"per-asset weight disagreement {float(diff.max()):.4f} >= {_WEIGHT_ABS}:\n"
        f"R:      {w_r.to_dict()}\nPython: {w_py.to_dict()}"
    )

    # Achieved annualized volatility of each weight vector under the SAME covariance.
    sigma = panel.cov()  # daily sample cov (ddof=1) — optimize.py's estimator, un-annualized
    order = list(w_r.index)
    s = sigma.loc[order, order].to_numpy()

    def ann_vol(w: pd.Series) -> float:
        v = w.to_numpy(dtype=float)
        return float(np.sqrt(252.0 * v @ s @ v))

    vol_r, vol_py = ann_vol(w_r), ann_vol(w_py)
    rel = abs(vol_r - vol_py) / vol_py
    assert rel <= _VOL_REL, (
        f"achieved annualized vol diverges by {rel:.3e} > {_VOL_REL} "
        f"(R weights -> {vol_r!r}, Python weights -> {vol_py!r})"
    )


# ---------------------------------------------------------------------------
# (4) The visual deliverable exists
# ---------------------------------------------------------------------------


def test_tearsheet_png_exists_and_nonempty(pipeline_run):
    """The tearsheet PNG (FR-6's visual deliverable) is produced, non-empty, and actually a
    PNG — a zero-byte file from a crashed device would satisfy exists() alone."""
    _require_r_success(pipeline_run)
    png = pipeline_run["out_dir"] / "tearsheet.png"
    assert png.exists(), "tearsheet.png was not produced"
    data = png.read_bytes()
    assert len(data) > 0, "tearsheet.png is empty"
    assert data[:8] == _PNG_MAGIC, "tearsheet.png does not start with the PNG signature"


# ---------------------------------------------------------------------------
# (5) Failure paths: a broken hand-off must exit non-zero, not fabricate output
# ---------------------------------------------------------------------------


def _run_r(data_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["Rscript", str(_SCRIPT), str(data_dir), str(out_dir)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_REPO_ROOT,
    )


def test_empty_data_dir_exits_nonzero(tmp_path):
    """No hand-off at all: the script must fail loudly (missing returns.parquet), because a
    zero-exit here would let a broken pipeline stage look green downstream."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    proc = _run_r(data_dir, tmp_path / "out")
    assert proc.returncode != 0, "tearsheet.R exited 0 on an empty data_dir"
    assert "returns.parquet" in proc.stderr, (
        f"diagnostic should name the missing file; stderr was:\n{proc.stderr}"
    )


def test_misnamed_returns_column_exits_nonzero(tmp_path):
    """A returns file with 'return' instead of 'ret' violates the interchange contract and
    must be rejected at R's input boundary. Built with raw pyarrow because write_frame would
    (correctly) refuse to produce such a file — the test forges what a buggy foreign writer
    could."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n = 30
    dates = pd.Series(pd.bdate_range("2024-01-02", periods=n)).dt.tz_localize("UTC")
    table = pa.table(
        {
            "date": pa.array(dates, type=pa.timestamp("ns", tz="UTC")),
            "return": pa.array(np.full(n, 0.001), type=pa.float64()),
        }
    )
    pq.write_table(table, data_dir / "returns.parquet")
    proc = _run_r(data_dir, tmp_path / "out")
    assert proc.returncode != 0, "tearsheet.R exited 0 on a misnamed returns column"
    assert proc.stderr.strip(), "a rejected hand-off should carry a diagnostic on stderr"
