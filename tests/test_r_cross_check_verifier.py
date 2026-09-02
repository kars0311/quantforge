"""Verifier suite for week-6 milestone: the RG-6 Python-side cross-check (test_r_cross_check.py).

Independent, adversarial re-proof of the milestone's claims. The builder's suite compares
tearsheet.R against ``compute_metrics`` / ``optimize_weights``; if a bug were SHARED between the
transcription and its reference, that comparison would still "agree". This suite breaks the
symmetry three ways, on the same real-pipeline hand-off:

1. Hand-numpy metrics: R's metrics_r.parquet values are re-derived here with plain numpy
   arithmetic on the series read back FROM THE PARQUET FILE — no performance.py, no tearsheet.R
   formulas — so a bug shared by both transcriptions cannot hide.
2. Optimality, not just agreement: R's min-variance weights must achieve annualized vol no
   worse than every corner portfolio and the equal-weight portfolio under the same annualized
   sample covariance — an argmin property no "two solvers share a setup bug" story satisfies.
3. Pipeline integrity: the portfolio series written to the hand-off equals the plain-numpy dot
   product of weights and panel at 1e-12, and every hand-off date lies inside the TRAIN split —
   the fixture's holdout-discipline claim (RG-4) is asserted, not just narrated.
4. Adversarial hand-off: the per-asset file dropped into the returns.parquet slot (a realistic
   pipeline wiring bug — both files come from the same directory) must exit non-zero and leave
   no metrics_r.parquet behind.

Everything is offline and deterministic (seeded synthetic prices, subprocess to a local Rscript,
all writes under pytest tmp paths). Skipped module-wide when Rscript or any package tearsheet.R
attaches is unavailable — the same policy, and the same probe shape, as the milestone suite.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantforge.data.loader import get_split_bounds
from quantforge.engine.python_engine import PythonEngine
from quantforge.interchange import read_frame, to_long, write_frame
from quantforge.metrics.performance import _KEYS
from quantforge.portfolio.optimize import combine_returns, optimize_weights
from quantforge.strategies import STRATEGIES, validate_params

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "analytics_r" / "tearsheet.R"

_STRATEGY_NAMES = ["momentum", "mean_reversion"]


def _r_deps_available() -> bool:
    """Same probe as the milestone suite: Rscript + every package tearsheet.R attaches."""
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
        "RG-6 verifier requires Rscript with arrow, xts, tidyquant, PerformanceAnalytics, "
        "PortfolioAnalytics, ROI and ROI.plugin.quadprog installed"
    ),
)


def _synthetic_prices(n: int = 400, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Seeded synthetic wide prices anchored at the train-split start (smoke-test pattern)."""
    train_start = get_split_bounds()["train"][0]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(train_start, periods=n)
    rets = rng.normal(0.0004, 0.01, size=(n, k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(k)])


@pytest.fixture(scope="session")
def verifier_run(tmp_path_factory):
    """The real pipeline rebuilt INDEPENDENTLY of the milestone suite's fixture.

    Deliberately does not import anything from tests/test_r_cross_check.py: this suite must
    reconstruct the hand-off from the frozen interfaces alone, so a bug in the builder's
    fixture (e.g. comparing R against a different series than the one written to disk) cannot
    propagate here. One Rscript run, session-scoped for the same wall-time reason.
    """
    root = tmp_path_factory.mktemp("r_cross_check_verifier")
    data_dir = root / "data"
    out_dir = root / "out"
    data_dir.mkdir()

    prices = _synthetic_prices()
    engine = PythonEngine()
    streams = {}
    for name in _STRATEGY_NAMES:
        strategy = STRATEGIES[name]()
        positions = strategy.generate_signals(prices, validate_params(name, {}))
        streams[name] = engine.run_backtest(prices, positions, {"cost_bps": 10}).returns

    panel = pd.concat(streams, axis=1, join="inner")
    panel.columns = list(_STRATEGY_NAMES)
    panel = panel.dropna()
    panel.index = panel.index.tz_localize("UTC")

    weights = optimize_weights(panel)
    portfolio = combine_returns(panel, weights)

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
    assert proc.returncode == 0, (
        f"tearsheet.R failed on the verifier's independently-built hand-off "
        f"(rc={proc.returncode}).\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "panel": panel,
        "weights": weights,
        "portfolio": portfolio,
    }


def test_r_metrics_match_hand_numpy_derivation(verifier_run):
    """R's six metrics re-derived with raw numpy on the series read back from the file.

    Three-way check: file -> numpy here, file -> R in tearsheet.R. Neither performance.py's
    compute_metrics nor R's transcription of it participates, so a bug common to both (the
    one failure mode the milestone suite cannot see) fails here. 1e-9 relative-with-floor
    matches the transcription bound the milestone asserts.
    """
    on_disk = read_frame(str(verifier_run["data_dir"] / "returns.parquet"), "returns")
    r = on_disk["ret"].to_numpy(dtype="float64")
    n = len(r)
    assert n > 0

    total_return = float(np.prod(1.0 + r) - 1.0)
    cagr = float((1.0 + total_return) ** (252.0 / n) - 1.0)
    std = float(np.std(r))  # numpy default ddof=0 — the population-std convention
    ann_vol = float(std * np.sqrt(252.0))
    sharpe = float(np.mean(r) / std * np.sqrt(252.0)) if std > 0 else float("nan")
    equity = np.cumprod(1.0 + r)
    max_drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0))
    hit_rate = float(np.mean(r > 0))
    hand = {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
    }

    metrics_r = read_frame(str(verifier_run["data_dir"] / "metrics_r.parquet"), "metrics")
    assert list(metrics_r["name"]) == _KEYS
    got = dict(zip(metrics_r["name"], metrics_r["value"]))
    for key in _KEYS:
        h, g = hand[key], float(got[key])
        if np.isnan(h) or np.isnan(g):
            assert np.isnan(h) and np.isnan(g), f"{key}: NaN mismatch (hand={h}, r={g})"
            continue
        rel = abs(g - h) / max(abs(h), 1.0)
        assert rel <= 1e-9, f"{key}: R diverges from the hand-numpy value (hand={h!r}, r={g!r})"


def test_handoff_portfolio_is_exact_weighted_blend_on_train_dates(verifier_run):
    """Pipeline integrity of the hand-off itself.

    The portfolio series the cross-check judges must BE the weighted blend of the panel —
    asserted against a plain-numpy dot product at 1e-12, no combine_returns — and every
    hand-off date must sit inside the train split, so the cross-check provably never touched
    validation or holdout data (RG-4). A fixture that quietly drifted from either property
    would make the whole RG-6 comparison a statement about the wrong series.
    """
    panel = verifier_run["panel"]
    weights = verifier_run["weights"]
    blend = panel.to_numpy() @ weights.reindex(panel.columns).to_numpy()
    diff = np.abs(verifier_run["portfolio"].to_numpy() - blend)
    assert float(diff.max()) < 1e-12

    on_disk = read_frame(str(verifier_run["data_dir"] / "returns.parquet"), "returns")
    np.testing.assert_allclose(
        on_disk["ret"].to_numpy(), verifier_run["portfolio"].to_numpy(), rtol=0, atol=0
    )

    train_start, train_end = (pd.Timestamp(b, tz="UTC") for b in get_split_bounds()["train"])
    dates = pd.DatetimeIndex(on_disk["date"])
    assert dates.min() >= train_start and dates.max() <= train_end, (
        f"hand-off dates [{dates.min()}, {dates.max()}] leave the train split "
        f"[{train_start}, {train_end}]"
    )


def test_r_weights_are_argmin_not_just_close_to_python(verifier_run):
    """R's weights beat every corner portfolio and equal-weight under the same covariance.

    'R agrees with Python' is vacuous if both optimizers share a setup bug; the argmin
    property is not. Under the annualized sample covariance (the estimator both sides use),
    the min-variance optimum must achieve vol <= that of each single-asset corner and of the
    equal-weight portfolio — all feasible under full-investment long-only constraints. The
    1e-6 slack covers solver/rounding precision only.
    """
    panel = verifier_run["panel"]
    weights_r_df = read_frame(str(verifier_run["data_dir"] / "weights_r.parquet"), "weights")
    w_r = pd.Series(
        weights_r_df["weight"].to_numpy(), index=list(weights_r_df["ticker"]), dtype=float
    )
    assert abs(float(w_r.sum()) - 1.0) < 1e-6
    assert (w_r >= -1e-8).all() and (w_r <= 1 + 1e-8).all()

    cols = list(panel.columns)
    sigma_ann = panel.cov().loc[cols, cols].to_numpy() * 252.0

    def vol(vec: np.ndarray) -> float:
        return float(np.sqrt(vec @ sigma_ann @ vec))

    v_r = vol(w_r.reindex(cols).to_numpy())
    k = len(cols)
    rivals = [np.full(k, 1.0 / k)] + [np.eye(k)[i] for i in range(k)]
    for rival in rivals:
        assert v_r <= vol(rival) + 1e-6, (
            f"R weights are not a minimum: vol {v_r!r} beats neither rival {rival} "
            f"(vol {vol(rival)!r}) — the QP setup, not solver precision, is wrong"
        )


def test_swapped_handoff_files_exit_nonzero_without_artifacts(verifier_run, tmp_path):
    """Adversarial wiring bug: the per-asset file in the returns.parquet slot.

    Both hand-off files live in one directory; a pipeline stage writing the wrong frame to
    the wrong name is a realistic mistake. The 3-column asset_returns frame violates the
    2-column returns contract, so tearsheet.R must exit non-zero with a diagnostic and leave
    no metrics_r.parquet behind (a partial artifact would poison the cross-check downstream).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(verifier_run["data_dir"] / "asset_returns.parquet", data_dir / "returns.parquet")
    shutil.copy(
        verifier_run["data_dir"] / "asset_returns.parquet", data_dir / "asset_returns.parquet"
    )
    proc = subprocess.run(
        ["Rscript", str(_SCRIPT), str(data_dir), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode != 0, "tearsheet.R accepted an asset_returns frame as returns"
    assert "returns.parquet" in proc.stderr
    assert not (data_dir / "metrics_r.parquet").exists()
    assert not (data_dir / "weights_r.parquet").exists()
