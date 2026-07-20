"""Smoke test for the working vertical slice (offline, synthetic data — no network/heavy deps).

Proves loader-free: momentum -> PythonEngine -> metrics runs end-to-end with sane outputs. This is the
pattern to extend; keep it green.
"""

import numpy as np
import pandas as pd

from quantforge.engine.python_engine import PythonEngine
from quantforge.metrics.performance import compute_metrics
from quantforge.strategies.momentum import MomentumStrategy

_KEYS = ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "hit_rate"]


def _synthetic_prices(n: int = 400, k: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    rets = rng.normal(0.0004, 0.01, size=(n, k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(k)])


def test_vertical_slice_runs_end_to_end():
    prices = _synthetic_prices()
    positions = MomentumStrategy().generate_signals(prices, {"lookback": 60, "top_n": 2})

    assert positions.shape == prices.shape
    # long-only, fully-invested-or-less, with warmup zeros
    assert (positions.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (positions >= -1e-12).all().all()

    result = PythonEngine().run_backtest(prices, positions, {"cost_bps": 10})
    assert len(result.equity_curve) == len(prices)
    assert not result.returns.isna().any()
    for key in _KEYS:
        assert key in result.metrics
        assert result.metrics[key] == result.metrics[key]  # not NaN


def test_metrics_zero_returns():
    m = compute_metrics(pd.Series([0.0] * 10))
    assert m["total_return"] == 0.0
    assert m["max_drawdown"] == 0.0
    assert m["hit_rate"] == 0.0
