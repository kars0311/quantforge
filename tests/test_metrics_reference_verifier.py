"""Independent verification of the AR-3 metric pins (adversarial pass on milestone wk-3).

WHY a second file: tests/test_metrics_reference.py recomputes each metric in-test with
numpy from the same literals. That catches a change in performance.py, but a correlated
drift (someone "fixing" the module and the test's numpy recomputation together) would
slip through. Here every expected value is a HARD LITERAL, hand-computed offline with
pure Python (math.prod / manual variance loop — no numpy, no pandas), so the constants
below cannot silently track any code change. Plus adversarial edges the milestone file
does not cover: zero-variance input (division-by-zero path in sharpe), all-NaN input,
a single observation, and a monotone series with exactly-zero drawdown.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from quantforge.metrics.performance import _KEYS, compute_metrics

# Same literal fixture as the milestone file — but the expected values below were
# computed OFFLINE with pure Python (math.prod, a manual population-variance loop,
# and an explicit running-peak drawdown scan) and pasted in as 17-digit literals.
RETURNS = [0.02, -0.01, 0.0, 0.03, -0.04, 0.01, 0.0, -0.02]

# Literal pins (population std, ANN=252, RISK_FREE=0, strict `r > 0` hit rate):
EXPECTED = {
    "total_return": -0.011694360447999985,  # prod(1+r) - 1
    "cagr": -0.3096407984997068,  # (1+TR)^(252/8) - 1
    "ann_vol": 0.3314456969097653,  # sqrt(sum((r-mean)^2)/8) * sqrt(252)
    "sharpe": -0.9503819266229833,  # mean(r) / pop_std(r) * sqrt(252)
    "max_drawdown": -0.04979200000000006,  # trough after the +3% peak (days 5-8)
    "hit_rate": 0.375,  # 3 of 8 days strictly > 0
}


def test_metrics_match_offline_hand_computed_literals():
    out = compute_metrics(pd.Series(RETURNS))
    for key in _KEYS:
        assert out[key] == pytest.approx(EXPECTED[key], rel=1e-12), key


def test_zero_variance_series_yields_nan_sharpe_not_a_crash():
    # Adversarial: constant returns make std == 0. The documented sharpe formula
    # divides by std, so the module must return NaN (its explicit `std > 0` guard),
    # never raise ZeroDivisionError or return +/-inf — the R layer and UI both
    # consume this dict blindly. All other keys stay finite and well-defined.
    out = compute_metrics(pd.Series([0.01] * 5))
    assert math.isnan(out["sharpe"])
    assert out["ann_vol"] == 0.0
    assert out["total_return"] == pytest.approx(1.01**5 - 1.0, rel=1e-12)
    assert out["max_drawdown"] == 0.0  # monotone equity: exactly zero, not negative
    assert out["hit_rate"] == 1.0
    assert list(out.keys()) == _KEYS


def test_all_nan_input_behaves_like_empty():
    # Adversarial: a series that is non-empty but contains NO usable data must take
    # the empty path (all keys NaN), not compute metrics on zero observations.
    out = compute_metrics(pd.Series([float("nan")] * 4))
    assert list(out.keys()) == _KEYS
    assert all(math.isnan(v) for v in out.values())


def test_single_observation():
    # n=1: population std is 0 -> sharpe NaN; the other metrics are still exact.
    out = compute_metrics(pd.Series([0.02]))
    assert out["total_return"] == pytest.approx(0.02, rel=1e-12)
    assert out["cagr"] == pytest.approx(1.02**252 - 1.0, rel=1e-12)
    assert math.isnan(out["sharpe"])
    assert out["max_drawdown"] == 0.0
    assert out["hit_rate"] == 1.0


def test_all_zero_days_hit_rate_is_zero():
    # Strict `r > 0` at its boundary: a series of only flat days has hit_rate 0.0
    # (with `>=` it would be 1.0 — the loudest possible failure for that mutation).
    out = compute_metrics(pd.Series([0.0] * 6))
    assert out["hit_rate"] == 0.0
