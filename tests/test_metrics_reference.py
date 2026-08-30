"""Pin the `_KEYS` metric definitions of ``metrics/performance.py`` (AR-3, week 3).

WHY this file exists: ``compute_metrics`` is the single source of truth for metric
definitions — the UI, ``BacktestResult.metrics``, the MCP ``get_metrics`` tool, and the
week-6 R tearsheet all consume it, and the R layer must *reproduce* these numbers.
If someone silently changes a definition (e.g. population std -> sample std, or
``r > 0`` -> ``r >= 0``), every downstream consumer drifts and the R cross-check
starts failing for a reason nobody can see.  So each metric is pinned here against
the documented formula (docs/components/06-metrics.md), recomputed in-test with
numpy from a tiny LITERAL returns series, at rel=1e-12 — tight enough that any
convention change on an 8-day series is a loud failure, not tolerance noise.

The fixture series is chosen deliberately:
- contains gains, losses, AND exact-zero days  -> pins hit_rate's strict ``r > 0``
  (with ``>=`` the zero days would count and the value jumps from 3/8 to 5/8);
- n=8 is tiny                                  -> ddof=0 vs ddof=1 differ by
  sqrt(8/7) - 1 ~ 6.9%, so a ddof switch cannot hide inside the tolerance;
- has a clear peak-then-crash                  -> pins a strictly negative
  max_drawdown, not the degenerate 0.0 of a monotone series.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantforge.metrics.performance import _KEYS, compute_metrics

# --- the literal fixture -------------------------------------------------------------
# 8 daily returns: 3 strictly positive, 3 strictly negative, 2 exactly zero.
RETURNS = [0.02, -0.01, 0.0, 0.03, -0.04, 0.01, 0.0, -0.02]
ANN = 252  # must match performance.ANN — the documented annualization factor
RISK_FREE = 0.0  # must match performance.RISK_FREE — documented simplification

REL = 1e-12  # tight relative tolerance: definition changes must fail, fp noise must not


@pytest.fixture()
def metrics() -> dict[str, float]:
    return compute_metrics(pd.Series(RETURNS))


def test_total_return(metrics):
    # total_return = prod(1 + r) - 1  (geometric compounding, not sum of returns)
    r = np.array(RETURNS)
    expected = np.prod(1.0 + r) - 1.0
    assert metrics["total_return"] == pytest.approx(expected, rel=REL)


def test_cagr(metrics):
    # cagr = (1 + total_return)^(ANN / n) - 1, with ANN = 252 trading days/year
    r = np.array(RETURNS)
    total_return = np.prod(1.0 + r) - 1.0
    expected = (1.0 + total_return) ** (ANN / len(r)) - 1.0
    assert metrics["cagr"] == pytest.approx(expected, rel=REL)


def test_ann_vol_uses_population_std(metrics):
    # ann_vol = std(r, ddof=0) * sqrt(ANN) — POPULATION std by contract.
    # np.std defaults to ddof=0, so this recomputation IS the documented formula.
    r = np.array(RETURNS)
    expected = np.std(r) * np.sqrt(ANN)
    assert metrics["ann_vol"] == pytest.approx(expected, rel=REL)

    # Teeth: at n=8 the sample-std (ddof=1) value differs by ~6.9%, far outside REL,
    # so a silent ddof switch in performance.py must fail the assertion above.
    sample_based = np.std(r, ddof=1) * np.sqrt(ANN)
    assert metrics["ann_vol"] != pytest.approx(sample_based, rel=1e-3)


def test_sharpe(metrics):
    # sharpe = (mean(r) - RISK_FREE/ANN) / std(r, ddof=0) * sqrt(ANN)
    # Same ddof=0 sensitivity as ann_vol: a ddof switch shifts sharpe by ~6.9%.
    r = np.array(RETURNS)
    expected = (np.mean(r) - RISK_FREE / ANN) / np.std(r) * np.sqrt(ANN)
    assert metrics["sharpe"] == pytest.approx(expected, rel=REL)


def test_max_drawdown(metrics):
    # max_drawdown = min(equity / cummax(equity) - 1), equity = cumprod(1 + r).
    # Reported as a negative number (a -4% crash after the peak guarantees < 0 here).
    r = np.array(RETURNS)
    equity = np.cumprod(1.0 + r)
    expected = np.min(equity / np.maximum.accumulate(equity) - 1.0)
    assert metrics["max_drawdown"] == pytest.approx(expected, rel=REL)
    assert metrics["max_drawdown"] < 0.0  # known drawdown in the fixture
    assert expected <= 0.0  # the definition can never be positive


def test_hit_rate_strict_inequality(metrics):
    # hit_rate = share of days with r > 0 — STRICTLY positive. The fixture has
    # 3 winners, 3 losers, and 2 flat days: zero days are NOT wins.
    assert metrics["hit_rate"] == pytest.approx(3 / 8, rel=REL)
    # Teeth: with `r >= 0` the two zero days would count and the value becomes 5/8.
    assert metrics["hit_rate"] != pytest.approx(5 / 8, rel=1e-3)


# --- edge cases: the input-handling contract -----------------------------------------


def test_empty_series_returns_all_nan_for_every_key():
    # Empty input -> every _KEYS key present, every value NaN. Downstream consumers
    # (UI, R layer) rely on the key set being stable even when there is no data.
    out = compute_metrics(pd.Series([], dtype=float))
    assert set(out.keys()) == set(_KEYS)
    assert all(math.isnan(v) for v in out.values())


def test_nans_in_input_are_dropped():
    # NaN returns (e.g. pre-listing days) are dropped, not propagated: a series with
    # NaNs interleaved must yield exactly the same metrics as its clean subset.
    dirty = [np.nan, 0.02, -0.01, np.nan, 0.0, 0.03, -0.04, 0.01, 0.0, -0.02, np.nan]
    clean = compute_metrics(pd.Series(RETURNS))
    with_nans = compute_metrics(pd.Series(dirty))
    for key in _KEYS:
        assert with_nans[key] == pytest.approx(clean[key], rel=REL)


def test_returned_key_set_is_exactly_keys():
    # The exact key set IS the cross-language contract for the week-6 R layer:
    # no extra keys (R would miss them), no missing keys (R would KeyError).
    out = compute_metrics(pd.Series(RETURNS))
    assert list(out.keys()) == _KEYS
