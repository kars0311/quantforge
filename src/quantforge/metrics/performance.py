"""Performance metrics (REFERENCE IMPLEMENTATION).

Single source of truth for metrics — the UI, BacktestResult.metrics, and the R cross-check all use
these definitions. Annualization factor and risk-free assumption are explicit below.

Extend (Sortino, Calmar, rolling stats, etc.) following this pattern. The R layer (analytics_r/) must
reproduce these within tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252          # trading days/year
RISK_FREE = 0.0    # simplification; document if you change it

_KEYS = ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "hit_rate"]


def compute_metrics(returns) -> dict[str, float]:
    """Headline metrics from a series of periodic (daily) returns."""
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return {k: float("nan") for k in _KEYS}

    n = len(r)
    total_return = float((1.0 + r).prod() - 1.0)
    cagr = float((1.0 + total_return) ** (ANN / n) - 1.0)
    std = float(r.std(ddof=0))
    ann_vol = float(std * np.sqrt(ANN))
    excess = r.mean() - RISK_FREE / ANN
    sharpe = float(excess / std * np.sqrt(ANN)) if std > 0 else float("nan")
    equity = (1.0 + r).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    hit_rate = float((r > 0).mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
    }
