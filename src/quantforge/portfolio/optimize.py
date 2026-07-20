"""Portfolio optimization via PyPortfolioOpt (week 5).

Combine per-strategy return streams into a portfolio: mean-variance weights + the efficient frontier.
The R PortfolioAnalytics optimizer (analytics_r/) independently cross-checks these results.

TODO: implement optimize_weights(returns, params) -> weights, and frontier(returns) -> points.
"""

from __future__ import annotations

from typing import Any


def optimize_weights(returns, params: dict[str, Any]):
    """Mean-variance optimal weights (e.g., max Sharpe). TODO: wrap PyPortfolioOpt EfficientFrontier."""
    raise NotImplementedError


def frontier(returns, n_points: int = 50):
    """Return (risk, return, weights) points along the efficient frontier for the UI. TODO."""
    raise NotImplementedError
