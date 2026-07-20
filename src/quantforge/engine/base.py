"""The swappable-engine seam: the Strategy and Engine interfaces.

Everything downstream (portfolio, metrics, UI, AI agent) depends ONLY on these abstractions, never on
a concrete engine. That is what lets a new backend (R, KNIME, or a future comparison engine) drop in
without touching the rest of the app. Freeze this interface early (week 4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BacktestResult:
    """Standardized result so every engine is interchangeable.

    equity_curve: per-date portfolio value (pandas Series).
    returns:      per-date returns (pandas Series).
    metrics:      dict of headline metrics (Sharpe, max_drawdown, cagr, ...).
    meta:         engine name, params, data window, costs, etc.
    """

    equity_curve: Any
    returns: Any
    metrics: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """A strategy maps prices + params to target positions (weights).

    Must be free of look-ahead: signals at time t may use information up to and including t-? only
    (decide and document the convention; default: information available at the close of t-1).
    """

    name: str = "base"

    @abstractmethod
    def generate_signals(self, prices, params: dict[str, Any]):
        """Return target positions (e.g., a DataFrame: date x ticker -> weight). TODO."""
        raise NotImplementedError


class Engine(ABC):
    """An engine runs a backtest given prices + positions and returns a BacktestResult.

    Concrete engines: PythonEngine (core), KnimeEngine (stretch), future MatlabEngine.
    """

    name: str = "base"

    @abstractmethod
    def run_backtest(self, prices, positions, params: dict[str, Any]) -> BacktestResult:
        """Execute the backtest. Must account for transaction costs and avoid look-ahead. TODO."""
        raise NotImplementedError
