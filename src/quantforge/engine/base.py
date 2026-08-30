"""The swappable-engine seam: the Strategy and Engine interfaces.

**FROZEN as of week 4 (2026-08).** Per AR-1, any change to the public surface of this module —
class names, dataclass fields, method signatures, or documented semantics — is a BREAKING CHANGE
that requires an explicit, recorded decision; it is never a casual edit. The freeze is enforced
mechanically by ``tests/test_interface_freeze.py``, which pins every name, field, and parameter
below: if you meant to change the seam, you must change that test too, and that friction is the
point.

WHY freeze at all: everything downstream (portfolio, metrics, UI, AI agent) depends ONLY on these
abstractions, never on a concrete engine. Per AR-4, this seam is the documented "next engine drops
in here" point — a future R, KNIME, or MATLAB backend subclasses ``Engine`` and the rest of the
app is untouched. That only works if the seam stops moving before anyone builds on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BacktestResult:
    """Standardized result so every engine is interchangeable.

    Two engines fed the same inputs must produce BacktestResults that agree within documented
    tolerance — that is what makes an engine swap a non-event downstream. The four fields:

    equity_curve: per-date portfolio value (pandas Series), normalized to START AT 1.0 so curves
                  from different engines/runs are directly comparable.
    returns:      per-date NET returns (pandas Series), i.e. after transaction costs.
    metrics:      dict of headline metrics whose keys are exactly the ``_KEYS`` list in
                  ``metrics/performance.py`` (total_return, cagr, ann_vol, sharpe, max_drawdown,
                  hit_rate) — that module is the single source of truth; engines must not invent
                  their own metric definitions.
    meta:         provenance for reproducibility: engine name, params, cost_bps, data window, etc.
                  Defaults to an empty dict so simple callers need not supply it.
    """

    equity_curve: Any
    returns: Any
    metrics: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """A strategy maps prices + params to target weights: "what I want to hold as of tonight's close".

    Look-ahead discipline is split in two on purpose (RG-1, one enforcement point): the STRATEGY
    may use everything through the close of row t to decide row t's weight; the ENGINE then applies
    the one-day execution lag so that weight only earns returns from t+1 on. Strategies never
    shift; engines always do.
    """

    name: str = "base"

    @abstractmethod
    def generate_signals(self, prices, params: dict[str, Any]):
        """Map prices + params to target portfolio weights.

        Contract (frozen — see module docstring):

        - ``prices``: WIDE pandas DataFrame — index=date, columns=ticker, values=adjusted close.
        - Returns: WIDE DataFrame of TARGET WEIGHTS with the same shape, index, and columns as
          ``prices``.
        - Timing: the weight on row t is decided using information through the CLOSE of t — and
          nothing later. The one-day execution lag is NOT the strategy's job: the engine shifts
          positions so a close-of-t decision earns t+1's return. (Strategies that shift themselves
          would be double-lagged.)
        - Gross exposure: Σ|w| ≤ 1 per row (no leverage).
        - Warmup: rows where the indicator is undefined (e.g., inside the lookback window) must be
          0.0, not NaN — flat until there is enough history to have an opinion.
        - No look-ahead: no same-row-or-later information beyond close-of-t, ever. Truncating the
          tail of ``prices`` must leave earlier rows' weights unchanged.
        """
        raise NotImplementedError


class Engine(ABC):
    """An engine runs a backtest given prices + positions and returns a BacktestResult.

    Concrete engines: PythonEngine (core), KnimeEngine (stretch), future MatlabEngine.
    """

    name: str = "base"

    @abstractmethod
    def run_backtest(self, prices, positions, params: dict[str, Any]) -> BacktestResult:
        """Execute the backtest over wide prices and wide target weights.

        Contract (frozen — see module docstring):

        - The ENGINE applies the one-day execution lag itself (``positions.shift(1)`` or an exact
          equivalent): a weight decided at the close of t earns t+1's return. This lives here, not
          in strategies, so there is exactly one place look-ahead can be enforced and audited.
        - Transaction costs are charged on TURNOVER (change in held weights), so a buy-and-hold
          book pays ~nothing and a churny one pays proportionally.
        - Inputs must NOT be mutated — callers may reuse ``prices``/``positions`` across engines.
        - Engines are interchangeable: the same inputs must yield the same BacktestResult fields
          within documented tolerance (proven against ``backtesting.py`` in
          ``tests/test_engine_vs_backtestingpy.py``).
        """
        raise NotImplementedError
