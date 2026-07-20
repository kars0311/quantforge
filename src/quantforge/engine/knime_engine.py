"""KnimeEngine (STRETCH) — run a KNIME workflow headless as an alternative engine.

Demonstrates the polyglot story: the same backtest, expressed as a visual KNIME workflow, behind the
same Engine interface. Invoke KNIME in batch mode (or via KNIME Server/Business Hub REST), exchanging
data through the Arrow/Parquet interchange contract.

TODO(stretch): write positions/prices to Parquet, invoke the KNIME workflow headless, read results
back via interchange, and adapt into a BacktestResult. Verify metrics match PythonEngine.
"""

from __future__ import annotations

from typing import Any

from .base import BacktestResult, Engine


class KnimeEngine(Engine):
    name = "knime"

    def run_backtest(self, prices, positions, params: dict[str, Any]) -> BacktestResult:
        raise NotImplementedError
