# KNIME engine (stretch)

Demonstrates the polyglot story: the same backtest expressed as a **visual KNIME workflow**, behind
the same `Engine` interface (`engine/knime_engine.py`), selectable as an "engine" in the demo.

**Interop via the Parquet contract:** the Python side writes prices/positions to Parquet; the KNIME
workflow reads them (KNIME reads Parquet natively), runs the backtest visually, and writes results
back; `KnimeEngine` adapts those into a `BacktestResult`. Verify metrics match `PythonEngine`.

Invocation options:
- KNIME batch mode (headless CLI) for a self-contained demo.
- KNIME Server / Business Hub REST for a hosted workflow.

TODO(stretch): add the `.knwf` workflow file(s) here and wire `KnimeEngine` to invoke it headless.
