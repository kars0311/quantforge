"""Correctness: the custom PythonEngine must agree with `backtesting.py` on a known strategy.

This is the proof the engine is right — the single most important rigor test. Run the SAME momentum
strategy through both and assert the equity curves / Sharpe match within tolerance.
"""

import pytest


@pytest.mark.skip(reason="TODO(week3): implement once PythonEngine + momentum exist")
def test_python_engine_matches_backtesting_py():
    # TODO: run momentum via PythonEngine and via backtesting.py on the same data;
    # assert final equity and Sharpe agree within tolerance.
    raise NotImplementedError
