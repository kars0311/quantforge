"""Mechanical pin of the frozen Strategy/Engine seam (AR-1, week 4).

WHY this test exists: ``engine/base.py`` is declared FROZEN at week 4 — any change to its public
surface is a breaking change requiring an explicit decision. Prose in a docstring can't stop an
accidental rename slipping through a refactor, so this file pins the surface *mechanically*:
dataclass field names and order, ABC-ness, the exact abstract-method sets, the exact parameter
names of each abstract method, and the subclass relationships every concrete strategy/engine must
satisfy. If you meant to change the seam, you must consciously edit this file too — that friction
is the enforcement.

Everything here is pure introspection (dataclasses/inspect/abc): no data, no I/O, no engine runs.
"""

from __future__ import annotations

import dataclasses
import inspect
from abc import ABC

import pytest

from quantforge import strategies
from quantforge.engine.base import BacktestResult, Engine, Strategy
from quantforge.engine.python_engine import PythonEngine

# ---------------------------------------------------------------------------
# BacktestResult: the standardized result container
# ---------------------------------------------------------------------------


def test_backtestresult_is_a_dataclass():
    assert dataclasses.is_dataclass(BacktestResult)


def test_backtestresult_field_names_and_order_frozen():
    # Order matters: positional construction BacktestResult(curve, rets, metrics) is public API.
    names = [f.name for f in dataclasses.fields(BacktestResult)]
    assert names == ["equity_curve", "returns", "metrics", "meta"]


def test_backtestresult_meta_defaults_to_empty_dict():
    meta_field = {f.name: f for f in dataclasses.fields(BacktestResult)}["meta"]
    # Must be a default_factory (a shared mutable {} default would leak state across results).
    assert meta_field.default is dataclasses.MISSING
    assert meta_field.default_factory is not dataclasses.MISSING
    assert meta_field.default_factory() == {}
    # And the other three fields are required (no defaults): construction with three
    # positional args works and yields meta == {}.
    result = BacktestResult(equity_curve=None, returns=None, metrics={})
    assert result.meta == {}


def test_backtestresult_meta_not_shared_between_instances():
    a = BacktestResult(equity_curve=None, returns=None, metrics={})
    b = BacktestResult(equity_curve=None, returns=None, metrics={})
    a.meta["engine"] = "python"
    assert b.meta == {}


# ---------------------------------------------------------------------------
# Strategy / Engine: ABCs with exactly one abstract method each
# ---------------------------------------------------------------------------


def test_strategy_and_engine_are_abcs():
    assert issubclass(Strategy, ABC)
    assert issubclass(Engine, ABC)


def test_abstract_method_sets_frozen():
    # Exact set equality: adding an abstract method silently breaks every existing subclass,
    # removing one silently un-enforces the contract. Both must be deliberate.
    assert set(Strategy.__abstractmethods__) == {"generate_signals"}
    assert set(Engine.__abstractmethods__) == {"run_backtest"}


def test_generate_signals_signature_frozen():
    params = list(inspect.signature(Strategy.generate_signals).parameters)
    assert params == ["self", "prices", "params"]


def test_run_backtest_signature_frozen():
    params = list(inspect.signature(Engine.run_backtest).parameters)
    assert params == ["self", "prices", "positions", "params"]


def test_class_level_name_attributes_exist():
    # 'name' is how results/meta identify which strategy/engine produced them; it must exist
    # on the base classes so every subclass has one even before overriding.
    assert isinstance(Strategy.name, str)
    assert isinstance(Engine.name, str)


def test_bases_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Strategy()
    with pytest.raises(TypeError):
        Engine()


# ---------------------------------------------------------------------------
# Concrete implementations plug into the seam
# ---------------------------------------------------------------------------


def test_every_registered_strategy_subclasses_the_seam():
    assert strategies.STRATEGIES, "vetted registry must not be empty"
    for key, cls in strategies.STRATEGIES.items():
        assert issubclass(cls, Strategy), f"STRATEGIES[{key!r}] = {cls!r} is not a Strategy"


def test_python_engine_subclasses_the_seam():
    assert issubclass(PythonEngine, Engine)
