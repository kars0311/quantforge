"""Verifier tests: strategy registry + PARAM_WHITELIST + validate_params (week 4, SF-3).

Independently proves the vetted-set choke point against docs/components/05-strategies.md:
registry keys locked to ai/guardrails.VETTED_STRATEGIES, whitelist literals exactly as documented,
validate_params merges defaults without mutating the caller, and every malformed input the public
demo could receive (unvetted strategy, unknown key, bool-for-int, fractional float, out-of-bounds,
bad choice) is rejected with the offending name in the message. Adversarial cases included: bool
lookalike for int, NaN/inf smuggled into float bounds, and a behavioral proof that whitelist
defaults are byte-identical to each strategy's inline defaults (same weights either way).
"""

import numpy as np
import pandas as pd
import pytest

from quantforge.ai.guardrails import VETTED_STRATEGIES
from quantforge.engine.base import Strategy
from quantforge.strategies import PARAM_WHITELIST, STRATEGIES, validate_params
from quantforge.strategies.mean_reversion import MeanReversionStrategy
from quantforge.strategies.momentum import MomentumStrategy


def _synthetic_prices(n: int = 300, k: int = 4, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    rets = rng.normal(0.0003, 0.012, size=(n, k))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"S{i}" for i in range(k)])


# ---------------------------------------------------------------- registry


def test_registry_exact_keys_and_classes():
    assert set(STRATEGIES) == {"momentum", "mean_reversion"}
    assert STRATEGIES["momentum"] is MomentumStrategy
    assert STRATEGIES["mean_reversion"] is MeanReversionStrategy
    for cls in STRATEGIES.values():
        assert issubclass(cls, Strategy)


def test_registry_matches_guardrails_vetted_set():
    # SF-3: the registry and the guardrail constant must never drift apart.
    assert set(STRATEGIES) == VETTED_STRATEGIES
    assert set(PARAM_WHITELIST) == VETTED_STRATEGIES


# ---------------------------------------------------------------- whitelist literals


def test_whitelist_momentum_literals_match_design_doc():
    wl = PARAM_WHITELIST["momentum"]
    assert set(wl) == {"lookback", "top_n"}
    assert wl["lookback"] == {"type": int, "min": 20, "max": 252, "default": 126}
    assert wl["top_n"] == {"type": int, "min": 0, "max": 10, "default": 0}


def test_whitelist_mean_reversion_literals_match_design_doc():
    wl = PARAM_WHITELIST["mean_reversion"]
    assert set(wl) == {"lookback", "entry_z", "exit_z", "mode"}
    assert wl["lookback"] == {"type": int, "min": 5, "max": 60, "default": 20}
    assert wl["entry_z"] == {"type": float, "min": 0.5, "max": 3.0, "default": 2.0}
    assert wl["exit_z"] == {"type": float, "min": 0.0, "max": 1.5, "default": 0.5}
    assert wl["mode"]["choices"] == {"long_flat", "long_short"}
    assert wl["mode"]["default"] == "long_flat"


# ---------------------------------------------------------------- defaults / merging


def test_empty_params_return_full_defaults():
    assert validate_params("momentum", {}) == {"lookback": 126, "top_n": 0}
    assert validate_params("mean_reversion", {}) == {
        "lookback": 20,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "mode": "long_flat",
    }


def test_partial_dict_merged_without_mutating_caller():
    caller = {"lookback": 40}
    out = validate_params("mean_reversion", caller)
    assert out == {"lookback": 40, "entry_z": 2.0, "exit_z": 0.5, "mode": "long_flat"}
    assert caller == {"lookback": 40}  # untouched
    assert out is not caller


def test_returns_new_dict_every_call():
    a = validate_params("momentum", {})
    b = validate_params("momentum", {})
    assert a == b and a is not b


# ---------------------------------------------------------------- rejection: strategy / keys


@pytest.mark.parametrize("bad", ["pairs", "evil", "Momentum", ""])
def test_unvetted_strategy_rejected_by_name(bad):
    with pytest.raises(ValueError) as ei:
        validate_params(bad, {})
    assert repr(bad) in str(ei.value) or bad in str(ei.value)


def test_unknown_param_key_rejected_by_name():
    with pytest.raises(ValueError) as ei:
        validate_params("momentum", {"lookbak": 100})
    assert "lookbak" in str(ei.value)


# ---------------------------------------------------------------- rejection: types (adversarial)


def test_bool_for_int_param_rejected():
    # isinstance(True, int) is True in Python — a naive check would accept it.
    with pytest.raises(ValueError) as ei:
        validate_params("momentum", {"lookback": True})
    assert "lookback" in str(ei.value)


def test_bool_for_float_param_rejected():
    with pytest.raises(ValueError) as ei:
        validate_params("mean_reversion", {"entry_z": True})
    assert "entry_z" in str(ei.value)


def test_fractional_float_for_int_param_rejected():
    with pytest.raises(ValueError) as ei:
        validate_params("momentum", {"lookback": 126.5})
    assert "lookback" in str(ei.value)


def test_string_number_rejected_for_numeric_params():
    with pytest.raises(ValueError):
        validate_params("momentum", {"lookback": "126"})
    with pytest.raises(ValueError):
        validate_params("mean_reversion", {"entry_z": "2.0"})


def test_integral_float_coerced_to_int_for_int_param():
    # JSON round-trips often float-ify ints; 126.0 means 126.
    out = validate_params("momentum", {"lookback": 126.0})
    assert out["lookback"] == 126 and type(out["lookback"]) is int


def test_int_coerced_to_float_for_float_param():
    out = validate_params("mean_reversion", {"entry_z": 2})
    assert out["entry_z"] == 2.0 and type(out["entry_z"]) is float


@pytest.mark.parametrize("evil", [float("nan"), float("inf"), -float("inf")])
def test_nan_and_inf_cannot_slip_through_float_bounds(evil):
    # NaN fails every comparison; a bounds check written as `not (lo <= v <= hi)` must reject it.
    with pytest.raises(ValueError):
        validate_params("mean_reversion", {"entry_z": evil})


# ---------------------------------------------------------------- rejection: bounds / choices


@pytest.mark.parametrize(
    ("strategy", "param", "value"),
    [
        ("momentum", "lookback", 19),
        ("momentum", "lookback", 253),
        ("momentum", "top_n", -1),
        ("momentum", "top_n", 11),
        ("mean_reversion", "lookback", 4),
        ("mean_reversion", "lookback", 61),
        ("mean_reversion", "entry_z", 0.4),
        ("mean_reversion", "entry_z", 3.1),
        ("mean_reversion", "exit_z", -0.1),
        ("mean_reversion", "exit_z", 1.6),
    ],
)
def test_out_of_bounds_rejected_by_name(strategy, param, value):
    with pytest.raises(ValueError) as ei:
        validate_params(strategy, {param: value})
    assert param in str(ei.value)


@pytest.mark.parametrize(
    ("strategy", "param", "value"),
    [
        ("momentum", "lookback", 20),
        ("momentum", "lookback", 252),
        ("mean_reversion", "entry_z", 0.5),
        ("mean_reversion", "entry_z", 3.0),
        ("mean_reversion", "exit_z", 0.0),
        ("mean_reversion", "exit_z", 1.5),
    ],
)
def test_bounds_are_inclusive(strategy, param, value):
    assert validate_params(strategy, {param: value})[param] == value


def test_bad_choice_rejected_by_name():
    with pytest.raises(ValueError) as ei:
        validate_params("mean_reversion", {"mode": "short_only"})
    assert "mode" in str(ei.value)
    with pytest.raises(ValueError):
        validate_params("mean_reversion", {"mode": True})


# ---------------------------------------------------------------- end-to-end with strategies


def test_strategies_accept_validated_dicts_and_produce_valid_weights():
    prices = _synthetic_prices()
    for name, cls in STRATEGIES.items():
        weights = cls().generate_signals(prices, validate_params(name, {}))
        assert weights.shape == prices.shape
        assert (weights.abs().sum(axis=1) <= 1.0 + 1e-9).all()
        assert not weights.isna().any().any()


def test_whitelist_defaults_behaviorally_identical_to_inline_defaults():
    # The whitelist promises its defaults equal the strategies' inline defaults. Prove it
    # behaviorally: validated-empty params and no params must yield identical weights.
    prices = _synthetic_prices()
    for name, cls in STRATEGIES.items():
        via_whitelist = cls().generate_signals(prices, validate_params(name, {}))
        via_inline = cls().generate_signals(prices, None)
        pd.testing.assert_frame_equal(via_whitelist, via_inline)


def test_long_short_validated_params_run_end_to_end():
    prices = _synthetic_prices()
    params = validate_params("mean_reversion", {"mode": "long_short", "lookback": 10})
    weights = MeanReversionStrategy().generate_signals(prices, params)
    assert weights.shape == prices.shape
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-9).all()
