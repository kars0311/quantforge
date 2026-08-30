"""Vetted strategy registry + parameter whitelist (SF-3 choke point).

WHY this module exists: the public deployment promises "parameter-only" AI — the model may pick a
vetted strategy and tune whitelisted knobs, but can never inject code or novel parameters into the
pipeline. That promise is only as strong as its narrowest gate, so there is exactly ONE:
``validate_params`` below. ``ai/guardrails.assert_no_codegen``, the MCP server, and the Streamlit UI
all funnel user/LLM-supplied strategy requests through this function before anything touches the
engine. Adding a strategy or param means editing this file (and the design doc), nowhere else.

Bounds come from docs/components/05-strategies.md and defaults are byte-identical to the inline
defaults inside each strategy's ``generate_signals`` — validated params and "no params at all" must
mean the same backtest.

Kept import-light on purpose: guardrails and the MCP layer import this at startup, and the
registry itself needs nothing beyond the strategy classes.
"""

from __future__ import annotations

from typing import Any

from ..engine.base import Strategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy

#: The vetted set. Keys MUST stay in lockstep with ai/guardrails.VETTED_STRATEGIES —
#: tests assert set equality so the two can never drift apart silently.
STRATEGIES: dict[str, type[Strategy]] = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
}

#: Per strategy: param -> spec. Numeric specs carry {type, min, max, default} (bounds inclusive);
#: categorical specs carry {type, choices, default}. Single source of truth for every gate.
PARAM_WHITELIST: dict[str, dict] = {
    "momentum": {
        "lookback": {"type": int, "min": 20, "max": 252, "default": 126},
        "top_n": {"type": int, "min": 0, "max": 10, "default": 0},
    },
    "mean_reversion": {
        "lookback": {"type": int, "min": 5, "max": 60, "default": 20},
        "entry_z": {"type": float, "min": 0.5, "max": 3.0, "default": 2.0},
        "exit_z": {"type": float, "min": 0.0, "max": 1.5, "default": 0.5},
        "mode": {"type": str, "choices": {"long_flat", "long_short"}, "default": "long_flat"},
    },
}


def _coerce_numeric(strategy: str, param: str, spec: dict, value: Any) -> int | float:
    """Type-check and coerce one numeric param, rejecting lookalikes.

    WHY the fussiness: inputs arrive from JSON (MCP), an LLM, or UI widgets, where 126.0 means the
    int 126 but 126.5 is a bug and True is *never* a window length — yet in Python
    ``isinstance(True, int)`` holds and ``int(126.9)`` silently truncates. Rejecting bools and
    fractional floats here keeps "validated" from meaning "mangled into validity".
    """
    if isinstance(value, bool):  # bool is an int subclass; must be checked first
        raise ValueError(
            f"{strategy}: param {param!r} must be {spec['type'].__name__}, got bool {value!r}"
        )
    if spec["type"] is int:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)  # 126.0 -> 126: JSON round-trips often float-ify ints
        raise ValueError(f"{strategy}: param {param!r} must be an integer, got {value!r}")
    # float params: any real number is fine; coerce ints so downstream math sees a float
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{strategy}: param {param!r} must be a number, got {value!r}")


def validate_params(strategy: str, params: dict) -> dict:
    """Validate ``params`` against the whitelist; return a NEW dict merged with defaults.

    This is the single choke point between untrusted input (LLM output, MCP requests, UI forms)
    and the backtest engine: everything not explicitly whitelisted is rejected, loudly, with the
    offending strategy/param named in the ValueError so the caller (or the model's next turn) can
    self-correct. The input dict is never mutated — callers may be holding it for logging/replay.

    Raises ValueError for: unvetted strategy, unknown param key, wrong type (bools and fractional
    floats are not ints), out-of-bounds value, or an unlisted choice.
    """
    if strategy not in PARAM_WHITELIST:
        raise ValueError(
            f"unknown strategy {strategy!r}; vetted strategies: {sorted(PARAM_WHITELIST)}"
        )
    whitelist = PARAM_WHITELIST[strategy]

    unknown = set(params) - set(whitelist)
    if unknown:
        raise ValueError(
            f"{strategy}: unknown param(s) {sorted(unknown)!r}; allowed: {sorted(whitelist)}"
        )

    validated: dict = {}
    for param, spec in whitelist.items():
        if param not in params:
            validated[param] = spec["default"]
            continue
        value = params[param]
        if "choices" in spec:
            if value not in spec["choices"]:
                raise ValueError(
                    f"{strategy}: param {param!r} must be one of "
                    f"{sorted(spec['choices'])}, got {value!r}"
                )
            validated[param] = value
        else:
            coerced = _coerce_numeric(strategy, param, spec, value)
            if not (spec["min"] <= coerced <= spec["max"]):
                raise ValueError(
                    f"{strategy}: param {param!r}={coerced!r} outside "
                    f"[{spec['min']}, {spec['max']}]"
                )
            validated[param] = coerced
    return validated


__all__ = [
    "STRATEGIES",
    "PARAM_WHITELIST",
    "validate_params",
    "MomentumStrategy",
    "MeanReversionStrategy",
]
