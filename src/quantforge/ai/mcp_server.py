"""MCP tool server — exposes the pipeline as four read-only, input-validated tools (component 11).

The research agent and the natural-language interface never touch the pipeline directly; they
call these four functions (FR-7), and the MCP ``build_server`` wrapper only adds schema/discovery
on top of them. Keeping the tool bodies as plain Python functions is deliberate: they are testable
without an MCP client, the Streamlit app can call them in-process, and every safety property below
is a property of the *function*, not of the transport.

Tools
-----
``load_data(start, end, tickers)``            -> ``{dataset_id, tickers, start, end, n_days}``
``run_backtest(dataset_id, strategy, ...)``   -> ``{result_id, metrics}``
``optimize_portfolio(result_ids | dataset_id)`` -> ``{weights, frontier}``
``get_metrics(result_id)``                    -> ``{metrics}``

Safety properties (SF-7 / SF-8 / AR-6)
--------------------------------------
* **No code, paths, or SQL.** Every string argument is enum-checked (tickers against
  ``loader.UNIVERSE``, strategies via ``strategies.validate_params``, engines against
  ``ENGINES``, objectives against a literal set) and every number is range-checked. There is no
  argument through which a caller could name a file, a module, or an expression, so there is
  nothing to ``open``/``eval``/``exec`` — the module contains none of those calls.
* **Holdout is structurally unreachable.** ``load_data`` REJECTS (never truncates) any date
  outside ``[train.start, validation.end]`` from ``loader.get_split_bounds()``. Even a
  prompt-injected agent cannot ask for 2023+ prices; the one holdout score goes through
  ``guardrails.score_holdout`` and its single-use handle, never through a tool.
* **Opaque handles, small outputs.** Datasets and results live in in-process dicts keyed by
  random ``secrets`` tokens. A handle is not a path, not a pickle, and not guessable; tool outputs
  are JSON-serializable dicts of scalars/lists — never a DataFrame, never raw prices — which keeps
  the agent's context cheap (AR-6) and means raw data never rides in a model prompt.
* **Engine-agnostic.** A tool call names an engine by string and the server resolves it via
  ``ENGINES`` (AR-1/AR-4); the agent never sees a concrete engine class.

``guardrails.assert_no_codegen`` (the PUBLIC_MODE parameter-only gate) runs inside
``run_backtest`` because that is the only tool that carries a ``{strategy, params}`` payload —
the other three accept nothing but enums, ISO dates, and opaque handles, so their own argument
validation is already the equivalent gate.

Registry lifetime: the handle dicts are process-local and unbounded by design for this phase
(one Streamlit process, short sessions). A multi-container deployment would replace them with a
shared store keyed by the same opaque ids; nothing in the tool signatures would change.

MCP layer (``TOOL_SCHEMAS`` + ``build_server``)
-----------------------------------------------
``TOOL_SCHEMAS`` is the four tools as Anthropic-style ``{name, description, input_schema}``
definitions in ``TOOLS`` order. It is generated once at import from the same literals the
functions validate against (``loader.UNIVERSE``, ``STRATEGIES``, ``PARAM_WHITELIST``, ``ENGINES``,
``OBJECTIVES``), so the schema the model reads and the check the server runs cannot drift. The
order is fixed and every enum is sorted because this list is also the cached prompt prefix for
the NL interface (AR-6): a byte that moves between calls is a cache miss.

The schemas are deliberately plain: no ``oneOf``/``anyOf`` combinators, only
``additionalProperties: false`` objects with typed, enum-bounded properties. That is why
``optimize_portfolio``'s exactly-one-of ``{result_ids, dataset_id}`` rule is stated in the tool
and property *descriptions* rather than encoded as ``oneOf`` — the function body enforces it
with a ValueError, and the model is told the rule in prose it can act on.

``build_server()`` wraps the same four functions in a ``FastMCP`` server for discovery over the
protocol. The in-process callers (agent, NL interface, Streamlit) keep calling the plain functions;
the server exists for external MCP clients. To point Claude Desktop (or any MCP client) at the
pipeline over stdio::

    python -m quantforge.ai.mcp_server

which is the ``__main__`` block below — not exercised in CI, since it blocks on stdin.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

from quantforge import interchange
from quantforge.ai import guardrails
from quantforge.data import loader
from quantforge.engine.base import BacktestResult, Engine
from quantforge.engine.python_engine import PythonEngine
from quantforge.portfolio import optimize
from quantforge.strategies import PARAM_WHITELIST, STRATEGIES, validate_params

TOOLS = ["load_data", "run_backtest", "optimize_portfolio", "get_metrics"]

#: The engine registry (AR-4): a tool call names an engine as a string and the server resolves it
#: here, so the agent stays engine-agnostic and a future R/KNIME/MATLAB backend is one new entry.
#: ``guardrails.score_holdout`` resolves engines through this same dict (via a function-local
#: import, since this module imports guardrails), so there is exactly one place an engine name
#: can mean something.
ENGINES: dict[str, Engine] = {"python": PythonEngine()}

#: Portfolio objectives a tool call may name. A literal (not imported from ``optimize``) so the
#: tool's public contract is visible here and cannot widen silently if the optimizer grows knobs
#: (``weight_bounds``, for instance, is deliberately NOT exposed to the agent).
OBJECTIVES = {"max_sharpe", "min_volatility"}

#: Transaction-cost bounds in basis points. 0 is frictionless; 100 bps (1%) per unit turnover is
#: already far above any realistic equity cost, so anything larger is a typo, not a model.
_COST_BPS_RANGE = (0.0, 100.0)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FRONTIER_POINTS = 20

# --------------------------------------------------------------------------------------------
# Handle registries
# --------------------------------------------------------------------------------------------

#: dataset_id -> WIDE prices (index=date UTC, columns=ticker). Stored wide because that is the
#: shape both ``Strategy.generate_signals`` and ``Engine.run_backtest`` consume; converting once
#: at load time means every backtest on the handle skips the pivot.
_DATASETS: dict[str, pd.DataFrame] = {}
#: result_id -> BacktestResult (the full object, so ``optimize_portfolio`` can reach ``returns``
#: while ``get_metrics`` hands out only the metrics dict).
_RESULTS: dict[str, BacktestResult] = {}


def _default_price_source() -> pd.DataFrame:
    """The full long panel from the loader's Parquet cache (no network once primed)."""
    return loader.load_prices()


#: Where ``load_data`` gets the full LONG price panel. A module attribute (not a hard-coded call)
#: so tests can point it at a synthetic frame and prove the tools never touch the Parquet cache or
#: the network; production leaves the default, which is the loader's cached-file path.
_price_source: Callable[[], pd.DataFrame] = _default_price_source


def _new_id(prefix: str) -> str:
    """``f"{prefix}_{16 hex chars}"`` from ``secrets`` — unguessable, meaningless, not a path."""
    return f"{prefix}_{secrets.token_hex(8)}"


def _get_dataset(dataset_id: str) -> pd.DataFrame:
    """Resolve a dataset handle; unknown ids raise ``ValueError`` (never a bare KeyError).

    ValueError rather than KeyError because, to the caller (an LLM's next turn, or the UI), an
    unknown handle is an invalid *argument* like any other rejection here — one exception type
    at the tool boundary is what lets the agent loop catch and self-correct uniformly.
    """
    if not isinstance(dataset_id, str) or dataset_id not in _DATASETS:
        raise ValueError(f"unknown dataset_id {dataset_id!r}; call load_data first")
    return _DATASETS[dataset_id]


def _get_result(result_id: str) -> BacktestResult:
    """Resolve a result handle; unknown ids raise ``ValueError`` (see :func:`_get_dataset`)."""
    if not isinstance(result_id, str) or result_id not in _RESULTS:
        raise ValueError(f"unknown result_id {result_id!r}; call run_backtest first")
    return _RESULTS[result_id]


def reset_registry() -> None:
    """Forget every dataset and result handle. TESTS ONLY.

    Exists so each test starts from an empty registry (handles from a previous test must not
    make an "unknown id" assertion pass or fail by accident). Production never calls it: a live
    agent holding a handle expects it to stay valid for the session.
    """
    _DATASETS.clear()
    _RESULTS.clear()


# --------------------------------------------------------------------------------------------
# Argument validation helpers (each names the offending field in its ValueError)
# --------------------------------------------------------------------------------------------


def _validate_date(field: str, value: Any) -> str:
    """Require a ``YYYY-MM-DD`` string naming a real calendar date; return it unchanged.

    The regex pins the exact shape (``strptime`` alone would accept ``2015-1-1``), and
    ``strptime`` rejects impossible dates like ``2015-02-30``. Rejecting anything else — a
    ``datetime``, a Timestamp, a slash-separated string — keeps the tool's JSON contract literal.
    """
    if not isinstance(value, str) or not _DATE_RE.match(value):
        raise ValueError(f"{field} must be an ISO date string 'YYYY-MM-DD', got {value!r}")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field}={value!r} is not a valid calendar date") from exc
    return value


def _validate_tickers(tickers: Any) -> list[str]:
    """Require a non-empty list of unique strings, each in ``loader.UNIVERSE``."""
    if not isinstance(tickers, list) or not tickers:
        raise ValueError(f"tickers must be a non-empty list of ticker symbols, got {tickers!r}")
    seen: set[str] = set()
    for ticker in tickers:
        if not isinstance(ticker, str):
            raise ValueError(f"tickers must be strings, got {ticker!r}")
        if ticker not in loader.UNIVERSE:
            raise ValueError(
                f"ticker {ticker!r} is not in the fixed universe; allowed: {loader.UNIVERSE}"
            )
        if ticker in seen:
            raise ValueError(f"tickers contains duplicate {ticker!r}")
        seen.add(ticker)
    return list(tickers)


def _validate_window(start: str, end: str) -> None:
    """SF-8: ``start <= end`` and both inside ``[train.start, validation.end]``; never truncate.

    Rejecting (instead of clamping to the allowed range) is the point: a silently truncated
    request would let an agent *ask* for holdout dates and get a plausible-looking answer, hiding
    the fact that it tried. A loud error is also what lets the model's next turn self-correct.
    ISO date strings compare correctly as plain strings because the shape is fixed-width.
    """
    if start > end:
        raise ValueError(f"start {start!r} must not be after end {end!r}")
    bounds = loader.get_split_bounds()
    lo, hi = bounds["train"][0], bounds["validation"][1]
    if start < lo or end > hi:
        raise ValueError(
            f"date range {start}..{end} lies outside the train+validation window {lo}..{hi}; "
            f"dates after {hi} belong to the holdout, which tools never expose (SF-8), and "
            f"dates before {lo} have no data"
        )


def _validate_cost_bps(cost_bps: Any) -> float:
    """Require a real number (bool rejected) within ``_COST_BPS_RANGE``; return it as float."""
    if isinstance(cost_bps, bool) or not isinstance(cost_bps, (int, float)):
        raise ValueError(f"cost_bps must be a number, got {cost_bps!r}")
    lo, hi = _COST_BPS_RANGE
    if not (lo <= cost_bps <= hi):
        raise ValueError(f"cost_bps={cost_bps!r} outside [{lo:g}, {hi:g}] basis points")
    return float(cost_bps)


# --------------------------------------------------------------------------------------------
# The four tools
# --------------------------------------------------------------------------------------------


def load_data(start: str, end: str, tickers: list[str] | None = None) -> dict:
    """Load a price window into the registry; return its handle and a small summary.

    ``tickers`` defaults to the whole fixed universe. Dates are inclusive and must lie within the
    train+validation window (see :func:`_validate_window` — holdout dates are rejected, not
    clamped). The returned ``tickers`` are the ones *actually present* in the window (a name that
    IPO'd after ``end`` simply has no rows), sorted; ``n_days`` is the number of trading days.

    Nothing about the prices themselves is returned — only the handle — so the agent's context
    never carries raw data (AR-6) and cannot be used to reconstruct it.
    """
    tickers = loader.UNIVERSE if tickers is None else _validate_tickers(tickers)
    start = _validate_date("start", start)
    end = _validate_date("end", end)
    _validate_window(start, end)

    full = _price_source()
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    sliced = full[full["ticker"].isin(tickers) & full["date"].between(lo, hi)]
    if sliced.empty:
        raise ValueError(f"no price rows for tickers {sorted(tickers)} between {start} and {end}")
    wide = interchange.to_wide(sliced.reset_index(drop=True), "prices")

    dataset_id = _new_id("ds")
    _DATASETS[dataset_id] = wide
    return {
        "dataset_id": dataset_id,
        "tickers": sorted(str(c) for c in wide.columns),
        "start": start,
        "end": end,
        "n_days": int(wide.shape[0]),
    }


def run_backtest(
    dataset_id: str,
    strategy: str,
    params: dict | None = None,
    engine: str = "python",
    cost_bps: float = 10.0,
) -> dict:
    """Run a vetted strategy on a loaded dataset; return a result handle plus its metrics.

    Validation order is deliberate and matches the body line for line: (1) the dataset handle
    via ``_get_dataset`` (an unknown id is a ValueError — nothing else matters if there is no
    data); (2) ``params`` is defaulted to ``{}`` and must then be a dict, else ValueError;
    (3) ``strategy`` must be a str, else ValueError; (4) the PUBLIC_MODE parameter-only gate,
    ``guardrails.assert_no_codegen({strategy, params})`` (a no-op outside public mode);
    (5) the whitelist, ``validate_params``, whose messages are passed through verbatim so the
    agent can self-correct on the exact param; (6) the engine name against ``ENGINES``;
    (7) ``_validate_cost_bps``. Only then does anything compute.

    Steps 2 and 3 run BEFORE the gate on purpose: a non-dict ``params`` or a non-str
    ``strategy`` is a malformed call, not a code-execution attempt, so it is rejected as a plain
    ValueError and in PUBLIC_MODE those two shapes never surface as ``PublicModeViolation``.
    The gate then sees exactly the ``{str, dict}`` payload it is written to inspect, which is
    why its own type checks and this function's cannot disagree.

    The pipeline is the same one ``app/streamlit_app.py`` runs: ``generate_signals`` (weights as
    of each close) then ``Engine.run_backtest`` (which applies the one-day execution lag and the
    turnover cost) — there is one backtest path, whichever front end drives it.
    """
    wide = _get_dataset(dataset_id)
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"params must be a dict of strategy parameters, got {params!r}")
    if not isinstance(strategy, str):
        raise ValueError(f"strategy must be a string naming a vetted strategy, got {strategy!r}")
    guardrails.assert_no_codegen({"strategy": strategy, "params": params})
    validated = validate_params(strategy, params)
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; registered engines: {sorted(ENGINES)}")
    cost = _validate_cost_bps(cost_bps)

    positions = STRATEGIES[strategy]().generate_signals(wide, validated)
    result = ENGINES[engine].run_backtest(
        wide, positions, {"cost_bps": cost, "strategy": strategy, **validated}
    )

    result_id = _new_id("res")
    _RESULTS[result_id] = result
    return {"result_id": result_id, "metrics": _metrics_dict(result)}


def optimize_portfolio(
    result_ids: list[str] | None = None,
    dataset_id: str | None = None,
    objective: str = "max_sharpe",
) -> dict:
    """Mean-variance weights + efficient frontier over strategy results OR raw assets.

    Exactly one selector is accepted:

    * ``result_ids`` — allocate capital across *strategy return streams* (the FR-4 story:
      momentum + mean-reversion blended into one portfolio). Streams are inner-joined on the
      dates all of them cover, the same alignment rule the Streamlit Portfolio tab uses.
    * ``dataset_id`` — allocate across the *assets* in a loaded window (buy-and-hold daily
      returns, ``pct_change``), the classic Markowitz picture.

    Both paths hand a wide returns panel to ``portfolio.optimize`` — the optimizer does not care
    whether columns are assets or strategies, the math is identical. The optimizer needs at least
    two columns, which is surfaced here as a ValueError naming the count rather than letting the
    optimizer's own message (which does not know what a result_id is) confuse the caller.
    Output is plain floats: ``weights`` keyed by result_id / ticker, and ``frontier`` as a list
    of ``{risk, ret}`` points whose keys are the interchange ``frontier`` schema's column names.
    """
    if (result_ids is None) == (dataset_id is None):
        raise ValueError("optimize_portfolio needs exactly one of result_ids or dataset_id")
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; allowed: {sorted(OBJECTIVES)}")

    if result_ids is not None:
        panel = _strategy_panel(result_ids)
    else:
        panel = _asset_panel(dataset_id)

    weights = optimize.optimize_weights(panel, {"objective": objective})
    frontier_df = optimize.frontier(panel, n_points=_FRONTIER_POINTS)
    frontier_cols = interchange.SCHEMAS["frontier"].names  # ("risk", "ret") — from the contract
    return {
        "weights": {str(label): float(w) for label, w in weights.items()},
        "frontier": [
            {col: float(row[col]) for col in frontier_cols} for _, row in frontier_df.iterrows()
        ],
    }


def get_metrics(result_id: str) -> dict:
    """Return ``{"metrics": ...}`` for a result handle — the ``performance._KEYS`` dict only."""
    return {"metrics": _metrics_dict(_get_result(result_id))}


# --------------------------------------------------------------------------------------------
# Internal helpers for optimize_portfolio / outputs
# --------------------------------------------------------------------------------------------


def _strategy_panel(result_ids: Any) -> pd.DataFrame:
    """Inner-join the net return series of >= 2 unique known results into a wide panel."""
    if not isinstance(result_ids, list):
        raise ValueError(f"result_ids must be a list of result handles, got {result_ids!r}")
    if len(set(result_ids)) != len(result_ids):
        raise ValueError(f"result_ids contains duplicates: {result_ids!r}")
    if len(result_ids) < 2:
        raise ValueError(
            f"optimize_portfolio needs at least 2 result_ids to allocate across, "
            f"got {len(result_ids)}"
        )
    series = {rid: _get_result(rid).returns for rid in result_ids}
    # join="inner" keeps only dates every stream covers (streams may come from different
    # windows); dropna then removes any warm-up rows an engine might leave as NaN.
    return pd.concat(series, axis=1, join="inner").dropna()


def _asset_panel(dataset_id: Any) -> pd.DataFrame:
    """Buy-and-hold daily simple returns of a loaded dataset's assets (>= 2 tickers)."""
    wide = _get_dataset(dataset_id)
    if wide.shape[1] < 2:
        raise ValueError(
            f"optimize_portfolio needs a dataset with at least 2 tickers, "
            f"dataset {dataset_id!r} has {wide.shape[1]}"
        )
    # dropna removes the first (undefined) return row and any date on which some asset has no
    # price (pre-IPO raggedness) — the optimizer rejects NaNs rather than guessing at them.
    return wide.pct_change().dropna()


def _metrics_dict(result: BacktestResult) -> dict[str, float]:
    """A fresh ``{name: float}`` copy of ``result.metrics`` (plain floats, JSON-serializable)."""
    return {str(k): float(v) for k, v in result.metrics.items()}


# --------------------------------------------------------------------------------------------
# MCP schema layer: TOOL_SCHEMAS (what the model reads) and build_server (discovery transport)
# --------------------------------------------------------------------------------------------

#: One paragraph an MCP client shows its model before the tool list. It states the three facts
#: an agent must not have to discover by trial: read-only, train+validation only, vetted only.
_INSTRUCTIONS = (
    "QuantForge research tools. All four tools are read-only: they load cached daily prices, run "
    "vetted backtests, optimize weights, and report metrics; nothing is written and no code is "
    "executed. Only the train and validation windows are available — the holdout period is never "
    "exposed and any request touching it is rejected outright. Strategies are limited to the "
    "vetted set and parameters to their whitelisted ranges; results pass by opaque handles "
    "(dataset_id / result_id) that are only meaningful to these tools."
)

_JSON_TYPE = {int: "integer", float: "number"}


def _params_schema() -> dict:
    """JSON Schema for ``run_backtest.params``: the union of ``PARAM_WHITELIST`` over strategies.

    One ``params`` object has to serve every strategy (the tool takes ``strategy`` as a sibling
    field, and JSON Schema's ``if/then`` is poorly supported by tool-calling models), so a knob
    shared by two strategies gets the LOOSEST bounds across them and the description spells out
    the per-strategy range. The schema is a first, coarse filter that keeps obviously wrong calls
    from ever being sent; ``validate_params`` remains the exact gate and rejects, say, a momentum
    ``lookback`` of 10 that this union schema (mean-reversion allows 5..60) lets through.

    Generated, not hand-written, so adding a param to the whitelist cannot leave the schema stale.
    A param whose kind (numeric vs categorical) or numeric type disagrees across strategies is
    refused at import: that needs a naming decision, not a silently widened schema.
    """
    specs: dict[str, list[tuple[str, dict]]] = {}
    for strategy in sorted(PARAM_WHITELIST):
        for param, spec in PARAM_WHITELIST[strategy].items():
            specs.setdefault(param, []).append((strategy, spec))

    properties: dict[str, dict] = {}
    for param in sorted(specs):
        entries = specs[param]
        kinds = {"choices" in spec for _, spec in entries}
        if len(kinds) != 1:
            raise ValueError(
                f"param {param!r} is categorical in one strategy and numeric in another"
            )
        if kinds == {True}:
            choices = sorted(set().union(*(spec["choices"] for _, spec in entries)))
            detail = "; ".join(
                f"{s}: one of {sorted(spec['choices'])} (default {spec['default']!r})"
                for s, spec in entries
            )
            properties[param] = {"type": "string", "enum": choices, "description": detail}
            continue
        types = {spec["type"] for _, spec in entries}
        if len(types) != 1:
            raise ValueError(f"param {param!r} has conflicting numeric types across strategies")
        detail = "; ".join(
            f"{s}: {spec['min']}..{spec['max']} (default {spec['default']})" for s, spec in entries
        )
        properties[param] = {
            "type": _JSON_TYPE[types.pop()],
            "minimum": min(spec["min"] for _, spec in entries),
            "maximum": max(spec["max"] for _, spec in entries),
            "description": detail,
        }
    return {
        "type": "object",
        "description": (
            "Strategy parameters. Every key must be whitelisted for the chosen strategy and "
            "within its range (per-strategy ranges are in each property's description); omitted "
            "keys take the strategy's defaults. No other keys are accepted."
        ),
        "properties": properties,
        "additionalProperties": False,
    }


def _build_tool_schemas() -> list[dict]:
    """The four Anthropic-style tool definitions, in ``TOOLS`` order, from the module's literals."""
    holdout_note = (
        "Dates must lie within the train+validation window; any date in the holdout period "
        "(or before the data begins) is rejected — never truncated — so keep the range inside it."
    )
    return [
        {
            "name": "load_data",
            "description": (
                "Read-only: load a window of cached daily close prices for tickers in the fixed "
                "universe and return an opaque dataset_id plus a small summary (no prices are "
                f"returned). {holdout_note} Omit tickers to load the whole universe."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "pattern": _DATE_RE.pattern,
                        "description": "First date, inclusive, YYYY-MM-DD.",
                    },
                    "end": {
                        "type": "string",
                        "pattern": _DATE_RE.pattern,
                        "description": "Last date, inclusive, YYYY-MM-DD; must not precede start.",
                    },
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(loader.UNIVERSE)},
                        "minItems": 1,
                        "uniqueItems": True,
                        "description": "Subset of the fixed universe; defaults to all of it.",
                    },
                },
                "required": ["start", "end"],
                "additionalProperties": False,
            },
        },
        {
            "name": "run_backtest",
            "description": (
                "Read-only: run one vetted strategy on a loaded dataset with the one-day execution "
                "lag and turnover-based transaction costs applied, and return an opaque result_id "
                "plus its performance metrics. Only the vetted strategies and their whitelisted "
                "parameters are accepted; no code of any kind. The dataset is train+validation "
                "data only (holdout is unreachable)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "Handle returned by load_data.",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": sorted(STRATEGIES),
                        "description": "One of the vetted strategies.",
                    },
                    "params": _params_schema(),
                    "engine": {
                        "type": "string",
                        "enum": sorted(ENGINES),
                        "default": "python",
                        "description": "Backtest engine to run on.",
                    },
                    "cost_bps": {
                        "type": "number",
                        "minimum": _COST_BPS_RANGE[0],
                        "maximum": _COST_BPS_RANGE[1],
                        "default": 10.0,
                        "description": "Transaction cost in basis points per unit turnover.",
                    },
                },
                "required": ["dataset_id", "strategy"],
                "additionalProperties": False,
            },
        },
        {
            "name": "optimize_portfolio",
            "description": (
                "Read-only: mean-variance weights and a 20-point efficient frontier over EITHER "
                "several backtest results (pass result_ids, at least two — allocate across "
                "strategy return streams) OR the assets of one dataset (pass dataset_id — "
                "allocate across tickers). Pass exactly one of the two selectors. Uses "
                "train+validation data only."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "result_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "uniqueItems": True,
                        "description": (
                            "Handles returned by run_backtest (at least two, unique). Mutually "
                            "exclusive with dataset_id: pass exactly one of the two."
                        ),
                    },
                    "dataset_id": {
                        "type": "string",
                        "description": (
                            "Handle returned by load_data (needs >= 2 tickers). Mutually "
                            "exclusive with result_ids: pass exactly one of the two."
                        ),
                    },
                    "objective": {
                        "type": "string",
                        "enum": sorted(OBJECTIVES),
                        "default": "max_sharpe",
                        "description": "Optimization objective.",
                    },
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_metrics",
            "description": (
                "Read-only: return the performance metrics of a previous backtest result "
                "(the same dict run_backtest returned)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "result_id": {
                        "type": "string",
                        "description": "Handle returned by run_backtest.",
                    },
                },
                "required": ["result_id"],
                "additionalProperties": False,
            },
        },
    ]


#: The four tool definitions, ``TOOLS`` order, JSON-serializable. Built once at import so every
#: consumer (NL interface prompt, MCP discovery, tests) sees byte-identical schemas.
TOOL_SCHEMAS: list[dict] = _build_tool_schemas()

#: name -> function, the registration table ``build_server`` walks. Kept next to the schemas so
#: the order is visibly the same as ``TOOLS`` (asserted in tests).
_TOOL_FUNCTIONS: dict[str, Callable[..., dict]] = {
    "load_data": load_data,
    "run_backtest": run_backtest,
    "optimize_portfolio": optimize_portfolio,
    "get_metrics": get_metrics,
}


def build_server() -> FastMCP:
    """Construct a fresh ``FastMCP`` server exposing ``TOOLS`` (names and order preserved).

    A new instance per call — no module-level singleton — so a test or a second client cannot
    inherit registrations (or ``warn_on_duplicate_tools`` noise) from an earlier one; the tool
    bodies are the module functions, so all servers still share one handle registry.

    Each function is registered with ``add_tool`` (FastMCP validates arguments with a pydantic
    model derived from the signature, then calls the function), after which the advertised
    ``inputSchema`` is replaced with the matching ``TOOL_SCHEMAS`` entry. The derived schema is
    not the contract: it says ``anyOf [..., null]`` where the tool means "optional", allows any
    ``params`` key, and carries no enums — while the hand-built one is strict
    (``additionalProperties: false``, enum-checked strings, ranged numbers).

    Advertising the strict schema is not the same as enforcing it: FastMCP registers its wire
    handler with the low-level server's ``validate_input=False`` (it relies on pydantic's *lax*
    coercion instead, which silently drops unknown top-level keys and turns ``cost_bps: true``
    into ``1.0``). So the handler is re-registered here with validation ON, which makes the
    low-level server ``jsonschema``-check every wire call against the advertised ``inputSchema``
    before FastMCP's pydantic step or the function runs — the schema the model reads is the gate
    the transport applies. In-process ``FastMCP.call_tool`` bypasses the wire handler and relies on
    the functions' own validation, which is proven separately. ValueErrors raised by a tool surface
    as ``mcp.server.fastmcp.exceptions.ToolError`` (in-process) or an ``isError`` result (on the
    wire) — never swallowed, so the model's next turn sees why it was refused.
    """
    server = FastMCP("quantforge", instructions=_INSTRUCTIONS)
    for schema in TOOL_SCHEMAS:
        name = schema["name"]
        server.add_tool(_TOOL_FUNCTIONS[name], name=name, description=schema["description"])
        # FastMCP offers no public way to hand add_tool a schema, so reach into the tool manager
        # (a test pins that it still takes effect on list_tools).
        server._tool_manager.get_tool(name).parameters = schema["input_schema"]
    # FastMCP has no public switch for the low-level input gate either; re-registering the same
    # handler with validate_input=True is the documented low-level API, just driven from here.
    # A wire-level test pins that an extra key / bad type is refused with "Input validation error".
    server._mcp_server.call_tool(validate_input=True)(server.call_tool)
    return server


if __name__ == "__main__":
    # `python -m quantforge.ai.mcp_server` -> stdio MCP server for an external client.
    build_server().run("stdio")
