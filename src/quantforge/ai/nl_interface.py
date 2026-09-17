"""Natural-language interface (component 12): plain English -> tool calls -> plain English.

"backtest momentum on AAPL and MSFT, 2015-2019, 10 bps costs" becomes a structured plan (one
forced tool call to a small, fast model), the plan becomes MCP tool calls (``load_data`` ->
``run_backtest`` -> ``get_metrics``), and the resulting metrics become three to five sentences a
person can read. This is the demo's crowd-pleaser and, because it sits behind a public URL with
paid API hooks, also its most exposed surface — so the module is written gate-first.

Why the shape is what it is
---------------------------
* **Two model calls, deliberately single-shot.** ``parse`` and ``explain`` are one request each;
  there is no tool-use loop here. Iteration is the research agent's job (component 13); the NL
  interface must stay cheap enough (~$0.001 per query on Haiku) to hand to strangers.
* **Forced tool use for parsing.** The parse call is told it MUST call ``plan_backtest``, whose
  ``input_schema`` is generated from the same literals the tools validate against (vetted
  strategies, whitelisted params, the fixed universe). The model's only job is to fill in a form
  whose fields are enums and ranges; it cannot name a file, a module, or an expression (SF-3/SF-5).
  A ``clarify`` field is the escape hatch: the prompt says to use it rather than invent a strategy
  or a date range, and ``parse`` refuses to guess essentials the model left blank.
* **Every model call is metered in one place.** ``_call`` is the only function that talks to the
  SDK; it charges ``budget`` with the response's real usage immediately, before returning, so a
  failure between the two calls can never lose spend (SF-1). Cached input tokens are billed at
  the full input rate on purpose — over-counting is the safe direction for a spend cap.
* **Gates before spend.** ``handle`` consults the rate limit and the budget before the first
  model call, checks PUBLIC_MODE's parameter-only rule on the raw plan before a single tool runs,
  and never calls ``explain`` on a failed backtest. A closed gate returns a ``fallback`` code, not
  an exception, so the UI can route to its cached scenarios instead of showing an error (SF-2).
* **Prompt caching on, prompts frozen.** Both system prompts and the tool definition are
  module-level constants built from stable literals (no timestamps, no ids), and ``_call`` marks
  them with ``cache_control`` so repeated queries share the prefix (AR-6). Haiku's minimum
  cacheable prefix is larger than these prompts, so the cache may not engage today; the
  breakpoints cost nothing and will engage as the prompts grow.
* **PUBLIC_MODE comes from the environment only.** The stub's ``public_mode`` kwarg is gone: a
  caller (or a prompt-injected one) must not be able to switch the demo into code-executing mode.

Model choice: ``claude-haiku-4-5`` for both calls — the project rule (AGENTS.md) reserves Haiku
for NL parsing/explaining and Sonnet for agent reasoning. The id is the bare form the API accepts.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import anthropic

from quantforge.ai import budget, guardrails, mcp_server
from quantforge.data.loader import UNIVERSE, get_split_bounds
from quantforge.strategies import STRATEGIES, validate_params

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"

#: Conservative per-call token guesses for the PRE-call ``budget.allow`` check (the real usage
#: from the response is what gets charged). Input covers the frozen system prompt, the tool
#: definition, and a 2000-char query with room to spare; output is above either ``max_tokens``.
_EST_TOKENS_IN = 2500
_EST_TOKENS_OUT = 600

_PARSE_MAX_TOKENS = 512
_EXPLAIN_MAX_TOKENS = 400

#: Hard cap on query length. A backtest request fits in a sentence; anything longer is either a
#: pasted document or an attempt to pad the prompt, and both should cost nothing.
_MAX_QUERY_CHARS = 2000

_DEFAULT_COST_BPS = 10.0

_FALLBACK_RATE = "rate"
_FALLBACK_BUDGET = "budget"
_FALLBACK_PUBLIC_MODE = "public_mode"
_FALLBACK_INVALID = "invalid"
_FALLBACK_API_ERROR = "api_error"

_CLARIFY_UNPARSEABLE = "I couldn't turn that into a backtest — which strategy and date range?"

#: ``handle``'s answer to an empty, whitespace-only, or over-long query. It is a clarification
#: (a normal outcome, ``fallback=None``) rather than an exception because the UI's text box makes
#: a blank submit or a pasted document an ordinary user action, not a caller bug — and it is
#: decided BEFORE the rate limit so a stranger hammering "submit" on an empty box burns neither
#: a slot nor a ledger read. Built from ``_MAX_QUERY_CHARS`` so the number cannot drift.
_CLARIFY_EMPTY_QUERY = (
    "Tell me what to backtest — a strategy and a date range — in under "
    f"{_MAX_QUERY_CHARS} characters."
)

#: ``_raw_plan``'s answer when the model fills ``tickers`` with something other than a list.
_CLARIFY_TICKERS_NOT_LIST = (
    "I couldn't read the tickers in that request — please list them, e.g. AAPL and MSFT."
)


# --------------------------------------------------------------------------------------------
# Frozen prompt material (cache-stable: built once from constants, no timestamps or ids)
# --------------------------------------------------------------------------------------------


def _run_backtest_params_schema() -> dict:
    """The generated ``params`` schema from ``mcp_server.TOOL_SCHEMAS`` (deep-copied).

    Reused rather than re-derived so the NL tool and the MCP tool cannot disagree about which
    knobs exist or what their ranges are: one generator, two consumers.
    """
    for schema in mcp_server.TOOL_SCHEMAS:
        if schema["name"] == "run_backtest":
            return copy.deepcopy(schema["input_schema"]["properties"]["params"])
    raise RuntimeError("mcp_server.TOOL_SCHEMAS has no run_backtest entry")  # pragma: no cover


def _window() -> tuple[str, str]:
    """``(train.start, validation.end)`` — the only dates the NL interface may ever touch."""
    bounds = get_split_bounds()
    return bounds["train"][0], bounds["validation"][1]


_WINDOW_START, _WINDOW_END = _window()

#: The single tool the parse call is forced to use. Its input mirrors ``run_backtest``/``load_data``
#: exactly, so a filled-in form is already a valid pipeline request; ``clarify`` is the one field
#: that is NOT a pipeline input — it is how the model declines to guess.
PLAN_TOOL: dict = {
    "name": "plan_backtest",
    "description": (
        "Turn the user's request into one backtest plan. Fill in strategy, start and end from "
        "the request; use params, tickers and cost_bps only when the user specified them "
        "(otherwise omit them so defaults apply). If the request is not a backtest request, or "
        "does not state a strategy or a date range, set clarify to ONE short question and "
        "nothing else — never invent a strategy or a date range."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "enum": sorted(STRATEGIES),
                "description": "One of the vetted strategies, as the user named it.",
            },
            "params": _run_backtest_params_schema(),
            "tickers": {
                "type": "array",
                "items": {"type": "string", "enum": list(UNIVERSE)},
                "minItems": 1,
                "uniqueItems": True,
                "description": "Only tickers the user named, from the fixed universe; omit for all.",
            },
            "start": {
                "type": "string",
                "pattern": mcp_server._DATE_RE.pattern,
                "description": f"First date, YYYY-MM-DD, no earlier than {_WINDOW_START}.",
            },
            "end": {
                "type": "string",
                "pattern": mcp_server._DATE_RE.pattern,
                "description": f"Last date, YYYY-MM-DD, no later than {_WINDOW_END}.",
            },
            "cost_bps": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "description": "Transaction cost in basis points per unit turnover, if stated.",
            },
            "clarify": {
                "type": "string",
                "description": (
                    "Set ONLY when the request is off-topic or missing an essential (strategy "
                    "or date range); ask one question."
                ),
            },
        },
        "additionalProperties": False,
    },
}

SYSTEM_PARSE: str = (
    "You convert a user's plain-English request into exactly one plan_backtest tool call for a "
    "quantitative research demo. Facts you may rely on:\n"
    f"- Vetted strategies (the only ones that exist): {', '.join(sorted(STRATEGIES))}.\n"
    f"- Fixed ticker universe: {', '.join(UNIVERSE)}.\n"
    f"- Available data: {_WINDOW_START} through {_WINDOW_END} (train + validation). Later dates "
    "are a held-out test set and can never be used.\n"
    "Rules: never invent a date range or strategy — use clarify. Map informal names to the "
    "vetted list only when the mapping is obvious (e.g. 'trend following' -> momentum); "
    "otherwise ask. Leave optional fields (params, tickers, cost_bps) out unless the user gave "
    "them explicitly. Do not write prose; call the tool."
)

SYSTEM_EXPLAIN: str = (
    "You explain the result of one backtest to a curious non-specialist in three to five plain "
    "sentences: what was run (strategy, tickers, dates, costs), the headline numbers, and one "
    "honest caveat (for example: the sample is train + validation data only, the universe is a "
    "fixed historical list so it carries survivorship bias, or past performance is not "
    "predictive). The exact plan and metric values are given to you as JSON; quote them as they "
    "are, rounding sensibly. Never invent, extrapolate, or estimate a number that is not in the "
    f"JSON. Data covers {_WINDOW_START} through {_WINDOW_END}; vetted strategies are "
    f"{', '.join(sorted(STRATEGIES))}. Metrics are annualized where applicable; total_return, "
    "cagr, ann_vol, max_drawdown and hit_rate are fractions (0.25 means 25%)."
)


# --------------------------------------------------------------------------------------------
# The one metered call
# --------------------------------------------------------------------------------------------

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """Lazily construct the SDK client (module attribute so tests can monkeypatch it).

    Lazy on purpose: importing this module must not require credentials (the Streamlit app and
    the test suite import it offline), and the SDK — not this module — reads
    ``ANTHROPIC_API_KEY``/its other credential sources at construction time.
    """
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _usage_int(usage: Any, field: str) -> int:
    """A usage counter as an int, with a missing/``None`` field counting as 0."""
    value = getattr(usage, field, None)
    return int(value) if value else 0


def _call(
    client: Any,
    *,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: dict | None = None,
    max_tokens: int,
) -> tuple[Any, float]:
    """Perform ONE Messages API request and charge the budget for it; return ``(response, usd)``.

    This is the single place in the module that talks to the SDK (the grep-level SF-1 check in
    ``docs/components/09-ai-budget.md`` relies on that). The system prompt and the LAST tool
    definition carry ``cache_control`` breakpoints: tools render before system before messages,
    so those two markers cache the whole stable prefix and leave only the user turn volatile.

    Metering: ``tokens_in`` is the sum of uncached, cache-write and cache-read input tokens, all
    priced at the full input rate. That over-counts cache reads (billed at a discount by the API),
    which is the safe error for a spend gate — the ledger can only ever say we spent MORE than we
    did. Charging happens here, immediately, so the caller cannot forget and a later failure
    cannot skip it. ``budget.charge`` is the one thing allowed to raise past this point (a corrupt
    ledger in PUBLIC_MODE) because spend must never go unrecorded silently.
    """
    request: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": messages,
    }
    if tools:
        cached_tools = [copy.deepcopy(tool) for tool in tools]
        cached_tools[-1]["cache_control"] = {"type": "ephemeral"}
        request["tools"] = cached_tools
    if tool_choice is not None:
        request["tool_choice"] = tool_choice

    response = client.messages.create(**request)

    usage = getattr(response, "usage", None)
    tokens_in = (
        _usage_int(usage, "input_tokens")
        + _usage_int(usage, "cache_creation_input_tokens")
        + _usage_int(usage, "cache_read_input_tokens")
    )
    tokens_out = _usage_int(usage, "output_tokens")
    usd = budget.estimate(MODEL, tokens_in, tokens_out)
    budget.charge(usd, model=MODEL, tokens_in=tokens_in, tokens_out=tokens_out)
    return response, usd


# --------------------------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------------------------


def _validate_query(query: Any) -> str:
    """Reject empty, non-string, or over-long queries BEFORE any spend; return the query."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if len(query) > _MAX_QUERY_CHARS:
        raise ValueError(f"query is {len(query)} chars; the limit is {_MAX_QUERY_CHARS}")
    return query


def _find_tool_use(response: Any, name: str) -> Any | None:
    """The first ``tool_use`` block named ``name``, or None — any other tool name is ignored."""
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == name:
            return block
    return None


def _raw_plan(response: Any) -> dict:
    """Turn the parse response into either ``{"clarify": ...}`` or a plan with RAW params.

    "Raw" means the model's ``params`` are passed through untouched (defaulting to ``{}``); the
    PUBLIC_MODE gate has to see them before ``validate_params`` normalizes them (see
    :func:`_finalize_plan`). Essentials (strategy, start, end) are never defaulted: a missing one
    becomes a clarification, because a guessed date range is a silently different experiment.
    Optional fields take the pipeline's defaults — whole universe, 10 bps.

    ``tickers`` is only trusted when it is a list (or absent/``None``/``[]``, meaning the whole
    universe). The schema says "array", but nothing forces the model to obey it, and ``list()``
    on a wrong-typed value would not fail — it would *mangle*: a string ``"AAPL"`` explodes into
    ``["A", "A", "P", "L"]`` and a dict into its keys, and the resulting nonsense would reach
    ``load_data`` looking like a legitimate (if unknown) ticker list. A non-list is therefore a
    clarification. A list of non-strings passes through and is rejected downstream by
    ``mcp_server._validate_tickers`` (fallback ``invalid``), exactly like an unknown ticker.
    """
    block = _find_tool_use(response, PLAN_TOOL["name"])
    if block is None:
        return {"clarify": _CLARIFY_UNPARSEABLE}
    data = getattr(block, "input", None)
    if not isinstance(data, dict):
        return {"clarify": _CLARIFY_UNPARSEABLE}

    clarify = data.get("clarify")
    if isinstance(clarify, str) and clarify.strip():
        return {"clarify": clarify.strip()}

    strategy, start, end = data.get("strategy"), data.get("start"), data.get("end")
    if not strategy or not start or not end:
        return {"clarify": _CLARIFY_UNPARSEABLE}

    tickers = data.get("tickers")
    if tickers is not None and not isinstance(tickers, list):
        return {"clarify": _CLARIFY_TICKERS_NOT_LIST}
    cost_bps = data.get("cost_bps")
    return {
        "strategy": strategy,
        "params": data.get("params") or {},
        "tickers": list(tickers) if tickers else list(UNIVERSE),
        "start": start,
        "end": end,
        "cost_bps": _DEFAULT_COST_BPS if cost_bps is None else cost_bps,
    }


def _finalize_plan(plan: dict) -> dict:
    """Apply the PUBLIC_MODE parameter-only gate, then whitelist-validate the params.

    Order matters and is the reason this is a separate step: ``guardrails.assert_no_codegen``
    must inspect the model's raw output. If ``validate_params`` ran first, a plan carrying a
    ``code`` key would be rejected as an ordinary bad parameter and surface as a polite
    clarification — the public-mode fallback (which the UI treats as an abuse signal, not a typo)
    would never fire. Outside public mode the gate is a no-op and ``validate_params`` alone
    decides; its ValueError becomes a clarification so the user can fix the knob.
    """
    if "clarify" in plan:
        return plan
    guardrails.assert_no_codegen({"strategy": plan["strategy"], "params": plan["params"]})
    if not isinstance(plan["params"], dict):
        # The schema says "object", but nothing forces the model to obey it; ``validate_params``
        # assumes a dict (a list of dicts would raise TypeError, not ValueError, and escape
        # ``handle`` as a traceback). Outside public mode this is a clarification like any
        # other bad parameter; in public mode the gate above has already rejected it.
        return {
            "clarify": f"params must be an object of strategy parameters, got {plan['params']!r}"
        }
    try:
        params = validate_params(plan["strategy"], plan["params"])
    except ValueError as err:
        return {"clarify": str(err)}
    return {**plan, "params": params}


def _parse_call(query: str, client: Any) -> tuple[dict, float]:
    """The metered parse request -> ``(raw plan or clarify, usd)``; no gates, no validation.

    Split from :func:`parse_with_spend` so ``handle`` can bank the charge BEFORE the
    parameter-only gate runs on the result — a public-mode violation must still report what the
    parse call cost.
    """
    query = _validate_query(query)
    client = _get_client() if client is None else client
    response, usd = _call(
        client,
        system=SYSTEM_PARSE,
        messages=[{"role": "user", "content": query}],
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": PLAN_TOOL["name"]},
        max_tokens=_PARSE_MAX_TOKENS,
    )
    return _raw_plan(response), usd


def parse_with_spend(query: str, *, client: Any = None) -> tuple[dict, float]:
    """``parse`` plus the USD actually charged for its one model call.

    Raises ``ValueError`` for a bad query (before any spend), ``anthropic.APIError`` if the SDK
    call fails (nothing charged), and ``guardrails.PublicModeViolation`` in PUBLIC_MODE when the
    model's plan is not parameter-only (charged: the call happened).
    """
    raw, usd = _parse_call(query, client)
    return _finalize_plan(raw), usd


def parse(query: str, *, client: Any = None) -> dict:
    """Plain English -> ``{strategy, params, tickers, start, end, cost_bps}`` or ``{"clarify"}``.

    The success shape is exactly what ``mcp_server.load_data``/``run_backtest`` consume. The
    clarify shape is returned — never raised — for anything the model could not or should not
    turn into a plan: off-topic text, a missing strategy or date range, or a parameter outside
    the whitelist (the whitelist message is the clarification, so the user sees the real range).
    """
    plan, _ = parse_with_spend(query, client=client)
    return plan


# --------------------------------------------------------------------------------------------
# explain
# --------------------------------------------------------------------------------------------


def _template_explanation(plan: dict, metrics: dict) -> str:
    """Deterministic fallback so ``explain`` never returns an empty string."""

    def num(key: str) -> float:
        try:
            return float(metrics.get(key, float("nan")))
        except (TypeError, ValueError):
            return float("nan")

    n = len(plan.get("tickers") or [])
    return (
        f"Ran {plan.get('strategy')} on {n} tickers {plan.get('start')}–{plan.get('end')}: "
        f"total return {num('total_return'):.1%}, Sharpe {num('sharpe'):.2f}, "
        f"max drawdown {num('max_drawdown'):.1%}."
    )


def explain_with_spend(plan: dict, metrics: dict, *, client: Any = None) -> tuple[str, float]:
    """``explain`` plus the USD actually charged for its one model call."""
    client = _get_client() if client is None else client
    content = json.dumps({"plan": plan, "metrics": metrics}, sort_keys=True, default=str)
    response, usd = _call(
        client,
        system=SYSTEM_EXPLAIN,
        messages=[{"role": "user", "content": content}],
        max_tokens=_EXPLAIN_MAX_TOKENS,
    )
    text = "".join(
        block.text
        for block in (getattr(response, "content", None) or [])
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
    ).strip()
    return (text or _template_explanation(plan, metrics)), usd


def explain(plan: dict, metrics: dict, *, client: Any = None) -> str:
    """Metrics -> three to five plain-English sentences; never empty, never invented numbers.

    The exact plan and metric values ride in the user turn as sorted JSON (sorted so the message
    bytes are stable for identical inputs), and the system prompt forbids numbers that are not in
    it. An empty model reply falls back to a fixed template built from the same values.
    """
    text, _ = explain_with_spend(plan, metrics, client=client)
    return text


# --------------------------------------------------------------------------------------------
# handle: the UI's single entry point
# --------------------------------------------------------------------------------------------


def _result(
    *,
    plan: dict | None,
    metrics: dict | None,
    explanation: str,
    spend_usd: float,
    fallback: str | None,
) -> dict:
    """The one return shape. Keyword-only so a reordered positional call cannot mislabel a field."""
    return {
        "plan": plan,
        "metrics": metrics,
        "explanation": explanation,
        "spend_usd": float(spend_usd),
        "fallback": fallback,
    }


def _run_tools(plan: dict) -> dict:
    """``load_data`` -> ``run_backtest`` -> ``get_metrics`` with the plan's fields; return metrics.

    Deliberately the same three tool functions the agent and any MCP client use — there is one
    backtest path. Their ValueErrors (holdout dates, unknown ticker, bad cost) are the caller's
    "invalid" fallback.
    """
    dataset = mcp_server.load_data(plan["start"], plan["end"], plan["tickers"])
    result = mcp_server.run_backtest(
        dataset["dataset_id"], plan["strategy"], plan["params"], "python", plan["cost_bps"]
    )
    return mcp_server.get_metrics(result["result_id"])["metrics"]


def handle(query: str, *, session_key: str = "anon", client: Any = None) -> dict:
    """Parse a NL query, run the pipeline via the tools, and explain the result — with every gate.

    Returns ``{"plan", "metrics", "explanation", "spend_usd", "fallback"}``. ``fallback`` is
    ``None`` on success (and on a clarification, which is a normal outcome) or one of:

    ``"rate"``         the per-session hourly limit is exhausted — no model call, no spend
    ``"budget"``       ``budget.allow`` refused the pre-call estimate of BOTH calls (or the
                       kill-switch is on) — no model call, no spend
    ``"public_mode"``  PUBLIC_MODE=on and the parsed plan was not parameter-only — parse was
                       charged, nothing ran
    ``"invalid"``      a tool rejected the plan (holdout dates, unknown ticker, ...) — the tool's
                       message is the explanation; no explain call is made, so nothing more is
                       spent on a request that failed
    ``"api_error"``    the SDK raised (network, auth, 4xx/5xx) — the SDK's message is the
                       explanation, never a traceback

    Gates run in this order, each at most once and all BEFORE the first model call:
    query shape -> rate limit -> budget -> parse -> parameter-only gate -> tools -> explain.
    Query validation comes first (before the rate limit) on purpose: an empty, whitespace-only
    or over-``_MAX_QUERY_CHARS`` query returns the ``_CLARIFY_EMPTY_QUERY`` clarification with
    ``fallback=None`` and ``spend_usd=0.0``, and it costs nothing and records nothing — no
    rate-limit slot, no ledger read, no model call. A blank submit is an ordinary thing for a
    person to do at a text box and must not eat into their hourly allowance. Budget is estimated
    for two calls, so a query that can afford to parse but not to explain is refused up front
    rather than left half-done. The parameter-only gate runs on the raw parse output before any
    tool executes. ``spend_usd`` is the sum of what the two calls actually charged; closed gates
    report ``0.0``.

    ``query`` must be a ``str`` and ``session_key`` a non-empty string — either being wrong is a
    caller bug, so it raises ``ValueError`` (before any gate) rather than falls back.
    """
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if not query.strip() or len(query) > _MAX_QUERY_CHARS:
        return _result(
            plan={"clarify": _CLARIFY_EMPTY_QUERY},
            metrics=None,
            explanation=_CLARIFY_EMPTY_QUERY,
            spend_usd=0.0,
            fallback=None,
        )

    if not guardrails.rate_limit(session_key):
        return _result(
            plan=None,
            metrics=None,
            explanation="Too many requests from this session in the last hour — try again later.",
            spend_usd=0.0,
            fallback=_FALLBACK_RATE,
        )

    per_call = budget.estimate(MODEL, _EST_TOKENS_IN, _EST_TOKENS_OUT)
    if not budget.allow(2 * per_call):
        return _result(
            plan=None,
            metrics=None,
            explanation="The live AI budget for today is used up — showing cached scenarios instead.",
            spend_usd=0.0,
            fallback=_FALLBACK_BUDGET,
        )

    spend = 0.0
    try:
        raw, usd = _parse_call(query, client)
    except anthropic.APIError as err:
        logger.warning("NL parse call failed: %s", err)
        return _result(
            plan=None,
            metrics=None,
            explanation=str(err),
            spend_usd=0.0,
            fallback=_FALLBACK_API_ERROR,
        )
    spend += usd

    try:
        plan = _finalize_plan(raw)
    except guardrails.PublicModeViolation as err:
        # The parse call already happened (and was charged); nothing downstream runs.
        return _result(
            plan=None,
            metrics=None,
            explanation=f"Public mode accepts parameter-only requests: {err}",
            spend_usd=spend,
            fallback=_FALLBACK_PUBLIC_MODE,
        )

    if "clarify" in plan:
        return _result(
            plan=plan, metrics=None, explanation=plan["clarify"], spend_usd=spend, fallback=None
        )

    try:
        metrics = _run_tools(plan)
    except ValueError as err:
        return _result(
            plan=plan,
            metrics=None,
            explanation=str(err),
            spend_usd=spend,
            fallback=_FALLBACK_INVALID,
        )

    try:
        explanation, usd = explain_with_spend(plan, metrics, client=client)
        spend += usd
    except anthropic.APIError as err:
        logger.warning("NL explain call failed: %s", err)
        return _result(
            plan=plan,
            metrics=metrics,
            explanation=str(err),
            spend_usd=spend,
            fallback=_FALLBACK_API_ERROR,
        )

    return _result(
        plan=plan, metrics=metrics, explanation=explanation, spend_usd=spend, fallback=None
    )
