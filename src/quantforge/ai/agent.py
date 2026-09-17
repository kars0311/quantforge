"""AI research agent (component 13): propose -> backtest -> read metrics -> refine.

The headline feature AND the rigor talking point. A Sonnet tool-use conversation proposes
``{strategy, params}`` experiments; the *runner* (this module's Python, not the model) executes
each one through the MCP tool functions on the train window and then the validation window, feeds
both metric sets back, and the model refines or declares convergence. When the loop ends the runner
scores the winning configuration ONCE on the holdout via ``guardrails.score_holdout`` — outside the
conversation, never visible to the model.

Why the shape is what it is
---------------------------
* **P-hacking is made impossible, not just discouraged (RG-4 / SF-8).** An LLM iterating against
  the test set is a data-mined result with a respectable-looking Sharpe. So the agent's context
  contains metric summaries for train and validation ONLY: the system prompt names the two
  windows and states that later data is a held-out set that can never be requested; the tools it
  can call do not even take a date range (the runner fixes the windows); and ``mcp_server`` itself
  rejects any request touching the holdout as a second, independent layer. No holdout date appears
  anywhere in the prompt material — ``tests/test_holdout_isolation.py`` greps for one.
* **Proposal tools, not pipeline tools.** The model does not get ``load_data``/``run_backtest``
  directly. It gets ``propose_experiment`` (a vetted strategy name from an enum plus params from
  the SAME generated schema the MCP ``run_backtest`` tool uses) and ``declare_done``. A proposal is
  a filled-in form; the runner decides what to do with it. That is what makes PUBLIC_MODE's
  parameter-only rule (SF-3) structural: there is no field in which code could ride, and
  ``guardrails.assert_no_codegen`` still checks every proposal before it runs. It also keeps the
  agent engine-agnostic (FR-9): swapping the engine changes nothing the model sees.
* **Every model call is metered in one place (SF-1).** ``_call`` is the only function in the
  module that talks to the SDK; it charges ``budget`` with the response's real usage immediately,
  before returning, so a failure later in the iteration can never lose spend. Cached input tokens
  are billed at the full input rate on purpose — over-counting is the safe direction for a cap.
* **Prompt caching on, prompt frozen (AR-6).** The system prompt and both tool definitions are
  module-level constants built from stable literals (strategy registry, whitelist, split bounds,
  metric keys — no timestamps, no ids, and NOT the goal, which is volatile and rides in the first
  user turn). ``_call`` places three ``cache_control`` breakpoints — system, last tool, last
  content block of the last message — so each iteration re-reads the growing conversation prefix
  from cache instead of paying for it again.
* **Caps are guardrail constants, not agent knobs (SF-7).** ``max_iters`` is validated against
  ``guardrails.MAX_AGENT_ITERS`` and cannot be raised per call; a caller (or a prompt-injected
  one) must not be able to buy itself more iterations.

Model choice: ``claude-sonnet-5`` — the project rule (AGENTS.md) reserves Sonnet for agent
reasoning and Haiku for the NL interface. Sonnet 5 runs adaptive thinking by default and rejects
``temperature``/``top_p``/``top_k``/``thinking.budget_tokens``, so the request never sets them.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from typing import Any

import anthropic

from quantforge.ai import budget, guardrails, mcp_server
from quantforge.data.loader import get_split_bounds
from quantforge.metrics.performance import _KEYS
from quantforge.strategies import PARAM_WHITELIST, STRATEGIES, validate_params

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"

#: Output cap per turn. A turn is one tool call (a short JSON form plus a rationale sentence or
#: two); anything longer is prose the runner would ignore, so it should not be paid for.
_MAX_TOKENS = 1024

#: Conservative token guesses for the PRE-call ``budget.allow`` check (the real usage from the
#: response is what gets charged). Input grows with the conversation: a fixed base (system prompt
#: + tool schemas + goal) plus a per-turn allowance (one proposal + two metric dicts).
_EST_TOKENS_IN_BASE = 4000
_EST_TOKENS_IN_PER_TURN = 1200
_EST_TOKENS_OUT = _MAX_TOKENS

#: Hard cap on goal length. A research goal is a sentence or a short paragraph; anything longer
#: is a pasted document or prompt padding, and both should cost nothing.
_MAX_GOAL_CHARS = 2000

#: Transaction cost applied to every experiment (bps per unit turnover). Fixed, not a proposal
#: field: the agent must not be able to "improve" a strategy by assuming free trading.
_DEFAULT_COST_BPS = 10.0

#: Shape of one history record (the UI's iteration timeline; ``docs/components/13-ai-agent.md``).
#: The six documented keys plus ``error``: ``None`` on success, else the tool's ValueError message
#: — with ``train_metrics``/``val_metrics`` = ``None`` — for a proposal the tools rejected. A
#: rejected proposal is kept as a record rather than dropped because it is real, paid-for history
#: (the model spent a turn on it and was told why it failed), and the timeline should show what
#: the agent tried, not only what worked. ``_select_best`` skips records with an error.
_RECORD_KEYS = ("iter", "strategy", "params", "train_metrics", "val_metrics", "rationale", "error")

#: The record slice the model sees and the loop's result carries as ``best``.
_CONFIG_KEYS = ("strategy", "params", "train_metrics", "val_metrics")

#: Decimal places for metrics in a tool result. Four is enough to rank configurations (a Sharpe
#: difference in the fifth decimal is noise on daily data) and keeps each result a few dozen
#: tokens, which matters because every result stays in the cached prefix for the rest of the loop.
_RESULT_DECIMALS = 4


# --------------------------------------------------------------------------------------------
# Frozen prompt material (cache-stable: built once from constants, no timestamps or ids)
# --------------------------------------------------------------------------------------------


def _run_backtest_params_schema() -> dict:
    """The generated ``params`` schema from ``mcp_server.TOOL_SCHEMAS`` (deep-copied).

    Reused rather than re-derived so the proposal tool, the NL tool and the MCP tool cannot
    disagree about which knobs exist or what their ranges are: one generator, several consumers.
    """
    for schema in mcp_server.TOOL_SCHEMAS:
        if schema["name"] == "run_backtest":
            return copy.deepcopy(schema["input_schema"]["properties"]["params"])
    raise RuntimeError("mcp_server.TOOL_SCHEMAS has no run_backtest entry")  # pragma: no cover


def _window() -> tuple[str, str]:
    """``(train.start, validation.end)`` — the only dates the agent may ever touch."""
    bounds = get_split_bounds()
    return bounds["train"][0], bounds["validation"][1]


_WINDOW_START, _WINDOW_END = _window()

#: The proposal form. ``strategy`` is an enum of the vetted set and ``params`` is the MCP
#: ``run_backtest`` schema, so a filled-in form is already a valid pipeline request and there is
#: no free-text field in which code, a file, or an expression could ride (SF-3/SF-5).
PROPOSE_TOOL: dict = {
    "name": "propose_experiment",
    "description": (
        "Propose ONE experiment: a vetted strategy and whitelisted parameters. The runner "
        "backtests it on the train window and then on the validation window and returns both "
        "metric sets. Omit a parameter to use the strategy's default. State in rationale what "
        "you expect to learn from this run relative to the previous ones."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "enum": sorted(STRATEGIES),
                "description": "One of the vetted strategies.",
            },
            "params": _run_backtest_params_schema(),
            "rationale": {
                "type": "string",
                "description": (
                    "One or two sentences: why this configuration, given the results so far."
                ),
            },
        },
        "required": ["strategy", "params", "rationale"],
        "additionalProperties": False,
    },
}

#: The exit. A separate tool (rather than "stop calling tools") so convergence is an explicit,
#: recorded decision with a reason the UI can show, and so a turn with no tool call is
#: unambiguously a protocol error rather than a quiet finish.
DONE_TOOL: dict = {
    "name": "declare_done",
    "description": (
        "Stop the research loop: call this once the goal is met or further proposals are not "
        "improving validation Sharpe. Give the reason in one or two sentences."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why the loop should stop now.",
            }
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
}

TOOLS: list[dict] = [PROPOSE_TOOL, DONE_TOOL]


def _describe_whitelist() -> str:
    """One line per vetted strategy listing each whitelisted param with its range and default.

    Rendered from ``PARAM_WHITELIST`` (sorted, so the bytes are stable) rather than hand-written,
    so a whitelist change cannot leave the prompt describing knobs that no longer exist.
    """
    lines = []
    for strategy in sorted(PARAM_WHITELIST):
        parts = []
        for param, spec in PARAM_WHITELIST[strategy].items():
            if "choices" in spec:
                parts.append(
                    f"{param}: one of {sorted(spec['choices'])} (default {spec['default']!r})"
                )
            else:
                parts.append(
                    f"{param}: {spec['type'].__name__} {spec['min']}..{spec['max']} "
                    f"(default {spec['default']})"
                )
        lines.append(f"- {strategy}: " + "; ".join(parts))
    return "\n".join(lines)


def _build_system_prompt() -> str:
    """Assemble ``SYSTEM_RESEARCH`` from constants only (the goal is NOT part of it).

    The train and validation windows are spelled out so the model can reason about regime
    differences between them; the holdout is described only as "later data" — its dates are
    deliberately absent from the agent's context.
    """
    bounds = get_split_bounds()
    train_start, train_end = bounds["train"]
    val_start, val_end = bounds["validation"]
    return (
        "You are a quantitative research agent running a bounded experiment loop for a "
        "backtesting demo. You never see prices or code; you see metrics.\n\n"
        f"Vetted strategies (the only ones that exist): {', '.join(sorted(STRATEGIES))}.\n"
        "Whitelisted parameters per strategy (range, default):\n"
        f"{_describe_whitelist()}\n\n"
        f"Data: {_WINDOW_START} through {_WINDOW_END}, split into a train window "
        f"({train_start} to {train_end}) and a validation window ({val_start} to {val_end}). "
        "Later data is a held-out test set: it is never available to you, it cannot be "
        "requested, and it is scored once by the runner after you finish.\n\n"
        "Protocol: every propose_experiment is run on the train window and then on the "
        "validation window, and both metric sets come back to you. Metric keys: "
        f"{', '.join(_KEYS)}. total_return, cagr, ann_vol, max_drawdown and hit_rate are "
        "fractions (0.25 means 25%); sharpe is annualized. Transaction costs are fixed at "
        f"{_DEFAULT_COST_BPS:g} bps per unit turnover.\n\n"
        "Objective: maximise VALIDATION sharpe. Train sharpe is diagnostic only — a large gap "
        "between train and validation signals overfitting, and a configuration that wins on "
        "train but loses on validation is worse than a modest one that holds up on both.\n\n"
        "Rules: call exactly ONE tool per turn — either propose_experiment or declare_done — "
        "and write no prose outside the tool call. Do not repeat a configuration you have "
        "already run. Call declare_done once the goal is met or your proposals have stopped "
        "improving validation sharpe."
    )


SYSTEM_RESEARCH: str = _build_system_prompt()


# --------------------------------------------------------------------------------------------
# Input validation (before any spend)
# --------------------------------------------------------------------------------------------


def _validate_goal(goal: Any) -> str:
    """Reject empty, non-string, or over-long goals BEFORE any spend; return the goal."""
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be a non-empty string")
    if len(goal) > _MAX_GOAL_CHARS:
        raise ValueError(f"goal is {len(goal)} chars; the limit is {_MAX_GOAL_CHARS}")
    return goal


def _validate_max_iters(max_iters: Any) -> int:
    """An int in ``1..guardrails.MAX_AGENT_ITERS``, else ValueError naming the cap.

    ``bool`` is rejected explicitly (it is an ``int`` subclass) and floats are not coerced: a
    caller passing ``True`` or ``2.5`` has a bug, and a cap check must not paper over it. The
    upper bound is the guardrail constant and cannot be raised per call (SF-7).
    """
    cap = guardrails.MAX_AGENT_ITERS
    if isinstance(max_iters, bool) or not isinstance(max_iters, int):
        raise ValueError(f"max_iters must be an int between 1 and {cap}, got {max_iters!r}")
    if not 1 <= max_iters <= cap:
        raise ValueError(
            f"max_iters must be between 1 and {cap} (guardrails.MAX_AGENT_ITERS), got {max_iters}"
        )
    return max_iters


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


def _with_last_block_cached(messages: list[dict]) -> list[dict]:
    """A copy of ``messages`` whose last content block of the last message is marked for caching.

    The conversation is the part of the prefix that grows every iteration; caching up to its
    tail means turn N+1 pays full price only for the tool result just appended. A ``str``
    content is promoted to a single text block first, because ``cache_control`` attaches to
    blocks. The last block must be a plain dict (the JSON shapes the runner builds); anything
    else is a caller bug and is refused before any spend rather than sent and rejected by the API.

    Only the containers on the path to that one block are copied — the outer list, the last
    message dict, its content list and the last block — and every other message is passed
    through as the very object the caller holds. That is deliberate, not laziness: the
    assistant turns in the transcript are the SDK's own response ``content`` lists, and the API
    requires thinking blocks (with their signatures) to be replayed byte-for-byte for tool use
    with adaptive thinking. Not copying them is the simplest way to guarantee nothing is
    altered, and it avoids deep-copying an ever-growing transcript on every turn. The caller's
    objects are never mutated either way: the marker lives only on the fresh block dict.
    """
    if not messages:
        raise ValueError("messages must contain at least one message")
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else None
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list) or not content or not isinstance(content[-1], dict):
        raise ValueError("the last message's content must be a str or a non-empty list of blocks")
    marked = [*content[:-1], {**content[-1], "cache_control": {"type": "ephemeral"}}]
    return [*messages[:-1], {**last, "content": marked}]


def _call(
    client: Any,
    *,
    system: str,
    messages: list[dict],
    tools: list[dict],
    tool_choice: dict,
    max_tokens: int,
) -> tuple[Any, float]:
    """Perform ONE Messages API request and charge the budget for it; return ``(response, usd)``.

    This is the single place in the module that talks to the SDK (the grep-level SF-1 check in
    ``docs/components/09-ai-budget.md`` relies on that). Three ``cache_control`` breakpoints —
    the system prompt, the LAST tool, and the LAST content block of the LAST message — cache the
    whole prefix in render order (tools -> system -> messages), so each iteration re-reads the
    conversation so far and pays full price only for what was just appended. The caller's
    ``tools`` are deep-copied and ``messages`` are copied along the path to the marked block
    (see ``_with_last_block_cached``); neither is ever mutated: they are module constants and
    the live transcript, and a marker leaking into either would silently move the breakpoints.

    Sonnet 5 runs adaptive thinking by default; the request deliberately sets no ``thinking``,
    ``temperature``, ``top_p`` or ``top_k`` (the sampling knobs are rejected on this model).
    Forced ``tool_choice`` (``any``) alongside that default-on adaptive thinking is a supported
    combination on the Claude API — verified against the ``claude-api`` skill
    (``shared/model-migration.md``: only Amazon Bedrock requires ``thinking: {type: "disabled"}``
    next to a forced ``tool_choice``, and forced ``any``/``tool`` returns 400 only on
    ``claude-fable-5-1``/``claude-mythos-5-1``) — so ``thinking`` is absent on purpose, not by
    omission; ``tests/test_handoff_open_items.py`` pins the request shape.

    ``tools`` must be non-empty: the last tool carries a cache breakpoint, and the loop's
    protocol assumes the model can always call something. An empty list is a caller bug, refused
    here with a ValueError before the request is built (so nothing is sent and nothing charged)
    rather than surfacing as an IndexError from the breakpoint placement.

    Metering: ``tokens_in`` is the sum of uncached, cache-write and cache-read input tokens, all
    priced at the full input rate. That over-counts cache reads (billed at a discount by the API),
    which is the safe error for a spend gate — the ledger can only ever say we spent MORE than we
    did. Charging happens here, immediately, so the caller cannot forget and a later failure
    cannot skip it. ``budget.charge`` is the one thing allowed to raise past this point (a corrupt
    ledger in PUBLIC_MODE) because spend must never go unrecorded silently.
    """
    if not tools:
        raise ValueError("tools must contain at least one tool definition")
    cached_tools = [copy.deepcopy(tool) for tool in tools]
    cached_tools[-1]["cache_control"] = {"type": "ephemeral"}
    request: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "tools": cached_tools,
        "tool_choice": tool_choice,
        "messages": _with_last_block_cached(messages),
    }

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


def _find_tool_use_blocks(response: Any) -> list[Any]:
    """Every ``tool_use`` block in the response, in order (each has ``.id``, ``.name``, ``.input``).

    Returns all of them rather than the first so the loop can enforce the one-tool-per-turn
    rule: a turn with zero or several calls is a protocol violation the runner must see, not
    silently pick from.
    """
    return [
        block
        for block in (getattr(response, "content", None) or [])
        if getattr(block, "type", None) == "tool_use"
    ]


# --------------------------------------------------------------------------------------------
# The experiment step: proposal -> two tool calls -> record
# --------------------------------------------------------------------------------------------


def _run_experiment(
    train_id: str,
    val_id: str,
    strategy: str,
    params: dict | None,
    engine: str,
    cost_bps: float,
) -> dict[str, Any]:
    """Backtest one proposal on the train window, then the validation window; return the record.

    The runner reaches the pipeline ONLY through ``mcp_server.run_backtest`` — the same function
    the NL interface and external MCP clients call — so every safety property of that tool
    (vetted strategies, whitelisted params, engine registry, cost bound, holdout-free handles)
    applies to the agent for free and cannot be bypassed by a code path of its own (FR-7/FR-9).

    Order of operations is deliberate: gate, whitelist, train, validation.

    * ``assert_no_codegen`` then ``validate_params`` run FIRST, in the order ``run_backtest``
      itself applies them, so a bad proposal executes nothing — no result handle is minted for a
      request that was never valid — and a public-mode violation surfaces as
      ``PublicModeViolation`` rather than as the whitelist's generic "unknown param" message.
    * The *validated* dict (defaults merged, numerics coerced) is what both backtests receive and
      what the record carries: there is exactly one params object describing what ran, and it is
      byte-identical to what ``guardrails.score_holdout`` will later be handed. Recording the raw
      proposal instead would leave "what did we actually run" ambiguous whenever a default filled
      in a knob.
    * Train runs before validation. If the validation run fails after the train run succeeded,
      the ValueError still propagates and NO partial record exists — a half-experiment would give
      ``_select_best`` a train number with nothing to check it against, which is the exact
      asymmetry this loop exists to prevent.

    Every ValueError from the tool (unknown handle, unvetted strategy, whitelist message,
    ``PublicModeViolation``, cost bound) propagates unchanged: the message is written for the
    model's next turn to self-correct on, and the loop records it verbatim as the ``error``.

    Returns ``{"strategy", "params", "train_metrics", "val_metrics"}`` — each metrics dict is the
    tool's ``metrics`` (plain floats keyed by ``performance._KEYS``); no result handles.
    """
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(
            f"params must be a dict of strategy parameters, got {type(params).__name__}"
        )
    if not isinstance(strategy, str):
        raise ValueError(f"strategy must be a string naming a vetted strategy, got {strategy!r}")
    guardrails.assert_no_codegen({"strategy": strategy, "params": params})
    validated = validate_params(strategy, params)

    train = mcp_server.run_backtest(train_id, strategy, validated, engine, cost_bps)
    val = mcp_server.run_backtest(val_id, strategy, validated, engine, cost_bps)
    return {
        "strategy": strategy,
        "params": validated,
        "train_metrics": dict(train["metrics"]),
        "val_metrics": dict(val["metrics"]),
    }


def _val_sharpe(record: dict[str, Any]) -> float:
    """A record's validation Sharpe for ranking, with NaN/inf mapped to ``-inf``.

    A non-finite Sharpe means the configuration never traded (zero volatility) or blew up;
    either way it is not a candidate, and mapping it to ``-inf`` keeps the comparison total
    (``nan > x`` is False both ways, which would otherwise make the winner depend on order).
    ``None`` gets the same treatment: a NaN that has been through a JSON round-trip (the UI
    caches runs as scenario files; ``_round_metrics`` emits ``null`` for NaN) comes back as
    ``None``, and ``float(None)`` would turn a ranking pass over stored history into a TypeError.
    """
    sharpe = record["val_metrics"]["sharpe"]
    if sharpe is None:
        return -math.inf
    sharpe = float(sharpe)
    return sharpe if math.isfinite(sharpe) else -math.inf


def _best_record(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The history record (itself, not a copy) with the highest VALIDATION Sharpe, or None.

    Why validation and never train — the anti-overfitting point of the whole loop: the model
    proposes with the train metrics in front of it, so the train Sharpe is the number it was
    (indirectly) optimising and is biased upward by construction; a configuration that wins on
    train and loses on validation has probably fitted noise. The validation window is the first
    data the proposal was not tuned against, so its Sharpe is the least-biased estimate the loop
    has. It is still not unbiased — the model saw validation results too, across iterations,
    which is exactly why the holdout is scored once more, outside the loop, on data neither this
    function nor the model ever touched. Picking by train Sharpe would make the validation
    metrics decorative and the holdout gap larger.

    Records with an ``error`` are skipped (they have no metrics). Non-finite Sharpe ranks as
    ``-inf`` (see ``_val_sharpe``); ties go to the EARLIEST ``iter`` so a re-run of the same
    configuration cannot displace the original and the choice is reproducible. Returns the
    record itself because the loop's progress line needs its ``iter``; ``_select_best`` is the
    copying, config-only view for callers outside the loop.
    """
    best: dict[str, Any] | None = None
    best_key: tuple[float, int] | None = None
    for record in history:
        if record.get("error") is not None:
            continue
        # Higher Sharpe wins; on equal Sharpe the smaller iter wins (hence the negation).
        key = (_val_sharpe(record), -int(record["iter"]))
        if best_key is None or key > best_key:
            best, best_key = record, key
    return best


def _select_best(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The winning configuration (``_CONFIG_KEYS`` of ``_best_record``) as fresh copies, or None.

    Ranking rules live in ``_best_record``. This returns copies of the four config keys — never
    aliases of the history record — so the caller (or the UI) mutating the result cannot rewrite
    the timeline.
    """
    best = _best_record(history)
    if best is None:
        return None
    return {k: copy.deepcopy(best[k]) for k in _CONFIG_KEYS}


def _round_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Round each metric to ``_RESULT_DECIMALS``; NaN/inf become ``None`` (JSON ``null``).

    ``json.dumps`` would otherwise emit the bare tokens ``NaN``/``Infinity``, which are not JSON
    and which some tool-use parsers reject; ``null`` is unambiguous to the model ("no number").
    """
    rounded: dict[str, float | None] = {}
    for name, value in metrics.items():
        number = float(value)
        rounded[name] = round(number, _RESULT_DECIMALS) if math.isfinite(number) else None
    return rounded


def _tool_result_text(record: dict[str, Any]) -> str:
    """The metrics summary the model sees for one experiment: a small, sorted JSON document.

    Exactly ``{params, strategy, train_metrics, val_metrics}`` — no result_ids or dataset_ids
    (handles are meaningless to the model and would only invite it to ask for them), no prices,
    no dates (AR-6: the context carries metric summaries only). Metrics are rounded to
    ``_RESULT_DECIMALS`` so each result costs a few dozen tokens for the rest of the loop;
    ``sort_keys`` keeps the bytes deterministic, which the transcript's cache prefix depends on.
    ``default=str`` is a last-resort fallback for a non-JSON scalar (a numpy type leaking through
    a param, say) — better a string than a crash after two backtests have already run.
    """
    payload = {
        "strategy": record["strategy"],
        "params": record["params"],
        "train_metrics": _round_metrics(record["train_metrics"]),
        "val_metrics": _round_metrics(record["val_metrics"]),
    }
    return json.dumps(payload, sort_keys=True, default=str)


# --------------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------------

#: Why ``_research_loop`` stopped. The component-13 design doc lists the first three; ``api_error``
#: is a build decision: an SDK failure mid-loop (network, 5xx after the SDK's own retries, 429)
#: is neither convergence nor a cap, and the history gathered before it is still real, paid-for
#: work the caller should get back rather than an exception that discards it.
STOP_REASONS = ("converged", "max_iters", "budget", "api_error")

#: Forced tool use, one call per turn. ``any`` makes a prose-only turn impossible in the common
#: case and ``disable_parallel_tool_use`` makes a multi-call turn impossible in the common case;
#: the loop still handles both, because a cap on paid turns must not depend on the API's manners.
_TOOL_CHOICE: dict = {"type": "any", "disable_parallel_tool_use": True}

_NO_TOOL_TEXT = "You must call exactly one tool: propose_experiment or declare_done."
_IGNORED_TOOL_TEXT = "ignored: one tool call per turn"
_DONE_TOO_EARLY_TEXT = "run at least one successful experiment before declaring done"
_UNKNOWN_TOOL_TEXT = "unknown tool"


def _tool_result(tool_use_id: str, content: str, *, is_error: bool = False) -> dict:
    """A ``tool_result`` block; ``is_error`` is only present when true (the API's default)."""
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def _progress_text(turn: int, max_iters: int, history: list[dict[str, Any]]) -> str:
    """The turn counter + best-so-far line appended after every turn.

    The model is told how many turns remain because a bounded budget it can see produces
    converge-or-refine decisions; one it cannot see produces a loop cut off mid-search at the
    cap. The best-so-far Sharpe is restated (rather than left for the model to re-derive from
    the transcript) so the objective it is optimising is always the runner's number.
    """
    head = f"Turn {turn} of {max_iters} used; {max_iters - turn} remaining."
    best = _best_record(history)
    if best is None:
        return f"{head} No successful experiment yet."
    sharpe = float(best["val_metrics"]["sharpe"])
    return f"{head} Best validation Sharpe so far: {sharpe:.4f} (turn {best['iter']})."


def _make_record(
    iter_: int,
    strategy: Any,
    params: Any,
    train_metrics: dict | None,
    val_metrics: dict | None,
    rationale: str,
    error: str | None,
) -> dict[str, Any]:
    """One history record, keys in ``_RECORD_KEYS`` order (the shape the UI timeline renders)."""
    record = {
        "iter": iter_,
        "strategy": strategy,
        "params": params,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "rationale": rationale,
        "error": error,
    }
    assert tuple(record) == _RECORD_KEYS
    return record


def _run_proposal(
    turn: int,
    proposal: Any,
    *,
    train_id: str,
    val_id: str,
    engine: str,
    cost_bps: float,
) -> dict[str, Any]:
    """Turn one ``propose_experiment`` input into a history record — success or rejection.

    A rejection is a record, not an exception: the turn was paid for and the model must be told
    exactly why it failed so the next proposal can fix it. The ``error`` text is the tool's
    ValueError message verbatim (it names the offending param and the allowed range), with one
    exception — a ``PublicModeViolation`` is prefixed with "public mode violation:" because the
    guardrail's own message ("params rejected by whitelist: ...") does not say which layer
    refused, and the timeline should make it visible that the public-mode gate fired, not just
    the whitelist. A non-dict input (the API guarantees an object, but the loop does not rely on
    that) is recorded as a rejection with no strategy so nothing runs.
    """
    if not isinstance(proposal, dict):
        return _make_record(
            turn,
            None,
            {},
            None,
            None,
            "",
            f"propose_experiment input must be an object, got {type(proposal).__name__}",
        )
    strategy = proposal.get("strategy")
    params = proposal.get("params") or {}
    rationale = str(proposal.get("rationale") or "")
    try:
        result = _run_experiment(train_id, val_id, strategy, params, engine, cost_bps)
    except guardrails.PublicModeViolation as err:
        return _make_record(
            turn, strategy, params, None, None, rationale, f"public mode violation: {err}"
        )
    except ValueError as err:
        return _make_record(turn, strategy, params, None, None, rationale, str(err))
    return _make_record(
        turn,
        result["strategy"],
        result["params"],
        result["train_metrics"],
        result["val_metrics"],
        rationale,
        None,
    )


def _research_loop(
    goal: str,
    *,
    train_id: str,
    val_id: str,
    engine: str,
    cost_bps: float,
    max_iters: int,
    client: Any,
) -> dict[str, Any]:
    """The capped, metered Sonnet tool-use conversation; returns the history and why it stopped.

    One iteration is ONE model call, so ``max_iters`` bounds SDK requests (``_call`` invocations)
    exactly — that is the SF-7 cap, counted in the unit that costs money, not in "experiments"
    (a turn the model wastes on prose or a rejected proposal still consumed a paid call). Before
    every call ``budget.allow`` is asked with a conservative estimate that grows with the
    transcript; a denial stops the loop with nothing sent and nothing charged (SF-1). An SDK
    error stops it too, keeping the history gathered so far (see ``STOP_REASONS``).

    Per turn: the assistant message is appended AS-IS (the full ``response.content`` list, so
    thinking blocks round-trip unchanged — required for tool use with adaptive thinking); the
    FIRST ``tool_use`` block is acted on; every additional one gets an ``is_error`` result (the
    API requires a result for every ``tool_use`` id, and answering only one would be a malformed
    request); and one user message carrying all the results plus a turn counter closes the
    turn. ``declare_done`` before any successful experiment is refused, because a run whose best
    configuration is "nothing" has no holdout score to report.

    Isolation (RG-4 / SF-8): this function receives dataset handles for the train and validation
    windows only. It never sees the holdout — not a handle, not a date — and the conversation it
    builds contains only the goal, the frozen prompt material, metric summaries and turn counters.
    Scoring the holdout is the caller's job, after this returns, outside any conversation.

    Returns ``{'history', 'stopped_because', 'spend_usd', 'error', 'n_model_calls'}``;
    ``n_model_calls`` counts every SDK request attempted, including a failed one.
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Research goal: {goal}\n\nYou have {max_iters} turns in total; this is turn 1. "
                "Call propose_experiment to test a configuration, or declare_done when finished."
            ),
        }
    ]
    history: list[dict[str, Any]] = []
    spend = 0.0
    n_model_calls = 0
    error: str | None = None

    for turn in range(1, max_iters + 1):
        est = budget.estimate(
            MODEL, _EST_TOKENS_IN_BASE + _EST_TOKENS_IN_PER_TURN * (turn - 1), _EST_TOKENS_OUT
        )
        if not budget.allow(est):
            stopped_because = "budget"
            break

        n_model_calls += 1
        try:
            response, usd = _call(
                client,
                system=SYSTEM_RESEARCH,
                messages=messages,
                tools=TOOLS,
                tool_choice=_TOOL_CHOICE,
                max_tokens=_MAX_TOKENS,
            )
        except anthropic.APIError as err:
            logger.warning("research loop: API error on turn %d: %s", turn, err)
            stopped_because = "api_error"
            error = str(err)
            break
        spend += usd
        messages.append({"role": "assistant", "content": response.content})

        blocks = _find_tool_use_blocks(response)
        if not blocks:
            # A wasted turn: no experiment, but the call was paid for and counts toward the cap.
            messages.append(
                {
                    "role": "user",
                    "content": f"{_NO_TOOL_TEXT} {_progress_text(turn, max_iters, history)}",
                }
            )
            continue

        first, extra = blocks[0], blocks[1:]
        results: list[dict[str, Any]] = []
        if first.name == "declare_done":
            if not any(record["error"] is None for record in history):
                results.append(_tool_result(first.id, _DONE_TOO_EARLY_TEXT, is_error=True))
            else:
                stopped_because = "converged"
                break
        elif first.name == "propose_experiment":
            record = _run_proposal(
                turn,
                first.input,
                train_id=train_id,
                val_id=val_id,
                engine=engine,
                cost_bps=cost_bps,
            )
            history.append(record)
            if record["error"] is None:
                results.append(_tool_result(first.id, _tool_result_text(record)))
            else:
                results.append(_tool_result(first.id, record["error"], is_error=True))
        else:
            results.append(_tool_result(first.id, _UNKNOWN_TOOL_TEXT, is_error=True))
        for block in extra:
            results.append(_tool_result(block.id, _IGNORED_TOOL_TEXT, is_error=True))

        messages.append(
            {
                "role": "user",
                "content": [
                    *results,
                    {"type": "text", "text": _progress_text(turn, max_iters, history)},
                ],
            }
        )
    else:
        stopped_because = "max_iters"

    return {
        "history": history,
        "stopped_because": stopped_because,
        "spend_usd": spend,
        "error": error,
        "n_model_calls": n_model_calls,
    }


# --------------------------------------------------------------------------------------------
# The public entry point: windows -> loop -> best -> ONE holdout score -> result contract
# --------------------------------------------------------------------------------------------

#: The result contract, in order: the five keys in ``docs/components/13-ai-agent.md`` plus
#: ``error`` (``None`` unless ``stopped_because == "api_error"``). Every value is plain JSON —
#: no handles, frames or registry ids — so the UI can cache a run as a scenario file as-is.
_RESULT_KEYS = ("best", "holdout_metrics", "history", "stopped_because", "spend_usd", "error")


def run_research(
    goal: str,
    *,
    max_iters: int = guardrails.MAX_AGENT_ITERS,
    tickers: list[str] | None = None,
    engine: str = "python",
    cost_bps: float = _DEFAULT_COST_BPS,
    client: Any = None,
) -> dict[str, Any]:
    """Run the bounded research loop, then score its winner ONCE on the holdout; return both.

    The loop (``_research_loop``): the model proposes a ``{strategy, params}`` experiment, the
    runner backtests it on the train window and then on the validation window through the MCP
    ``run_backtest`` tool, both metric sets go back to the model, and it refines or declares
    convergence — until it converges, ``max_iters`` model calls are used, ``budget.allow`` says
    no, or the SDK fails. ``stopped_because`` is one of ``STOP_REASONS`` (``converged``,
    ``max_iters``, ``budget``, ``api_error``), and ``error`` carries the SDK message only in the
    last case; the history gathered before any stop is returned, never discarded.

    Everything the agent can touch is fixed before it runs. The goal, iteration cap, engine name,
    cost and tickers are validated first so a bad call costs nothing — no client is built (so an
    unknown ticker fails the same way with or without credentials), no data is loaded, no model
    is called. The agent's data then enters ONLY through the ``load_data`` tool
    with the loader's train and validation bounds (RG-4; layer two of SF-8 is the tool itself
    rejecting any other window), so the loop holds two dataset handles and nothing else. The
    tool's own errors (a ticker outside the universe, a window with no rows) propagate unchanged.

    Best is the record with the highest VALIDATION Sharpe (``_select_best``; the reasoning is in
    ``_best_record``): train Sharpe is the number the model was optimising with the train
    metrics in front of it, so it is biased upward by construction; validation is the first
    data the proposal was not tuned against. Picking by train would make the validation numbers
    decorative and the holdout gap larger.

    The holdout is scored by the RUNNER, never by the agent, and only after the loop has
    returned: the model saw validation results across iterations too, so even the validation
    Sharpe is mildly optimistic, and the only honest number is one from data no part of the
    loop ever saw. The ``HoldoutHandle`` is created here (a local, after the conversation is
    over), scored exactly once, and never stored in the result. When the agent was given a
    ticker subset, the holdout is cut from that same subset — scoring the winner on a different
    universe would not measure the configuration the agent actually iterated on. The holdout is
    always scored at ``guardrails._HOLDOUT_COST_BPS`` (10 bps) regardless of the ``cost_bps``
    the loop used, because the holdout number must not be tunable — a caller who could lower
    the cost for the final score alone would have a knob that only ever flatters it. An empty
    holdout (data ending before the holdout window) raises from ``score_holdout``: a missing
    holdout is a data-configuration fault the caller must see, never a silently missing number.
    If no proposal succeeded there is nothing to score, and neither ``split_data`` nor
    ``score_holdout`` is called.

    Honest-reporting rule: the UI must show ``best["val_metrics"]`` and ``holdout_metrics`` side
    by side. A large validation-to-holdout gap is not a failure to hide — it IS the demo's
    teaching moment (the loop overfit to the windows it could see).

    PUBLIC_MODE is read from the environment by the guardrails, never from a kwarg: a caller —
    or a prompt-injected one — must not be able to switch the parameter-only gate off per call.

    Returns exactly ``_RESULT_KEYS``: ``best`` (``{strategy, params, train_metrics,
    val_metrics}`` or ``None``), ``holdout_metrics`` (a ``performance._KEYS`` dict or ``None``),
    ``history`` (one record per proposal, see ``_RECORD_KEYS``), ``stopped_because``,
    ``spend_usd`` (what this run charged the ledger) and ``error``. JSON-serialisable throughout.
    """
    # All validation before any spend: nothing below this block may raise on a bad argument.
    goal = _validate_goal(goal)
    max_iters = _validate_max_iters(max_iters)
    # ``isinstance`` first: an unhashable engine (a list, say) would otherwise raise TypeError
    # from the ``in`` test instead of the documented ValueError.
    if not isinstance(engine, str) or engine not in mcp_server.ENGINES:
        raise ValueError(
            f"unknown engine {engine!r}; registered engines: {sorted(mcp_server.ENGINES)}"
        )
    # The first run_backtest would reject a bad cost too, but only after a paid model call.
    mcp_server._validate_cost_bps(cost_bps)
    # load_data re-validates tickers below (same function, same message); doing it here as well
    # is what keeps a bad ticker from needing an SDK client — and credentials — to be rejected.
    if tickers is not None:
        mcp_server._validate_tickers(tickers)
    client = _get_client() if client is None else client

    # The agent's data enters only through the load_data tool, on the fixed train and
    # validation bounds (RG-4). The tool's ValueErrors (bad ticker, no rows) propagate.
    bounds = get_split_bounds()
    train = mcp_server.load_data(bounds["train"][0], bounds["train"][1], tickers)
    val = mcp_server.load_data(bounds["validation"][0], bounds["validation"][1], tickers)

    loop = _research_loop(
        goal,
        train_id=train["dataset_id"],
        val_id=val["dataset_id"],
        engine=engine,
        cost_bps=cost_bps,
        max_iters=max_iters,
        client=client,
    )
    best = _select_best(loop["history"])

    # Holdout: outside and after the loop, once, by the runner. The handle is a local that
    # exists only from here to the score call and is never part of the result.
    holdout_metrics: dict[str, float] | None = None
    if best is not None:
        prices = mcp_server._price_source()
        if tickers is not None:
            prices = prices[prices["ticker"].isin(tickers)].reset_index(drop=True)
        _, _, handle = guardrails.split_data(prices)
        scored = guardrails.score_holdout(handle, best["strategy"], best["params"], engine)
        holdout_metrics = {str(k): float(v) for k, v in scored.items()}

    result = {
        "best": best,
        "holdout_metrics": holdout_metrics,
        "history": loop["history"],
        "stopped_because": loop["stopped_because"],
        "spend_usd": float(loop["spend_usd"]),
        "error": loop["error"],
    }
    assert tuple(result) == _RESULT_KEYS
    return result
