"""Two kinds of guardrails in one place: anti-overfitting (RG-4/SF-8) and public-demo safety.

Why this module exists
----------------------
The research agent is an LLM that iterates on strategy parameters. Left alone it will do exactly
what a careless quant does: tune until the *test* set looks good. That is p-hacking, and it is the
single failure that would make every number this project reports meaningless. So the statistical
guardrails here are not defensive polish — they *are* the rigor claim:

* ``split_data`` slices the panel by the FIXED dates in ``loader.get_split_bounds()``
  (2010–19 train / 2020–22 validation / 2023+ holdout). There are no ratio arguments on purpose:
  one split definition, defined in one place (RG-4).
* The holdout never comes back as a DataFrame. It lives inside a closure created by ``split_data``
  and is reachable only through :class:`HoldoutHandle`, whose one capability is "score a vetted
  strategy on me". ``vars()``, ``repr()``, ``dir()``, ``pickle`` and attribute access all come up
  empty (SF-8) — proven by ``tests/test_holdout_isolation.py``.
* ``score_holdout`` runs **once** per handle. The ``consumed`` flag flips *before* the run, so a
  crash mid-run still burns the shot: the agent cannot "retry" its way into a second look.

The public-demo guardrails exist because the app is a public URL with paid AI hooks behind it:

* ``public_mode()`` reads ``PUBLIC_MODE`` at CALL time (the stub read it at import time, which
  made it impossible to test with ``monkeypatch`` and impossible to flip without a restart).
* ``assert_no_codegen`` is the SF-3 gate: in public mode the only action shape that may reach the
  pipeline is ``{"strategy": <vetted name>, "params": <whitelisted scalars>}``. Nothing else —
  no ``code``, ``source``, ``python``, ``eval``, ``exec``, ``file`` keys, no nested payloads.
* ``rate_limit`` is the SF-5 nuisance gate (per-session/IP calls per hour); the money gate is
  ``budget.allow``. They are separate because they fail differently: a rate-limited user waits an
  hour, a budget-limited *deployment* is off for the day.
* ``MAX_AGENT_ITERS`` (SF-7) lives here so the cap is a guardrail constant, not an agent knob.

The holdout's opacity is enforced twice, independently: in-process here (closure + one-shot flag)
and structurally in the MCP ``load_data`` tool, which rejects (never truncates) any date range
outside train+validation. Even a prompt-injected agent cannot ask the tools for 2023+ data.

Environment variables (see ``docs/components/18-runtime-config.md``)
--------------------------------------------------------------------
``PUBLIC_MODE``             ``on`` (case-insensitive) enables parameter-only enforcement.
``AI_RATE_LIMIT_PER_HOUR``  sliding-window cap per ``rate_limit`` key; default 20.
``AI_LEDGER_PATH``          the rate-limit state lives next to the budget ledger, as
                            ``ai_rate_limits.json`` — one config knob relocates both files.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from quantforge import interchange
from quantforge.ai import budget
from quantforge.data import loader
from quantforge.strategies import STRATEGIES, validate_params

logger = logging.getLogger(__name__)

#: Hard cap on agent iterations (product.md §5). ``agent.run_research`` obeys it; it is a
#: guardrail constant rather than an agent parameter so nobody can raise it per-call.
MAX_AGENT_ITERS = 10

#: The vetted set. Kept as a literal (not derived from ``STRATEGIES``) so that adding a strategy
#: to the registry without consciously vetting it here fails ``test_strategy_registry_verifier``.
VETTED_STRATEGIES = {"momentum", "mean_reversion"}

#: Transaction-cost assumption used for the one holdout score. Fixed, not a parameter: the
#: holdout number is the one figure that must not be tunable after the fact.
_HOLDOUT_COST_BPS = 10.0

_DEFAULT_RATE_LIMIT_PER_HOUR = 20
_RATE_WINDOW_SECONDS = 3600.0
_RATE_STATE_FILENAME = "ai_rate_limits.json"

# Scalar types a public-mode param value may have. Nested dicts/lists are rejected because a
# whitelist of scalars is auditable; a payload that can carry structure can carry code.
_SCALAR_TYPES = (int, float, str, bool)


class HoldoutAlreadyScored(RuntimeError):
    """The one holdout score has already been taken from this handle (RG-4)."""


class PublicModeViolation(ValueError):
    """An action reached the pipeline in PUBLIC_MODE that is not parameter-only (SF-3)."""


def public_mode() -> bool:
    """True when ``PUBLIC_MODE=on`` (case-insensitive), read from the environment on every call.

    Call-time rather than import-time so the operator can flip the mode without a restart and so
    tests can prove both branches with ``monkeypatch.setenv``.
    """
    return os.getenv("PUBLIC_MODE", "off").strip().lower() == "on"


# --------------------------------------------------------------------------------------------
# Statistical guardrails: split + opaque holdout
# --------------------------------------------------------------------------------------------


class HoldoutHandle:
    """Opaque, single-use capability to score the holdout slice — never the slice itself.

    Constructed ONLY by :func:`split_data`. The holdout DataFrame is captured in the ``_score``
    closure that ``split_data`` builds; it is not an attribute, so there is nothing for the agent
    (or an over-curious ``repr``/``vars``/``dir``/debugger) to read. ``__slots__`` removes
    ``__dict__`` entirely, pickling is refused (a pickle would serialise the closure's captured
    frame), and no ``__iter__``/``__getitem__``/``__len__`` are defined so the object cannot be
    mistaken for a container.

    The closure is a guard against accidental or tool-mediated access (an agent, a tool result,
    a debugger repr, a pickle) — not a defence against deliberate in-process introspection of
    the closure cell by code that already runs in this process; the structural guarantee is the
    MCP ``load_data`` rejection, proven in ``tests/test_holdout_isolation.py``.

    What *is* public: ``n_days``, ``start``, ``end`` (the holdout's date range is a documented,
    fixed fact — see ``loader.SPLITS``) and ``consumed``. Dates are public knowledge; prices are not.
    """

    __slots__ = ("consumed", "n_days", "start", "end", "_score")

    def __init__(
        self,
        *,
        score: Callable[[str, dict, str], dict[str, float]],
        n_days: int,
        start: str,
        end: str,
    ) -> None:
        self._score = score
        self.n_days = int(n_days)
        self.start = start
        self.end = end
        self.consumed = False

    def __repr__(self) -> str:
        # Metadata only — no price values can ever appear here (asserted by the isolation test).
        return (
            f"HoldoutHandle(start={self.start}, end={self.end}, "
            f"n_days={self.n_days}, consumed={self.consumed})"
        )

    __str__ = __repr__

    def __getstate__(self):
        raise TypeError("holdout handle is not picklable")

    def __reduce__(self):
        raise TypeError("holdout handle is not picklable")


def _slice_long(prices: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    """Inclusive [start, end] slice of a long prices frame by date (bounds as ISO strings).

    The bounds are parsed as midnight UTC to match the daily UTC timestamps the interchange
    contract mandates, so both endpoint days are kept — the same convention as ``load_prices``.
    """
    lo, hi = pd.Timestamp(bounds[0], tz="UTC"), pd.Timestamp(bounds[1], tz="UTC")
    return prices[prices["date"].between(lo, hi)].reset_index(drop=True)


def _resolve_engine(name: str):
    """Look ``name`` up in ``mcp_server.ENGINES``; unknown names raise ValueError.

    The import is function-local on purpose: ``mcp_server.py`` imports this module at top
    level (``from quantforge.ai import guardrails``, for ``assert_no_codegen`` in
    ``run_backtest``), so a top-level import of ``mcp_server`` here would be a cycle. Resolving
    through the registry — never a concrete class — keeps the holdout score engine-agnostic
    (AR-1/AR-4).
    """
    from quantforge.ai.mcp_server import ENGINES

    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; registered engines: {sorted(ENGINES)}")
    return ENGINES[name]


def split_data(
    prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, HoldoutHandle]:
    """Split a long ``prices`` frame into (train_wide, validation_wide, holdout_handle).

    ``prices`` is a long interchange ``"prices"`` frame — the loader's output — and defaults to
    ``loader.load_prices()``. It is validated against the contract first so a malformed frame
    fails here with a schema error rather than as a KeyError inside the slicing.

    The three slices are cut by the FIXED inclusive dates in ``loader.get_split_bounds()``; the
    slices are disjoint and together cover every in-range row (rows outside 2010-01-01..END are
    dropped). Train and validation come back as wide date x ticker frames, ready for
    ``Strategy.generate_signals``. The holdout comes back only as a :class:`HoldoutHandle`: the
    wide holdout frame is bound into the ``_score`` closure below and nothing else references it.

    A holdout with zero rows (synthetic data ending before 2023) yields a handle with
    ``n_days == 0``; scoring it raises ``ValueError("holdout slice is empty")``.
    """
    if prices is None:
        prices = loader.load_prices()
    interchange.validate_frame(prices, "prices")

    bounds = loader.get_split_bounds()
    train_long = _slice_long(prices, bounds["train"])
    validation_long = _slice_long(prices, bounds["validation"])
    holdout_long = _slice_long(prices, bounds["holdout"])

    train = interchange.to_wide(train_long, "prices")
    validation = interchange.to_wide(validation_long, "prices")

    if holdout_long.empty:
        # to_wide on an empty frame gives an empty pivot; keep an explicitly empty wide frame so
        # the closure's emptiness check is unambiguous.
        holdout = pd.DataFrame()
        n_days, start, end = 0, bounds["holdout"][0], bounds["holdout"][1]
    else:
        holdout = interchange.to_wide(holdout_long, "prices")
        n_days = int(holdout.shape[0])
        start = holdout.index[0].date().isoformat()
        end = holdout.index[-1].date().isoformat()

    def _score(strategy: str, params: dict, engine_name: str) -> dict[str, float]:
        """Run ``strategy`` on the captured holdout frame and return ONLY the metrics dict.

        This closure is the sole reference to ``holdout``. It returns ``result.metrics`` and
        nothing else — not the result object (its ``returns``/``equity_curve`` would leak the
        holdout's daily path), never the frame.
        """
        if holdout.empty:
            raise ValueError("holdout slice is empty")
        engine = _resolve_engine(engine_name)
        positions = STRATEGIES[strategy]().generate_signals(holdout, params)
        result = engine.run_backtest(
            holdout, positions, {"cost_bps": _HOLDOUT_COST_BPS, "strategy": strategy, **params}
        )
        return dict(result.metrics)

    handle = HoldoutHandle(score=_score, n_days=n_days, start=start, end=end)
    return train, validation, handle


def score_holdout(
    handle: HoldoutHandle, strategy: str, params: dict, engine: str = "python"
) -> dict[str, float]:
    """Score ``strategy``/``params`` on the hidden holdout exactly once; return the metrics dict.

    Called by the research runner AFTER the agent loop has finished — never by the agent. Order of
    operations matters and is deliberate:

    1. ``validate_params`` and engine resolution run first, so a typo in the request (unvetted
       strategy, out-of-range param, unknown engine) raises ``ValueError`` *without* spending the
       handle's single shot — those are caller mistakes, not attempts at a second look.
    2. If the handle is already consumed, raise :class:`HoldoutAlreadyScored`.
    3. Mark the handle consumed BEFORE running. One shot means one shot: a run that raises
       halfway (bad data, engine bug) still burns it, otherwise "make it crash, then retry" would
       be a way to peek twice.

    Returns a dict with exactly the keys in ``metrics.performance._KEYS``.
    """
    if not isinstance(handle, HoldoutHandle):
        raise TypeError(f"expected a HoldoutHandle from split_data, got {type(handle).__name__}")
    validated = validate_params(strategy, params)
    _resolve_engine(engine)  # unknown engine -> ValueError, before the shot is spent
    if handle.consumed:
        raise HoldoutAlreadyScored(
            "the holdout has already been scored once; it is never scored again (RG-4)"
        )
    handle.consumed = True
    return handle._score(strategy, validated, engine)


# --------------------------------------------------------------------------------------------
# Public-demo safety: rate limit
# --------------------------------------------------------------------------------------------


def _now() -> float:
    """Unix time; a separate function so tests can advance the clock past the window."""
    return time.time()


def _rate_limit_per_hour() -> int:
    """Read ``AI_RATE_LIMIT_PER_HOUR`` at call time; blank/unparseable/negative → default 20."""
    raw = os.getenv("AI_RATE_LIMIT_PER_HOUR")
    if raw is None or not raw.strip():
        return _DEFAULT_RATE_LIMIT_PER_HOUR
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "AI_RATE_LIMIT_PER_HOUR=%r is not an integer; using %d",
            raw,
            _DEFAULT_RATE_LIMIT_PER_HOUR,
        )
        return _DEFAULT_RATE_LIMIT_PER_HOUR
    if value < 0:
        logger.warning(
            "AI_RATE_LIMIT_PER_HOUR=%r is negative; using %d", raw, _DEFAULT_RATE_LIMIT_PER_HOUR
        )
        return _DEFAULT_RATE_LIMIT_PER_HOUR
    return value


def _rate_state_path() -> Path:
    """``ai_rate_limits.json`` beside the budget ledger — derived, so there is no second env var."""
    return budget._ledger_path().with_name(_RATE_STATE_FILENAME)


def _read_rate_state(path: Path) -> dict[str, list[float]]:
    """Load ``{key: [unix_timestamps]}``; missing → empty. Must be called under ``budget._locked``.

    A corrupt file is treated as empty (with a warning) in every mode, unlike the budget ledger.
    The asymmetry is deliberate: resetting rate-limit state costs at most one hour of extra calls
    for one key, and the money gate (``budget.allow``) still stands behind it — whereas silently
    resetting *spend* history would lift a hard dollar cap.
    """
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("rate-limit state at %s unreadable (%s); treating as empty", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("rate-limit state at %s has wrong shape; treating as empty", path)
        return {}
    state: dict[str, list[float]] = {}
    for key, stamps in data.items():
        if isinstance(key, str) and isinstance(stamps, list):
            state[key] = [float(t) for t in stamps if isinstance(t, (int, float))]
    return state


def rate_limit(key: str) -> bool:
    """Return True (and record the call) if ``key`` has made fewer than the hourly limit of calls.

    Sliding 3600-second window: timestamps older than an hour are pruned on every call, so a
    denied user is admitted again exactly one hour after their oldest surviving call, not at a
    fixed wall-clock boundary. If the count is already at the limit the call is denied and NOT
    recorded — denials must not extend the lockout.

    State is a JSON file next to the budget ledger, guarded by the same thread + file lock and
    written atomically via ``budget._write_ledger`` (a generic JSON writer despite its name), so
    a crash mid-write cannot leave a half-file and concurrent Streamlit sessions cannot lose
    each other's writes.
    """
    if not isinstance(key, str) or not key:
        raise ValueError(f"rate_limit key must be a non-empty string, got {key!r}")
    limit = _rate_limit_per_hour()
    now = _now()
    path = _rate_state_path()
    with budget._locked(path):
        state = _read_rate_state(path)
        recent = [t for t in state.get(key, []) if now - t < _RATE_WINDOW_SECONDS]
        allowed = len(recent) < limit
        if allowed:
            recent.append(now)
        state[key] = recent
        budget._write_ledger(path, state)
    return allowed


# --------------------------------------------------------------------------------------------
# Public-demo safety: parameter-only actions
# --------------------------------------------------------------------------------------------


def assert_no_codegen(action: dict[str, Any]) -> None:
    """In PUBLIC_MODE, raise :class:`PublicModeViolation` unless ``action`` is parameter-only.

    The only accepted shape is ``{"strategy": <name in VETTED_STRATEGIES>, "params": <dict of
    int/float/str/bool>}`` whose params pass ``strategies.validate_params``. The key set must be
    EXACTLY ``{"strategy", "params"}`` — so ``code``, ``source``, ``python``, ``eval``, ``exec``,
    ``file`` (or any other extra key) are rejected without needing a blocklist to name them.
    Param values must be flat scalars: a nested dict/list, a callable, or bytes has no legitimate
    use in a whitelisted-knob call and is exactly the kind of container a code payload rides in.

    Outside public mode this is a no-op: local research mode is allowed to run generated code on
    the author's own machine. That laxity never ships — ``PUBLIC_MODE=on`` is set in the deploy.
    """
    if not public_mode():
        return None
    if not isinstance(action, dict):
        raise PublicModeViolation(
            f"public mode accepts only a {{'strategy', 'params'}} dict, got {type(action).__name__}"
        )
    keys = set(action)
    if keys != {"strategy", "params"}:
        raise PublicModeViolation(
            f"public mode accepts exactly the keys {{'strategy', 'params'}}, got {sorted(map(str, keys))}"
        )
    strategy = action["strategy"]
    params = action["params"]
    if not isinstance(strategy, str) or strategy not in VETTED_STRATEGIES:
        raise PublicModeViolation(
            f"strategy {strategy!r} is not vetted; allowed: {sorted(VETTED_STRATEGIES)}"
        )
    if not isinstance(params, dict):
        raise PublicModeViolation(f"params must be a dict, got {type(params).__name__}")
    for name, value in params.items():
        if not isinstance(name, str):
            raise PublicModeViolation(f"param names must be strings, got {name!r}")
        if not isinstance(value, _SCALAR_TYPES):
            raise PublicModeViolation(
                f"param {name!r} must be a flat int/float/str/bool, got {type(value).__name__}"
            )
    try:
        validate_params(strategy, params)
    except ValueError as exc:
        raise PublicModeViolation(f"params rejected by whitelist: {exc}") from exc
    return None
