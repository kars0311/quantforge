"""Streamlit UI — week-5 shell: config sidebar + Backtest and Portfolio tabs with Plotly charts.

Built incrementally from week 5 (this shell + core charts) through week 9 (AI panel + engine
selector + deploy). The AI Chat and Research mode tabs are informative placeholders until weeks
7–8; the R/KNIME engine paths are visible-but-disabled labels until week 6+. Nothing under
``quantforge.ai`` is imported here yet — the week-5 UI must be importable (and deployable) with
zero AI dependencies, and tests assert that.

Layout (docs/components/14-streamlit-app.md): sidebar = configuration (tickers from the frozen
UNIVERSE, dates bounded to train+validation, strategy + params driven by PARAM_WHITELIST,
cost_bps, engine selector); main area = tabs Backtest · Portfolio · AI Chat · Research mode ·
Methodology.

Testability rule: chart builders (:func:`plot_equity`, :func:`plot_drawdown`,
:func:`plot_frontier`) and spec helpers (:func:`param_control_specs`,
:func:`equal_weight_benchmark`) are PURE — importing this module and calling them requires no
Streamlit server. Widget calls live only inside :func:`sidebar_config`, :func:`render_metrics`,
and the tab renderers invoked by :func:`main`.

SAFETY (PUBLIC_MODE): live AI gated by DEMO_PASSCODE; ungated visitors see cached scenarios only;
show remaining AI budget; never expose the API key client-side.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# `streamlit run app/streamlit_app.py` executes this file as a script with only app/ on
# sys.path — the src/ layout is invisible to it (the repo-root conftest.py performs the same
# insert, but only under pytest, and the package is deliberately not pip-installed). Insert
# src/ here so a fresh clone runs per AGENTS.md with no install step. Must precede the
# quantforge imports, hence the noqa on them.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from quantforge import interchange  # noqa: E402
from quantforge.data.loader import UNIVERSE, get_split_bounds, load_prices  # noqa: E402
from quantforge.engine.base import BacktestResult  # noqa: E402
from quantforge.engine.python_engine import PythonEngine  # noqa: E402
from quantforge.metrics.performance import _KEYS, compute_metrics  # noqa: E402
from quantforge.portfolio.optimize import combine_returns, frontier, optimize_weights  # noqa: E402
from quantforge.strategies import PARAM_WHITELIST, STRATEGIES, validate_params  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Sidebar bounds: interactive runs may touch train + validation ONLY (RG-4). The holdout is
# scored once at the end of the project and never optimized against — a UI date picker that
# could reach into it would let a user (or a screen-driving LLM) p-hack the test set one click
# at a time. Sourced from get_split_bounds() so a split change propagates here automatically.
# ---------------------------------------------------------------------------------------------
_SPLITS = get_split_bounds()
UI_DATE_MIN: date = date.fromisoformat(_SPLITS["train"][0])
UI_DATE_MAX: date = date.fromisoformat(_SPLITS["validation"][1])

#: Default sidebar selection: a cross-sector subset of UNIVERSE listed for the full window since
#: 2010, so the out-of-the-box demo has no pre-IPO NaN raggedness to explain on first contact.
_DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "JNJ", "XOM"]

#: Slider step for float params. 0.05 divides every whitelisted float range exactly, so the
#: whitelist bounds themselves are always reachable positions on the slider.
_FLOAT_STEP = 0.05

#: Human labels + formats for the metric row, keyed on metrics/performance._KEYS — the single
#: source of truth for metric names; this dict may only relabel, never add or drop a key.
_METRIC_LABELS = {
    "total_return": ("Total return", "{:.1%}"),
    "cagr": ("CAGR", "{:.1%}"),
    "ann_vol": ("Ann. vol", "{:.1%}"),
    "sharpe": ("Sharpe", "{:.2f}"),
    "max_drawdown": ("Max drawdown", "{:.1%}"),
    "hit_rate": ("Hit rate", "{:.1%}"),
}

#: The one canonical price cache file (see data/loader.py cache design). The UI checks for it
#: instead of letting load_prices fall through to a download: a public demo box must NEVER hit
#: the network on a visitor's click — a missing cache is an ops mistake to surface, not repair.
_CACHE_FILE = Path("data_cache") / "prices.parquet"


# ---------------------------------------------------------------------------------------------
# Pure helpers (no st.* calls) — unit-testable without a Streamlit server
# ---------------------------------------------------------------------------------------------


def param_control_specs(strategy: str) -> dict[str, dict]:
    """Translate PARAM_WHITELIST[strategy] into widget specs (kind, bounds, default, step).

    WHY this exists as a separate pure function: the whitelist is the single source of truth for
    what a parameter may be (SF-3), so the UI must *derive* its control ranges from it rather
    than repeat the numbers — a hand-copied slider bound that drifted from the whitelist would
    let the UI offer values ``validate_params`` then rejects (or, worse, hide values it allows).
    Tests assert this sourcing by mutating the whitelist and watching the specs follow.
    """
    if strategy not in PARAM_WHITELIST:
        raise ValueError(
            f"unknown strategy {strategy!r}; vetted strategies: {sorted(PARAM_WHITELIST)}"
        )
    specs: dict[str, dict] = {}
    for param, spec in PARAM_WHITELIST[strategy].items():
        if "choices" in spec:
            specs[param] = {
                "widget": "selectbox",
                "options": sorted(spec["choices"]),
                "default": spec["default"],
            }
        elif spec["type"] is int:
            specs[param] = {
                "widget": "slider",
                "min": spec["min"],
                "max": spec["max"],
                "default": spec["default"],
                "step": 1,
            }
        else:  # float param
            specs[param] = {
                "widget": "slider",
                "min": float(spec["min"]),
                "max": float(spec["max"]),
                "default": float(spec["default"]),
                "step": _FLOAT_STEP,
            }
    return specs


def equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    """Equity curve of a costless equal-weight portfolio over the SAME wide price window.

    WHY it runs through PythonEngine instead of a quick ``prices.pct_change().mean(axis=1)``:
    the engine is the one audited place where the no-look-ahead shift lives, so reusing it makes
    the benchmark shift-consistent with the strategy curve *by construction* — both curves buy
    at the close of day t and first earn day t+1's return, so comparing them is apples to
    apples. Costless (cost_bps=0) because the benchmark is a passive reference, not a strategy
    we claim is implementable for free; NaN (pre-IPO) prices contribute zero return under the
    engine's documented NaN policy, which is the honest treatment.
    """
    n = prices.shape[1]
    weights = pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)
    result = PythonEngine().run_backtest(prices, weights, {"cost_bps": 0.0})
    return result.equity_curve


def plot_equity(result: BacktestResult, prices: pd.DataFrame | None = None) -> go.Figure:
    """Equity curve, optionally vs the equal-weight buy-and-hold benchmark.

    ``prices`` is the wide price window the backtest ran on; when provided, the benchmark trace
    is computed from it via :func:`equal_weight_benchmark` (same window, same engine, same
    shift). It is optional so the doc-14 one-argument call still works where no benchmark
    context exists (e.g. the combined-portfolio curve, whose "window" is strategy returns, not
    prices).
    """
    fig = go.Figure()
    eq = result.equity_curve
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Strategy"))
    if prices is not None:
        bench = equal_weight_benchmark(prices)
        fig.add_trace(
            go.Scatter(
                x=bench.index,
                y=bench.values,
                mode="lines",
                name="Equal-weight benchmark",
                line={"dash": "dash"},
            )
        )
    fig.update_layout(
        title="Equity curve (growth of $1, net of costs)",
        yaxis_title="Portfolio value",
        xaxis_title="Date",
        legend={"orientation": "h"},
    )
    return fig


def plot_drawdown(result: BacktestResult) -> go.Figure:
    """Underwater chart: equity / running peak − 1 (≤ 0 everywhere, 0 at each new peak).

    Computed from the equity curve rather than stored separately so the chart can never
    disagree with the curve it annotates — same definition as ``max_drawdown`` in
    metrics/performance.py (equity / cummax − 1), the single source of truth.
    """
    eq = result.equity_curve
    dd = eq / eq.cummax() - 1.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd.values, mode="lines", fill="tozeroy", name="Drawdown")
    )
    fig.update_layout(
        title="Drawdown (underwater)",
        yaxis_title="Drawdown",
        xaxis_title="Date",
        yaxis_tickformat=".0%",
    )
    return fig


def plot_frontier(frontier_df: pd.DataFrame, weights: pd.Series) -> go.Figure:
    """Efficient-frontier scatter with the max-Sharpe point starred.

    ``frontier_df`` is the interchange 'frontier' frame (risk, ret; both annualized) from
    ``portfolio.optimize.frontier``; ``weights`` is the max-Sharpe allocation from
    ``optimize_weights`` and is rendered as the star's hover text so the chart answers "and
    what do I hold there?". With RISK_FREE = 0 (metrics/performance.py) the tangency portfolio
    is the frontier point maximizing ret/risk, so the star is located by that ratio over the
    frame itself — no re-optimization, no second estimator that could disagree with the sweep.
    """
    if len(frontier_df) == 0:
        raise ValueError("frontier_df is empty; nothing to plot")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier_df["risk"],
            y=frontier_df["ret"],
            mode="lines+markers",
            name="Efficient frontier",
        )
    )
    # risk == 0 cannot honestly have a Sharpe; mask it rather than divide by zero.
    risk = frontier_df["risk"].to_numpy(dtype=float)
    ret = frontier_df["ret"].to_numpy(dtype=float)
    sharpe = [r / s if s > 0 else float("-inf") for r, s in zip(ret, risk)]
    pos = int(max(range(len(sharpe)), key=sharpe.__getitem__))
    holdings = ", ".join(f"{name}: {w:.1%}" for name, w in weights.items())
    fig.add_trace(
        go.Scatter(
            x=[risk[pos]],
            y=[ret[pos]],
            mode="markers",
            marker={"symbol": "star", "size": 16},
            name="Max Sharpe",
            hovertext=[holdings],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Efficient frontier (annualized)",
        xaxis_title="Risk (ann. vol)",
        yaxis_title="Return (ann.)",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
    )
    return fig


# ---------------------------------------------------------------------------------------------
# Streamlit layer — widgets and rendering (only ever invoked via main())
# ---------------------------------------------------------------------------------------------


def render_metrics(metrics: dict) -> None:
    """One st.metric per name in metrics/performance._KEYS — exactly those, in that order.

    Iterating _KEYS (not the input dict) keeps the row in lockstep with the metrics module: a
    new metric added there appears here without a UI edit, and the UI can never invent a metric
    of its own. A missing key is a caller bug and raises rather than rendering a partial row.
    """
    missing = [k for k in _KEYS if k not in metrics]
    if missing:
        raise ValueError(f"metrics dict is missing required key(s) {missing}; expected {_KEYS}")
    cols = st.columns(len(_KEYS))
    for col, key in zip(cols, _KEYS):
        label, fmt = _METRIC_LABELS[key]
        value = metrics[key]
        text = (
            fmt.format(value) if isinstance(value, (int, float)) and math.isfinite(value) else "—"
        )
        col.metric(label, text)


def sidebar_config() -> dict:
    """Render the sidebar controls and return the validated run config.

    Returns ``{"tickers", "start", "end", "strategy", "params", "cost_bps", "engine"}`` with
    dates as ISO strings (stable st.cache_data keys) and ``params`` already passed through
    ``validate_params`` — the UI is one of the untrusted-input funnels into the SF-3 choke
    point, same as the MCP layer, so nothing widget-shaped reaches the engine unvalidated.
    """
    st.sidebar.header("Configuration")

    tickers = st.sidebar.multiselect("Tickers", options=UNIVERSE, default=_DEFAULT_TICKERS)

    # Date pickers hard-bounded to train+validation: holdout dates are UNSELECTABLE (RG-4).
    start = st.sidebar.date_input(
        "Start date", value=UI_DATE_MIN, min_value=UI_DATE_MIN, max_value=UI_DATE_MAX
    )
    end = st.sidebar.date_input(
        "End date", value=UI_DATE_MAX, min_value=UI_DATE_MIN, max_value=UI_DATE_MAX
    )
    st.sidebar.caption(
        f"Interactive runs are limited to train + validation "
        f"({UI_DATE_MIN} → {UI_DATE_MAX}). The holdout stays untouched until final scoring."
    )

    strategy = st.sidebar.selectbox("Strategy", options=list(STRATEGIES))

    # Param controls generated from the whitelist — bounds/defaults come from
    # param_control_specs (i.e. from PARAM_WHITELIST), never typed here.
    raw_params: dict = {}
    for param, spec in param_control_specs(strategy).items():
        if spec["widget"] == "selectbox":
            raw_params[param] = st.sidebar.selectbox(
                param,
                options=spec["options"],
                index=spec["options"].index(spec["default"]),
            )
        else:
            raw_params[param] = st.sidebar.slider(
                param,
                min_value=spec["min"],
                max_value=spec["max"],
                value=spec["default"],
                step=spec["step"],
            )
    params = validate_params(strategy, raw_params)

    cost_bps = st.sidebar.number_input(
        "Transaction cost (bps of turnover)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
    )

    # Engine selector: Python is the only live engine this week. R (week 6+) and KNIME
    # (stretch) are shown so the polyglot story is visible, but they are not selectable —
    # Streamlit selectboxes cannot disable individual options, so the honest rendering is a
    # one-option selector plus labeled disabled entries.
    engine = st.sidebar.selectbox(
        "Engine", options=["python"], format_func=lambda _: "Python (custom vectorized engine)"
    )
    st.sidebar.caption("R engine — week 6+ (disabled) · KNIME — stretch goal (disabled)")

    return {
        "tickers": list(tickers),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "strategy": strategy,
        "params": params,
        "cost_bps": float(cost_bps),
        "engine": engine,
    }


@st.cache_data(show_spinner="Loading cached prices…")
def _load_wide_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Cached: Parquet cache → long prices → wide date x ticker panel.

    Separate from run_pipeline so the Backtest tab can fetch the same window for the benchmark
    trace without re-reading the Parquet file — with identical arguments this is a cache hit.
    """
    long_prices = load_prices(list(tickers), start=start, end=end)
    return interchange.to_wide(long_prices, "prices")


@st.cache_data(show_spinner="Running backtest…")
def run_pipeline(cfg: dict) -> BacktestResult:
    """Cached full pipeline: load (cached parquet) → signals → engine → BacktestResult.

    st.cache_data keys on cfg, so repeat clicks with the same config are free — which is also
    why interactive (non-AI) backtests need no budget gate: they cost CPU only. load_prices is
    called normally; by the loader's cache design a primed deployment always cache-hits, so
    this path does no network I/O (the Methodology tab documents the primed-cache expectation,
    and the tab renderers refuse to run at all when the cache file is absent).
    """
    wide = _load_wide_prices(tuple(cfg["tickers"]), cfg["start"], cfg["end"])
    params = validate_params(cfg["strategy"], cfg["params"])
    strategy = STRATEGIES[cfg["strategy"]]()
    positions = strategy.generate_signals(wide, params)
    engine_params = {"cost_bps": cfg["cost_bps"], "strategy": cfg["strategy"], **params}
    return PythonEngine().run_backtest(wide, positions, engine_params)


def _config_problems(cfg: dict) -> list[str]:
    """Human-readable reasons this config cannot run (empty list = runnable)."""
    problems = []
    if not cfg["tickers"]:
        problems.append("Select at least one ticker.")
    if cfg["start"] > cfg["end"]:
        problems.append("Start date is after end date.")
    if not _CACHE_FILE.exists():
        problems.append(
            f"Price cache not found at `{_CACHE_FILE}`. Prime it once (outside the UI) with "
            '`python -c "from quantforge.data.loader import load_prices; load_prices()"` — '
            "the UI never downloads data itself (see Methodology)."
        )
    return problems


def _render_backtest_tab(cfg: dict) -> None:
    """Backtest tab: metric row + equity-vs-benchmark + underwater chart for one strategy."""
    problems = _config_problems(cfg)
    if problems:
        for p in problems:
            st.warning(p)
        return
    try:
        result = run_pipeline(cfg)
        wide = _load_wide_prices(tuple(cfg["tickers"]), cfg["start"], cfg["end"])
    except ValueError as exc:
        st.error(f"Backtest failed: {exc}")
        return
    render_metrics(result.metrics)
    st.plotly_chart(plot_equity(result, wide), use_container_width=True)
    st.plotly_chart(plot_drawdown(result), use_container_width=True)


def _render_portfolio_tab(cfg: dict) -> None:
    """Portfolio tab: both vetted strategies → optimizer → weights, frontier, combined equity.

    Both strategies run with their whitelist DEFAULT params (the sidebar's params belong to the
    single selected strategy; the portfolio view is the vetted-defaults combination story), over
    the sidebar's window/tickers/costs. Returns are inner-joined on dates both streams cover —
    the alignment rule proven in tests/test_portfolio_combination.py.
    """
    problems = _config_problems(cfg)
    if problems:
        for p in problems:
            st.warning(p)
        return
    try:
        panel = pd.concat(
            {
                name: run_pipeline(
                    {**cfg, "strategy": name, "params": validate_params(name, {})}
                ).returns
                for name in STRATEGIES
            },
            axis=1,
            join="inner",
        ).dropna()
        weights = optimize_weights(panel)  # default objective: max_sharpe
        frontier_df = frontier(panel)
        combined = combine_returns(panel, weights)
    except ValueError as exc:
        st.error(f"Portfolio optimization failed: {exc}")
        return

    st.subheader("Max-Sharpe weights across vetted strategies (default params)")
    st.dataframe(weights.rename("weight").to_frame().style.format("{:.1%}"))
    st.plotly_chart(plot_frontier(frontier_df, weights), use_container_width=True)

    combined_result = BacktestResult(
        equity_curve=(1.0 + combined).cumprod(),
        returns=combined,
        metrics=compute_metrics(combined),
        meta={"weights": weights.to_dict(), "objective": "max_sharpe"},
    )
    st.subheader("Combined portfolio")
    render_metrics(combined_result.metrics)
    st.plotly_chart(plot_equity(combined_result), use_container_width=True)


def _render_placeholder_tab(name: str, week: str, description: str) -> None:
    """Informative placeholder for a feature that lands in a later week — no AI imports here."""
    st.subheader(name)
    st.info(f"**Coming in {week}.** {description}")


def _render_methodology_tab() -> None:
    """Static methodology notes: the rigor story, honestly stated (no computation here)."""
    st.subheader("Methodology & caveats")
    st.markdown(
        f"""
**No look-ahead bias.** A weight decided using data through day *t* earns day *t+1*'s return —
the engine enforces this with `positions.shift(1)` in exactly one audited place. The
equal-weight benchmark on the Backtest tab runs through the same engine, so both curves obey
the same execution lag.

**Transaction costs** are charged on turnover (sum of absolute weight changes) at the
configured bps rate; all returns and metrics shown are net of costs.

**Survivorship bias.** The universe is a fixed, hand-picked list of *today's* names, not a
point-in-time constituent history — results are flattered by the exclusion of delisted
companies. Documented, not hidden.

**Data splits (enforced in the UI).** Train {_SPLITS["train"][0]} → {_SPLITS["train"][1]},
validation {_SPLITS["validation"][0]} → {_SPLITS["validation"][1]}, holdout
{_SPLITS["holdout"][0]} → {_SPLITS["holdout"][1]}. The date pickers cannot reach the holdout:
it is scored once at the end of the project and never optimized against, by human or AI.

**Data & caching.** Prices are adjusted closes cached to a single Parquet file
(`{_CACHE_FILE}`), primed once outside the UI. **The UI expects a primed cache and never
downloads data itself** — a visitor's click can trigger CPU work only, never network I/O, and
a missing cache renders a plain-language warning instead of fetching.

**Metrics.** All metrics come from `metrics/performance.py` (annualization 252, risk-free 0) —
the single source of truth the R layer (week 6) must reproduce within tolerance.

*R-vs-Python comparison lands week 6; AI chat and Research mode land weeks 7–8.*
"""
    )


def main() -> None:
    st.set_page_config(page_title="QuantForge", layout="wide")
    st.title("QuantForge — AI-driven quant research pipeline")

    cfg = sidebar_config()

    tab_backtest, tab_portfolio, tab_chat, tab_research, tab_method = st.tabs(
        ["Backtest", "Portfolio", "AI Chat", "Research mode", "Methodology"]
    )
    with tab_backtest:
        _render_backtest_tab(cfg)
    with tab_portfolio:
        _render_portfolio_tab(cfg)
    with tab_chat:
        _render_placeholder_tab(
            "AI Chat",
            "week 7",
            "Natural-language backtests ('run momentum on tech, 2015–2019') parsed by the "
            "budget-capped NL interface into vetted strategies + whitelisted params only.",
        )
    with tab_research:
        _render_placeholder_tab(
            "Research mode",
            "week 8",
            "The AI research agent iterates on train/validation under guardrails; the holdout "
            "is scored once, at the end, and never optimized against.",
        )
    with tab_method:
        _render_methodology_tab()


if __name__ == "__main__":
    main()
