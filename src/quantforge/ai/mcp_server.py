"""MCP server — exposes the pipeline as tools the AI agent and NL interface call.

Tools (read-only, input-validated):
  - load_data(tickers, start, end)            -> dataset handle / summary
  - run_backtest(strategy, params, engine)    -> metrics + curve handle
  - optimize_portfolio(returns_handles)       -> weights + frontier
  - get_metrics(result_handle)                -> metrics dict

Safety: validate every argument (whitelist tickers/date-range/params); never accept or execute code;
key stays server-side. The agent being engine-agnostic falls out of this — it only ever sees tools.

TODO(week7): implement with the `mcp` Python SDK. Bound all inputs. Wire through guardrails + budget.
"""

from __future__ import annotations

# from mcp.server import Server  # TODO

TOOLS = ["load_data", "run_backtest", "optimize_portfolio", "get_metrics"]


def build_server():
    """Construct and return the MCP server exposing TOOLS. TODO."""
    raise NotImplementedError
