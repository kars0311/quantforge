"""Streamlit UI — the snazzy, interactive demo.

Built incrementally from week 5 (charts) through week 9 (AI panel + engine selector + deploy).

Sections (TODO):
  - Controls: tickers, date range, strategy, params, transaction costs.
  - Charts: equity curve, drawdown, metrics table, efficient frontier (Plotly).
  - Engine selector: Python / R / KNIME (the polyglot story, visible in the UI).
  - AI chat panel: natural-language queries (ai.nl_interface).
  - "Research mode": run the AI research agent (ai.agent) with guardrails.
  - Prebuilt one-click demo scenarios (also the cached fallback when the AI budget is exhausted).

SAFETY (PUBLIC_MODE): live AI gated by DEMO_PASSCODE; ungated visitors see cached scenarios only;
show remaining AI budget; never expose the API key client-side.
"""

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="QuantForge", layout="wide")
    st.title("AI-driven quant research pipeline")
    st.info("Scaffold — UI not implemented yet. See docs/TEN_WEEK_PLAN.md.")
    # TODO: build out the sections listed in the module docstring.


if __name__ == "__main__":
    main()
