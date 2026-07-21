# Component 19 — `docs/` (methodology writeup + demo assets)

**Week:** 10 · **Status:** missing · **Depends on:** everything (written last)

## Function

The narrative deliverables: an honest methodology writeup, the 60–90s demo video/GIF, and the
architecture diagram. This is where every documented caveat gets its permanent home.

## Requirements satisfied

**DL-5** · **DL-6** · **RG-3** (survivorship caveat's canonical statement).

## Deliverables

1. **`docs/METHODOLOGY.md`** — outline:
   - Data & universe (the §5 decisions; survivorship caveat; ADR note; fixed cutoff).
   - Backtest assumptions (t→t+1 execution, 10bps turnover costs, **no borrow fees on shorts**).
   - Engine validation (vs backtesting.py, tolerance, what matched).
   - Out-of-sample protocol (split dates, one-shot holdout, iteration cap) + the val-vs-holdout
     results table — reported even if unflattering (especially then).
   - Cross-language consistency (R deltas and why: ddof, annualization).
   - AI guardrails + safety architecture (the SF table, with links to the proving tests).
   - Limitations & next steps (point-in-time universe, borrow modeling, MATLAB engine seam).
2. **Demo video/GIF** — shot list: NL query → charts appear → Research mode iterating →
   holdout reveal → R tearsheet → frontier. 60–90s, hosted URL visible.
3. **Architecture diagram** — the pipeline + seam + guardrails picture (mermaid in the README,
   exported PNG for the writeup).

## Done when

A reader who knows quant can find every assumption, every caveat, and the test that proves each
claim, without reading source code; repo tagged `v1.0`.
