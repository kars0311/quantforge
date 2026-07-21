# Component 05 — `src/quantforge/strategies/` (momentum, mean-reversion)

**Weeks:** 2 (momentum ✓), 4 (mean-reversion) · **Status:** momentum working; mean-reversion stub
**Depends on:** engine/base

## Function

Each strategy maps wide prices + whitelisted params to wide target weights, using only
information available at the close of each row's date. Strategies are also the "vetted set"
that PUBLIC_MODE restricts the AI to — so this package owns the param whitelist.

## Requirements satisfied

- **FR-3** — the two core strategies. **RG-1** — strictly backward-looking windows (the engine's
  shift is the backstop, not an excuse). **AR-1** — implement `Strategy`. **SF-3** — the registry
  + whitelist below is what guardrails/MCP validate against.

## Interface

```python
STRATEGIES: dict[str, type[Strategy]]   # {"momentum": ..., "mean_reversion": ...} — the vetted set

PARAM_WHITELIST: dict[str, dict]        # per strategy: param -> {type, min/max or choices, default}
# momentum:        lookback int [20, 252] def 126 · top_n int [0, 10] def 0
# mean_reversion:  lookback int [5, 60] def 20 · entry_z float [0.5, 3.0] def 2.0
#                  exit_z float [0.0, 1.5] def 0.5 · mode in {"long_flat","long_short"} def long_flat

def validate_params(strategy: str, params: dict) -> dict
    # Returns params merged with defaults; raises ValueError naming the offending param if the
    # strategy is unvetted or any param is outside the whitelist. Single choke-point used by
    # guardrails.assert_no_codegen, the MCP server, and the UI.
```

### `MomentumStrategy` (implemented — reference)

Signal: trailing `lookback`-day return `prices / prices.shift(lookback) − 1`. If `top_n > 0`:
equal-weight the top-n ranked names; else weight ∝ positive momentum, normalized to Σw = 1.
Warmup rows → 0. Long-only by construction.

### `MeanReversionStrategy` (week 4 — spec)

Fade short-term extremes via a rolling z-score (all windows `min_periods=lookback`, backward-looking):

```
z_t = (P_t − SMA_lookback(P)_t) / SD_lookback(P)_t
```

- **long_flat (default):** enter long a name when `z < −entry_z`; stay long until `z > −exit_z`
  (hysteresis via vectorized state: entry/exit masks + forward-fill). Equal weight across active
  longs, Σw ≤ 1; no positions during warmup.
- **long_short:** additionally short when `z > entry_z`, cover when `z < exit_z`. Equal weight
  per active name with gross exposure normalized so Σ|w| ≤ 1.
- **Documented caveat (product.md §5, honesty requirement):** short-side frictions beyond
  symmetric transaction costs — borrow fees, locate constraints, squeeze risk — are not modeled.
  Stated in the module docstring and the methodology writeup, never hidden.

## Done when

- Mean-reversion passes: shape == prices.shape, gross ≤ 1, warmup rows flat, a hand-built
  oscillating price series produces the expected entries/exits in both modes; `validate_params`
  covered by tests; interfaces frozen (week 4).
