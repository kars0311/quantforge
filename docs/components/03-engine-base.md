# Component 03 — `src/quantforge/engine/base.py` (Strategy/Engine interfaces — the seam)

**Week:** 2–4 (**FROZEN at week 4**) · **Status:** working (vertical slice) · **Depends on:** nothing

## Function

Declares the two abstractions everything downstream depends on, plus the standardized result
container. This is the keystone: portfolio, metrics, UI, and the AI agent talk to these
interfaces, never to a concrete engine — which is what lets R/KNIME/MATLAB drop in later.

## Requirements satisfied

- **AR-1** — the single seam; frozen once mean-reversion lands (week 4). Any change after that is
  a breaking change requiring an explicit decision.
- **AR-4** — the documented "next engine drops in here" point.

## Interface (already implemented — documented here as the contract)

```python
@dataclass
class BacktestResult:
    equity_curve: pd.Series          # per-date portfolio value, starts at 1.0
    returns: pd.Series               # per-date net returns (after costs)
    metrics: dict[str, float]        # keys = metrics/performance.py _KEYS
    meta: dict[str, Any]             # engine name, params, cost_bps, data window

class Strategy(ABC):
    name: str
    def generate_signals(self, prices, params: dict[str, Any]) -> pd.DataFrame
        # prices: WIDE DataFrame (index=date, columns=ticker, adjusted close).
        # returns: WIDE DataFrame of TARGET WEIGHTS, same shape/index/columns as prices.
        # Semantics: the weight on row t is decided using information through the close of t.
        # Gross exposure Σ|w| ≤ 1 per row. Warmup rows (indicator undefined) -> 0.0.
        # MUST NOT use any same-row-or-later information beyond close-of-t (no look-ahead).

class Engine(ABC):
    name: str
    def run_backtest(self, prices, positions, params: dict[str, Any]) -> BacktestResult
        # Applies the one-day execution lag itself (positions.shift(1) or equivalent):
        # a weight decided at close of t earns t+1's return. Charges costs on turnover.
        # Must not mutate its inputs. Engines are interchangeable: same inputs -> same
        # BacktestResult fields within documented tolerance.
```

## Design notes

- Wide-form pandas in the interfaces, long-form Parquet at every persistence/language boundary
  (`interchange.to_wide`/`to_long` convert). Wide is what vectorized math wants; long is what
  Arrow/R want. The interfaces are still engine-agnostic: a non-Python engine's adapter converts
  at the boundary (see KNIME/stretch doc).
- The one-day-lag responsibility lives in the **engine**, not strategies — one enforcement point
  (RG-1). Strategies just express "what I want to hold as of tonight's close".

## Done when

- Interface untouched through week 4, then marked frozen in the module docstring; every concrete
  strategy/engine subclasses these ABCs and the test suite instantiates them only via the seam.
