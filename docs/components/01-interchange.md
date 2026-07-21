# Component 01 — `src/quantforge/interchange.py` (Arrow/Parquet interchange contract)

**Week:** 1 · **Status:** stub · **Depends on:** nothing (foundation)

## Function

Defines the cross-language data contract once: table schemas (columns, dtypes, timezone
conventions) for every artifact that crosses a stage or language boundary, plus read/write/validate
helpers. Python writes these files; R (`arrow`), KNIME, and a future MATLAB engine read them
natively. The file *is* the contract — no in-process language bridges.

## Requirements satisfied

- **AR-2** — every cross-stage/cross-language hand-off goes through these schemas.
- **AR-4** — a new engine only needs to read/write these files to slot in.
- Schema changes after week 6 are breaking changes (treat like a frozen API).

## Schemas (the contract)

All `date` columns: `timestamp[ns]`, UTC, daily. All value columns: `float64`. `ticker`/`name`:
`string`. Long format throughout (one row per date×ticker) because R/KNIME/Arrow handle long
data most naturally; Python converts to wide internally for the vectorized engine.

| kind | columns | semantics |
|------|---------|-----------|
| `prices` | `date, ticker, close` | adjusted close |
| `positions` | `date, ticker, weight` | target weights decided at close of `date`; gross exposure Σ\|w\| ≤ 1 per date (long-short allowed per §5 decision) |
| `returns` | `date, ret` | portfolio daily returns (net of costs) |
| `asset_returns` | `date, ticker, ret` | per-asset daily returns |
| `metrics` | `name, value` | one row per metric; names = `metrics/performance.py` `_KEYS` |
| `weights` | `ticker, weight` | optimizer output, Σw = 1 |
| `frontier` | `risk, ret` | efficient-frontier points (annualized vol, annualized return) |

## Interface (integral functions)

```python
SCHEMAS: dict[str, pa.Schema]        # kind -> pyarrow schema (the source of truth above)

class SchemaError(ValueError): ...   # raised with an actionable message (missing col, wrong dtype)

def validate_frame(df: pd.DataFrame, kind: str) -> None
    # Raises SchemaError unless df matches SCHEMAS[kind] (columns, dtypes, tz-aware UTC dates).

def write_frame(df: pd.DataFrame, path: str, kind: str) -> None
    # validate_frame() then write Parquet via pyarrow. Never writes an invalid file.

def read_frame(path: str, kind: str) -> pd.DataFrame
    # Read Parquet, validate, normalize `date` to UTC. Returns a long-format DataFrame.

def to_wide(df: pd.DataFrame, kind: str) -> pd.DataFrame
    # Long -> wide (index=date, columns=ticker) for `prices`/`positions`/`asset_returns`.
    # This is the form Strategy.generate_signals and Engine.run_backtest consume.

def to_long(df: pd.DataFrame, kind: str) -> pd.DataFrame
    # Wide -> long, inverse of to_wide (used before every write_frame of wide data).
```

## Design notes

- Validation is strict on write *and* read: a bad file must fail loudly at the boundary, not
  produce silent NaNs three stages later.
- The `positions` gross-exposure convention (Σ|w| ≤ 1) supersedes the earlier "sum ≤ 1" draft in
  the module docstring — updated for the long-short mean-reversion decision (product.md §5).

## Done when

- Round-trip test green: `write_frame` → `read_frame` → identical frame, for every kind.
- `Rscript -e 'arrow::read_parquet(...)'` reads a written prices file with correct types.
