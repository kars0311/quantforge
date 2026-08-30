# Component 02 — `src/quantforge/data/loader.py` (data ingestion)

**Week:** 1 · **Status:** stub · **Depends on:** interchange

## Function

Downloads daily adjusted-close prices from yfinance for the fixed universe, caches them to Parquet
via the interchange contract, and owns the train/validation/holdout date boundaries (defined here,
in exactly one place). After the first download, everything runs offline from cache — the demo
must never depend on yfinance being up.

## Requirements satisfied

- **FR-1** — download + cache daily prices for a fixed universe.
- **RG-3** — survivorship caveat documented here (module docstring) and in the writeup.
- **RG-4** — the split boundaries live here; guardrails and tests import them, never redefine them.
- **AR-2** — cache files conform to the `prices` schema.

## Fixed universe & dates (product.md §5 decisions)

- **Range:** 2010-01-01 → 2026-06-30 (fixed cutoff — reproducible, never "up to today").
- **Splits (chronological 60/20/20):** train 2010-01-01→2019-12-31 · validation
  2020-01-01→2022-12-31 · holdout 2023-01-01→2026-06-30.
- **Universe (30 names, tech-heavy, incl. ADRs):** list below; verified for data availability and
  **frozen** in week 1.
  - Tech: AAPL MSFT NVDA GOOGL AMZN META CRM ADBE ORCL AVGO
  - ADRs (international, US-listed, USD): TSM ASML SAP TM NVO SONY
  - Financials: JPM GS V · Healthcare: JNJ UNH PFE · Consumer: PG KO MCD WMT HD
  - Energy/Industrial: XOM CVX CAT
- Names that IPO'd after 2010 (META 2012, etc.) enter the panel when their data begins; earlier
  dates are NaN and strategies/engine already tolerate that. Direct foreign listings (e.g.
  `2330.TW`) are excluded — ADRs keep everything on the US calendar in USD.

## Interface (integral functions)

```python
UNIVERSE: list[str]                  # the frozen ticker list
START: str; END: str                 # "2010-01-01", "2026-06-30"
SPLITS: dict[str, tuple[str, str]]   # {"train": (start, end), "validation": ..., "holdout": ...}

def load_prices(tickers: list[str] | None = None,   # default: UNIVERSE
                start: str = START, end: str = END,
                cache_dir: str = "data_cache") -> pd.DataFrame
    # Returns long-format prices (date, ticker, close) per the interchange contract.
    # Cache hit -> read_frame(cache) only, no network. Cache miss -> yfinance download
    # (auto_adjust=True), normalize, write_frame(), return. Deterministic given the cache.

def get_split_bounds() -> dict[str, tuple[str, str]]
    # Returns SPLITS. Exists so callers (guardrails, tests, UI) never hard-code dates.
```

## Design notes

- One cache file (`data_cache/prices.parquet`) for the whole universe; partial-universe requests
  are sliced from it. Refreshing requires deleting the file deliberately — no silent refetch.
- yfinance quirks to normalize: multi-index columns on multi-ticker downloads, tz-naive index
  (localize to UTC), missing rows on holidays (leave missing — the panel is business-day ragged).

## Done when

- `load_prices()` works online once, then fully offline; output passes `validate_frame`.
- A committed test uses a small cached fixture (no network in CI).
