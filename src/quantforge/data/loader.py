"""Data ingestion: yfinance -> Parquet cache, plus the frozen universe and date splits.

Bias notes (document honestly in the README — this is a maturity signal):
- yfinance is unofficial and rate-limited; cache aggressively to Parquet.
- SURVIVORSHIP BIAS: using today's tickers ignores delisted names. Use a FIXED historical universe
  and state the caveat. (A point-in-time constituent source would be the proper fix.)
- LOOK-AHEAD: adjusted close already embeds future splits/dividends in a benign way, but never let
  *signals* peek at same-day data they couldn't have known at decision time.
- Names that IPO'd after 2010 (e.g. META, 2012) enter the panel when their data begins; earlier
  dates are simply absent and the strategies/engine tolerate that raggedness.
- Direct foreign listings (e.g. ``2330.TW``) are deliberately excluded in favor of US-listed ADRs
  (TSM, ASML, ...), which keep the whole panel on the US trading calendar in USD.

Cache design: ONE file (``{cache_dir}/prices.parquet``) holding the whole universe for START..END;
every request — any ticker subset, any sub-range — is sliced from it. After the first download the
pipeline is fully offline, and refreshing requires *deleting the file deliberately*: there is no
silent refetch, because a refetch would change the data under every previously computed result.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantforge import interchange

# The frozen universe (docs/components/02-data-loader.md). Fixed and hand-picked so every backtest
# is reproducible; the survivorship caveat above applies because this is *today's* list of names,
# not a point-in-time constituent history.
UNIVERSE: list[str] = [
    # Tech
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "CRM",
    "ADBE",
    "ORCL",
    "AVGO",
    # ADRs (international, US-listed, USD)
    "TSM",
    "ASML",
    "SAP",
    "TM",
    "NVO",
    "SONY",
    # Financials
    "JPM",
    "GS",
    "V",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    # Consumer
    "PG",
    "KO",
    "MCD",
    "WMT",
    "HD",
    # Energy / Industrial
    "XOM",
    "CVX",
    "CAT",
]

# Fixed date range. END is a hard cutoff, never "today": re-running the pipeline next month must
# produce byte-identical results, and a moving end date would silently change every metric.
START: str = "2010-01-01"
END: str = "2026-06-30"

# Chronological ~60/20/20 split (RG-4). Defined here and ONLY here — guardrails, tests, and the UI
# import these bounds rather than redefining them, so there is exactly one place a date typo could
# live. The holdout is scored once and never optimized against (see AGENTS.md: an LLM iterating on
# the test set is p-hacking).
SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2010-01-01", "2019-12-31"),
    "validation": ("2020-01-01", "2022-12-31"),
    "holdout": ("2023-01-01", "2026-06-30"),
}


def get_split_bounds() -> dict[str, tuple[str, str]]:
    """Return the train/validation/holdout date bounds, i.e. ``SPLITS``.

    Exists so callers (guardrails, tests, UI) have a function to call instead of hard-coding
    dates: if the splits ever changed, every consumer would pick up the change from this single
    source of truth (RG-4).
    """
    return SPLITS


def _download_universe() -> pd.DataFrame:
    """Download the FULL universe for the FULL START..END range and normalize to long prices.

    Always the whole panel, never the requested subset: the cache holds one canonical file that
    every future request slices from, so a partial download would poison the cache for every
    other caller. This is the only function in the module that touches the network.
    """
    # Lazy import (why): the cache-hit path must run fully offline — including on machines where
    # yfinance isn't even installed (CI, the public demo). Importing at module top would make
    # *every* loader use depend on an unofficial, rate-limited package.
    import yfinance as yf

    raw = yf.download(
        UNIVERSE,
        start=START,
        # yfinance's `end` is exclusive; add a day so the END date itself is included, keeping
        # the doc's promise that the range runs *through* 2026-06-30.
        end=pd.Timestamp(END) + pd.Timedelta(days=1),
        auto_adjust=True,  # "Close" then IS the adjusted close — the only column we keep
        progress=False,
    )

    # Multi-ticker downloads come back with MultiIndex columns: level 0 = price field
    # ("Close", "High", ...), level 1 = ticker (yfinance's default group_by="column" layout).
    # Selecting "Close" flattens that to a wide date x ticker frame of adjusted closes.
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError(
            "unexpected yfinance layout: expected MultiIndex columns for a multi-ticker "
            f"download, got columns {list(raw.columns)!r}"
        )
    close = raw["Close"].copy()

    # yfinance's daily index is tz-naive; the interchange contract requires tz-aware UTC.
    # Localizing (not converting) is correct for naive input — these are calendar dates.
    idx = pd.DatetimeIndex(close.index)
    close.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    close = close.astype("float64")

    # to_long drops NaN cells (holidays, pre-IPO dates): in the long form "missing" is an
    # absent row, not a NaN row, so the panel stays honestly ragged. Output is sorted by
    # (date, ticker) and re-validated against the "prices" schema.
    return interchange.to_long(close, "prices")


def load_prices(
    tickers: list[str] | None = None,
    start: str = START,
    end: str = END,
    cache_dir: str = "data_cache",
) -> pd.DataFrame:
    """Return adjusted-close prices (long format: date, ticker, close), cached to Parquet.

    Cache hit: read the one canonical file via ``interchange.read_frame`` — zero network, and
    yfinance is never imported (see :func:`_download_universe`). Cache miss: download the whole
    universe once, normalize, ``write_frame`` it, and proceed exactly as a hit — the returned
    frame is always sliced from the *file's* contents, so hit and miss are byte-identical and
    every call is deterministic given the cache. Refreshing data requires deleting the file.
    """
    cache_path = Path(cache_dir) / "prices.parquet"
    if not cache_path.exists():
        interchange.write_frame(_download_universe(), str(cache_path), "prices")
    full = interchange.read_frame(str(cache_path), "prices")

    if tickers is None:
        tickers = UNIVERSE
    # Inclusive [start, end] slice. The bounds are parsed as midnight UTC, matching the daily
    # dates in the file, so `between` keeps both endpoint days.
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    out = full[full["ticker"].isin(tickers) & full["date"].between(lo, hi)]
    out = out.reset_index(drop=True)  # canonical RangeIndex regardless of slice position

    # Defensive re-validate: slicing preserves the schema today, but the contract promise is
    # "every frame this returns passes validate_frame", so assert it at the boundary.
    interchange.validate_frame(out, "prices")
    return out
