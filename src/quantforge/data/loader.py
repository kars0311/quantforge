"""Data ingestion: yfinance -> Parquet cache.

Bias notes (document honestly in the README — this is a maturity signal):
- yfinance is unofficial and rate-limited; cache aggressively to Parquet.
- SURVIVORSHIP BIAS: using today's tickers ignores delisted names. Use a FIXED historical universe
  and state the caveat. (A point-in-time constituent source would be the proper fix.)
- LOOK-AHEAD: adjusted close already embeds future splits/dividends in a benign way, but never let
  *signals* peek at same-day data they couldn't have known at decision time.

TODO(week1): implement load_prices() with a parquet cache via interchange.write_frame/read_frame.
"""

from __future__ import annotations


def load_prices(tickers: list[str], start: str, end: str, cache_dir: str = "data_cache"):
    """Return adjusted-close prices (long format: date, ticker, close), cached to Parquet.

    TODO: fetch via yfinance; normalize to the interchange "prices" schema; cache + reuse.
    """
    raise NotImplementedError
