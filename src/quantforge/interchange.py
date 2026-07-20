"""Arrow/Parquet interchange contract — the polyglot backbone.

Every cross-stage / cross-language hand-off goes through Parquet files whose schema is defined here,
once. Python writes; R (`arrow`), KNIME, and (later) MATLAB read the same files natively. The file is
the contract — no brittle in-process language bridge.

Conventions (keep these stable; downstream tools depend on them):
- Time index column: ``date`` (UTC, daily).
- Prices: long format, columns ``date, ticker, close`` (adjusted close).
- Positions: ``date, ticker, weight`` (target weights, sum<=1 per date).
- Returns: ``date, ret`` (portfolio) or ``date, ticker, ret`` (per-asset).
- Metrics: ``name, value`` (one row per metric).

TODO(week1): implement read/write helpers + a schema-validation function and unit-test round-trips.
"""

from __future__ import annotations

# import pyarrow as pa
# import pyarrow.parquet as pq
# import pandas as pd


def write_frame(df, path: str, kind: str) -> None:
    """Write a dataframe to Parquet after validating it against the ``kind`` schema.

    kind in {"prices", "positions", "returns", "metrics"}.
    TODO: validate columns/dtypes against the conventions above; write with pyarrow.
    """
    raise NotImplementedError


def read_frame(path: str, kind: str):
    """Read a Parquet file and validate it against the ``kind`` schema. Returns a DataFrame.

    TODO: read with pyarrow; assert schema; normalize the ``date`` index to UTC.
    """
    raise NotImplementedError
