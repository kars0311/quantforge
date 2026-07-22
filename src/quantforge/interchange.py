"""Arrow/Parquet interchange contract — the polyglot backbone.

Every cross-stage / cross-language hand-off goes through Parquet files whose schema is defined here,
once. Python writes; R (`arrow`), KNIME, and (later) MATLAB read the same files natively. The file is
the contract — no brittle in-process language bridge.

Conventions (keep these stable; downstream tools depend on them):
- Time index column: ``date`` (``timestamp[ns]``, tz-aware UTC, daily).
- Prices: long format, columns ``date, ticker, close`` (adjusted close).
- Positions: ``date, ticker, weight`` — target weights decided at the close of ``date``;
  gross exposure ``sum(|w|) <= 1`` per date (long-short allowed).
- Returns: ``date, ret`` (portfolio) or ``date, ticker, ret`` (per-asset).
- Metrics: ``name, value`` (one row per metric).
- Weights: ``ticker, weight`` (optimizer output, sum(w) = 1).
- Frontier: ``risk, ret`` (efficient-frontier points; annualized vol / annualized return).

Validation is strict at the boundary on purpose: a malformed file must fail loudly here, not
surface as silent NaNs three pipeline stages later.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Shared field types, named once so every schema below stays consistent. `date` is tz-aware UTC
# because tz-naive timestamps are ambiguous across languages (R's arrow reader would guess), and
# nanosecond precision matches pandas' native resolution so round-trips are lossless.
_DATE = pa.timestamp("ns", tz="UTC")
_VALUE = pa.float64()
_LABEL = pa.string()

#: kind -> pyarrow schema. This dict *is* the cross-language contract: every artifact that crosses
#: a stage or language boundary must match one of these schemas exactly (columns, order, dtypes).
SCHEMAS: dict[str, pa.Schema] = {
    "prices": pa.schema([("date", _DATE), ("ticker", _LABEL), ("close", _VALUE)]),
    "positions": pa.schema([("date", _DATE), ("ticker", _LABEL), ("weight", _VALUE)]),
    "returns": pa.schema([("date", _DATE), ("ret", _VALUE)]),
    "asset_returns": pa.schema([("date", _DATE), ("ticker", _LABEL), ("ret", _VALUE)]),
    "metrics": pa.schema([("name", _LABEL), ("value", _VALUE)]),
    "weights": pa.schema([("ticker", _LABEL), ("weight", _VALUE)]),
    "frontier": pa.schema([("risk", _VALUE), ("ret", _VALUE)]),
}


class SchemaError(ValueError):
    """A DataFrame does not conform to the interchange contract.

    Subclasses ``ValueError`` (not a bare ``KeyError``/``TypeError``) so callers can catch one
    well-named exception at the boundary, and so the message can say *which* column or dtype is
    wrong and what was expected — a bad hand-off should be diagnosable from the traceback alone.
    """


def validate_frame(df: pd.DataFrame, kind: str) -> None:
    """Raise :class:`SchemaError` unless ``df`` matches ``SCHEMAS[kind]``; pass silently if it does.

    Checks, in order: known ``kind``; exact column names *and order* (order matters because the
    Parquet file's physical layout is part of the cross-language contract); dtypes per column
    (``float64`` for value columns, string-like for labels); and that every ``date`` column is
    tz-aware UTC — a tz-naive or non-UTC date column would be reinterpreted differently by each
    consumer language, which is exactly the ambiguity this contract exists to eliminate.
    """
    if kind not in SCHEMAS:
        known = ", ".join(sorted(SCHEMAS))
        raise SchemaError(f"unknown interchange kind {kind!r}; known kinds: {known}")

    schema = SCHEMAS[kind]
    expected = schema.names
    actual = list(df.columns)
    if actual != expected:
        missing = [c for c in expected if c not in actual]
        extra = [c for c in actual if c not in expected]
        problems = []
        if missing:
            problems.append(f"missing column(s) {missing}")
        if extra:
            problems.append(f"unexpected column(s) {extra}")
        if not problems:  # same set, wrong order
            problems.append("columns are out of order")
        raise SchemaError(
            f"{kind!r} frame {'; '.join(problems)}: expected columns {expected}, got {actual}"
        )

    for field in schema:
        _check_column_dtype(df[field.name], field, kind)


def _check_column_dtype(col: pd.Series, field: pa.Field, kind: str) -> None:
    """Check one column's pandas dtype against its pyarrow field; raise SchemaError if wrong."""
    name, dtype = field.name, col.dtype

    if isinstance(field.type, pa.TimestampType):
        # Require tz-aware UTC explicitly rather than coercing: silently localizing a naive
        # timestamp would guess a timezone and could shift every date in the file.
        if not isinstance(dtype, pd.DatetimeTZDtype):
            raise SchemaError(
                f"{kind!r} column {name!r} must be timestamp[ns] tz-aware UTC, "
                f"got tz-naive dtype {dtype} (localize with .dt.tz_localize('UTC'))"
            )
        if str(dtype.tz) != "UTC":
            raise SchemaError(
                f"{kind!r} column {name!r} must be UTC, got timezone {dtype.tz!r} "
                f"(convert with .dt.tz_convert('UTC'))"
            )
        # Note on resolution: pandas 3.x defaults new datetimes to 'us', so we accept any
        # tz-aware UTC unit here — daily dates are exactly representable in all of them, and
        # rejecting the pandas default would make every naturally-built frame fail. The *file*
        # is still always timestamp[ns] (write_frame casts via SCHEMAS[kind]) and read_frame
        # normalizes back to ns, so the on-disk contract stays pinned.
    elif field.type == pa.float64():
        if not pd.api.types.is_float_dtype(dtype) or dtype != "float64":
            raise SchemaError(
                f"{kind!r} column {name!r} must be float64, got {dtype} "
                f"(cast with .astype('float64'))"
            )
    elif field.type == pa.string():
        # Accept object-of-str or pandas StringDtype: both serialize to Arrow `string`, and
        # plain-Python strings default to object dtype in pandas, so rejecting object would
        # fail every naturally-constructed frame for no interchange benefit.
        if not (pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)):
            raise SchemaError(f"{kind!r} column {name!r} must be string, got {dtype}")
    else:  # pragma: no cover - defensive: only the three types above appear in SCHEMAS
        raise SchemaError(f"{kind!r} column {name!r}: unhandled schema type {field.type}")


#: The kinds that have a per-ticker panel shape (one row per date x ticker) and therefore a
#: meaningful wide form, mapped to the column that becomes the cell values. The other kinds
#: (`returns`, `metrics`, `weights`, `frontier`) are already 2-column tables with no ticker/date
#: cross-section, so "wide" is undefined for them and asking for it is a caller bug.
_WIDE_KINDS: dict[str, str] = {
    "prices": "close",
    "positions": "weight",
    "asset_returns": "ret",
}


def _require_wide_kind(kind: str) -> str:
    """Return the value column for a per-ticker ``kind``; raise SchemaError for any other kind."""
    if kind not in _WIDE_KINDS:
        known = ", ".join(sorted(_WIDE_KINDS))
        raise SchemaError(
            f"kind {kind!r} has no wide form (only per-ticker kinds do: {known}); "
            f"to_wide/to_long apply only to date x ticker panels"
        )
    return _WIDE_KINDS[kind]


def to_wide(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Pivot a validated long ``kind`` frame to wide: index = date, columns = ticker.

    This is the shape ``Strategy.generate_signals`` and ``Engine.run_backtest`` consume — the
    vectorized engine wants a date x ticker matrix so that operations like ``positions.shift(1)``
    apply to every asset at once. Only the per-ticker kinds (``prices``, ``positions``,
    ``asset_returns``) have a wide form; any other kind raises :class:`SchemaError`.

    Missing date x ticker pairs (e.g. a ticker that starts trading later) become ``NaN`` cells,
    which is the wide-format representation of "no data" — engines must handle it explicitly
    rather than us fabricating values here. Duplicate (date, ticker) pairs are rejected loudly:
    a pivot cannot represent them and silently keeping one row would hide a data bug upstream.
    """
    value_col = _require_wide_kind(kind)
    validate_frame(df, kind)

    dup = df.duplicated(subset=["date", "ticker"])
    if dup.any():
        first = df.loc[dup, ["date", "ticker"]].iloc[0]
        raise SchemaError(
            f"{kind!r} frame has duplicate (date, ticker) rows, e.g. "
            f"({first['date']}, {first['ticker']!r}); each pair must appear at most once "
            f"to pivot to wide form — deduplicate the source data first"
        )

    # pivot names the index 'date' and the columns axis 'ticker' from the source column names,
    # which is exactly the contract for the wide form (and what to_long relies on to invert).
    return df.pivot(index="date", columns="ticker", values=value_col)


def to_long(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Melt a wide date x ticker frame back to the long interchange form for ``kind``.

    Inverse of :func:`to_wide`; use it before every ``write_frame`` of wide data so only
    contract-shaped long frames ever reach a Parquet file. ``NaN`` cells are dropped — in the
    long form "missing" is represented by the *absence* of a row, not a NaN row, so a ragged
    panel round-trips exactly: ``to_long(to_wide(df), kind)`` equals ``df`` (sorted by
    date, ticker). Output is sorted (date-major, then ticker) to be canonical regardless of
    the wide frame's column order, and is re-validated so a malformed wide frame (naive index,
    non-numeric cells) fails here, loudly, not at the next stage.
    """
    value_col = _require_wide_kind(kind)

    # Check the index up front so the error names the actual problem ("your index is tz-naive")
    # instead of a downstream validate_frame complaint about a column the caller never built.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise SchemaError(
            f"wide {kind!r} frame must have a DatetimeIndex of dates, got {type(df.index).__name__}"
        )
    if df.index.tz is None:
        raise SchemaError(
            f"wide {kind!r} frame index must be tz-aware UTC, got tz-naive "
            f"(localize with .tz_localize('UTC'))"
        )
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            raise SchemaError(
                f"wide {kind!r} frame column {col!r} must be numeric ({value_col} values), "
                f"got dtype {df[col].dtype}"
            )

    # rename_axis: a caller-built wide frame may have unnamed axes; the names become the long
    # frame's column names, so pin them. stack() in pandas 3 keeps NaN cells, hence the explicit
    # dropna — absent rows, not NaN rows, are the long-form representation of missing data.
    stacked = df.rename_axis(index="date", columns="ticker").stack().dropna()
    long = stacked.rename(value_col).reset_index()
    long = long[["date", "ticker", value_col]]
    long["date"] = long["date"].dt.tz_convert("UTC")
    long[value_col] = long[value_col].astype("float64")
    long = long.sort_values(["date", "ticker"]).reset_index(drop=True)

    validate_frame(long, kind)
    return long


def write_frame(df: pd.DataFrame, path: str, kind: str) -> None:
    """Write a dataframe to Parquet after validating it against the ``kind`` schema.

    Validation runs *before* anything touches the filesystem (no directory creation, no file
    handle) so a rejected frame can never leave a partial or invalid artifact behind — downstream
    languages must be able to trust that any file at an interchange path conforms to the contract.
    The Arrow table is built with ``SCHEMAS[kind]`` explicitly (not inferred) so the physical
    Parquet types are identical no matter which pandas dtypes produced them; the index is dropped
    because row position carries no meaning in the contract — the columns are the data.
    """
    validate_frame(df, kind)
    table = pa.Table.from_pandas(df, schema=SCHEMAS[kind], preserve_index=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_frame(path: str, kind: str) -> pd.DataFrame:
    """Read a Parquet file, validate it against the ``kind`` schema, and return a long DataFrame.

    Reading is the other trust boundary: files may come from R, KNIME, or a hand-run export, so we
    re-validate here rather than assume our own writer produced them — a wrong-schema file raises
    :class:`SchemaError` immediately instead of surfacing as NaNs three stages later. The one
    normalization we do first is the ``date`` column: a tz-naive file is unambiguous by our own
    convention (all interchange dates *are* UTC), so localizing to UTC recovers the intended value
    rather than guessing; an aware non-UTC column is converted, which is lossless. Everything else
    (columns, dtypes) must already be right or validation fails loudly. Pandas index metadata in
    the file is ignored so the result always has a default RangeIndex.
    """
    table = pq.read_table(path)
    # ignore_metadata: a foreign writer (or an older pandas) may have stored an index or exotic
    # extension dtypes in the pandas metadata blob; the contract is columns-only, so we drop it
    # and always hand back a plain RangeIndex frame.
    df = table.to_pandas(ignore_metadata=True)

    if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
        date = df["date"]
        date = date.dt.tz_localize("UTC") if date.dt.tz is None else date.dt.tz_convert("UTC")
        # Force nanosecond precision: Parquet files written elsewhere may use us/ms timestamps,
        # and the contract (and pandas round-trip equality) is pinned to timestamp[ns].
        df["date"] = date.astype("datetime64[ns, UTC]")

    validate_frame(df, kind)
    return df
