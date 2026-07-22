"""Verification tests for the interchange contract: SCHEMAS + SchemaError + validate_frame.

Locks the cross-language contract (7 kinds, exact columns/dtypes, tz-aware UTC dates) and proves
validate_frame fails loudly — with SchemaError, never a bare KeyError/TypeError — on every class of
malformed frame. Offline, synthetic data only.
"""

import pandas as pd
import pyarrow as pa
import pytest

import quantforge.interchange as interchange
from quantforge.interchange import SCHEMAS, SchemaError, validate_frame

_DATE = pa.timestamp("ns", tz="UTC")
_F64 = pa.float64()
_STR = pa.string()

#: The frozen contract, restated independently of the implementation.
_EXPECTED = {
    "prices": [("date", _DATE), ("ticker", _STR), ("close", _F64)],
    "positions": [("date", _DATE), ("ticker", _STR), ("weight", _F64)],
    "returns": [("date", _DATE), ("ret", _F64)],
    "asset_returns": [("date", _DATE), ("ticker", _STR), ("ret", _F64)],
    "metrics": [("name", _STR), ("value", _F64)],
    "weights": [("ticker", _STR), ("weight", _F64)],
    "frontier": [("risk", _F64), ("ret", _F64)],
}


def _utc_dates(n: int = 3) -> pd.Series:
    return pd.Series(pd.date_range("2024-01-02", periods=n, tz="UTC"))


def _good_frame(kind: str) -> pd.DataFrame:
    """Hand-build a conforming frame for each kind (no helpers from the module under test)."""
    cols = {}
    for name, typ in _EXPECTED[kind]:
        if typ == _DATE:
            cols[name] = _utc_dates()
        elif typ == _F64:
            cols[name] = [0.1, 0.2, 0.3]
        else:
            cols[name] = ["AAPL", "MSFT", "GOOG"]
    return pd.DataFrame(cols)


# ---------------------------------------------------------------- SCHEMAS contract


def test_schemas_has_exactly_the_seven_kinds():
    assert set(SCHEMAS) == set(_EXPECTED)


@pytest.mark.parametrize("kind", sorted(_EXPECTED))
def test_schema_columns_and_dtypes_exact(kind):
    schema = SCHEMAS[kind]
    assert isinstance(schema, pa.Schema)
    assert [(f.name, f.type) for f in schema] == _EXPECTED[kind]


def test_schema_error_is_a_value_error():
    assert issubclass(SchemaError, ValueError)


def test_docstring_states_gross_exposure_convention():
    doc = interchange.__doc__
    assert "sum(|w|) <= 1" in doc
    assert "long-short" in doc


# ---------------------------------------------------------------- conforming frames pass


@pytest.mark.parametrize("kind", sorted(_EXPECTED))
def test_validate_frame_accepts_conforming_frame(kind):
    validate_frame(_good_frame(kind), kind)  # must pass silently


def test_validate_frame_accepts_empty_typed_frame():
    df = pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns, UTC]"),
            "ret": pd.Series([], dtype="float64"),
        }
    )
    validate_frame(df, "returns")


# ---------------------------------------------------------------- rejections (adversarial)


def test_unknown_kind_raises_schema_error_naming_the_kind():
    with pytest.raises(SchemaError, match="quotes"):
        validate_frame(_good_frame("prices"), "quotes")


def test_missing_column_raises_schema_error_not_key_error():
    df = _good_frame("prices").drop(columns=["close"])
    try:
        validate_frame(df, "prices")
    except SchemaError as exc:
        assert "close" in str(exc)
    else:
        pytest.fail("missing column accepted")


def test_extra_column_rejected():
    df = _good_frame("prices").assign(volume=[1.0, 2.0, 3.0])
    with pytest.raises(SchemaError, match="volume"):
        validate_frame(df, "prices")


def test_misordered_columns_rejected():
    df = _good_frame("prices")[["ticker", "date", "close"]]
    with pytest.raises(SchemaError, match="order"):
        validate_frame(df, "prices")


@pytest.mark.parametrize("bad_dtype", ["float32", "int64"])
def test_wrong_value_dtype_rejected_with_column_named(bad_dtype):
    df = _good_frame("prices")
    df["close"] = df["close"].astype(bad_dtype)
    with pytest.raises(SchemaError, match="close"):
        validate_frame(df, "prices")


def test_nullable_float64_rejected():
    # Pandas' masked Float64 has null semantics the contract does not define — reject it.
    df = _good_frame("frontier")
    df["risk"] = df["risk"].astype("Float64")
    with pytest.raises(SchemaError, match="risk"):
        validate_frame(df, "frontier")


def test_tz_naive_date_rejected_not_type_error():
    df = _good_frame("returns")
    df["date"] = df["date"].dt.tz_localize(None)
    try:
        validate_frame(df, "returns")
    except SchemaError as exc:
        assert "date" in str(exc)
    else:
        pytest.fail("tz-naive date accepted")


def test_non_utc_timezone_rejected():
    df = _good_frame("returns")
    df["date"] = df["date"].dt.tz_convert("US/Eastern")
    with pytest.raises(SchemaError, match="date"):
        validate_frame(df, "returns")


def test_non_string_label_rejected():
    df = _good_frame("weights")
    df["ticker"] = [1, 2, 3]  # int64, not string
    with pytest.raises(SchemaError, match="ticker"):
        validate_frame(df, "weights")


@pytest.mark.parametrize(
    "df, kind",
    [
        (pd.DataFrame(), "no_such_kind"),
        (pd.DataFrame({"wrong": [1]}), "returns"),
        (pd.DataFrame({"date": [1, 2], "ret": [0.1, 0.2]}), "returns"),
    ],
)
def test_never_raises_bare_builtin_exceptions(df, kind):
    # The boundary must fail with one well-named exception type, never leak KeyError/TypeError.
    with pytest.raises(SchemaError):
        validate_frame(df, kind)
