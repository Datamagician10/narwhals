from __future__ import annotations

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from narwhals.dependencies import is_narwhals_lazyframe
from narwhals.testing import assert_frame_equal
from tests.utils import POLARS_VERSION, Constructor

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}


def test_to_lowercase(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo", "BAR") * 2).name.to_lowercase())
    expected = {"foo": [2, 4, 6], "bar": [8, 10, 12]}
    if is_narwhals_lazyframe(result):
        result = result.collect()
    expected_frame = nw.from_dict(expected, backend=result.implementation)
    assert_frame_equal(result, expected_frame, check_dtypes=False)


def test_to_lowercase_after_alias(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < PkgVersion("1.32"):
        pytest.skip(reason="https://github.com/pola-rs/polars/issues/23765")
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("BAR")).alias("ALIAS_FOR_BAR").name.to_lowercase())
    expected = {"alias_for_bar": [4, 5, 6]}
    if is_narwhals_lazyframe(result):
        result = result.collect()
    expected_frame = nw.from_dict(expected, backend=result.implementation)
    assert_frame_equal(result, expected_frame, check_dtypes=False)


def test_to_lowercase_raise_anonymous(constructor: Constructor) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.select(nw.all().name.to_lowercase())
    expected = {"foo": [1, 2, 3], "bar": [4, 5, 6]}
    if is_narwhals_lazyframe(result):
        result = result.collect()
    expected_frame = nw.from_dict(expected, backend=result.implementation)
    assert_frame_equal(result, expected_frame, check_dtypes=False)
