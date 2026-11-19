from __future__ import annotations

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from narwhals.dependencies import is_narwhals_lazyframe
from narwhals.testing import assert_frame_equal
from tests.utils import POLARS_VERSION, Constructor

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}


def map_func(s: str | None) -> str:
    return str(s)[::-1].lower()


def test_map(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo", "BAR") * 2).name.map(function=map_func))
    expected = {"oof": [2, 4, 6], "rab": [8, 10, 12]}
    if is_narwhals_lazyframe(result):
        result = result.collect()
    expected_frame = nw.from_dict(expected, backend=result.implementation)
    assert_frame_equal(result, expected_frame, check_dtypes=False)


def test_map_after_alias(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < PkgVersion("1.32"):
        pytest.skip(reason="https://github.com/pola-rs/polars/issues/23765")
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo")).alias("alias_for_foo").name.map(function=map_func))
    expected = {"oof_rof_saila": data["foo"]}
    if is_narwhals_lazyframe(result):
        result = result.collect()
    expected_frame = nw.from_dict(expected, backend=result.implementation)
    assert_frame_equal(result, expected_frame, check_dtypes=False)


def test_map_anonymous(constructor: Constructor) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.select(nw.all().name.map(function=map_func))
    expected = {"oof": [1, 2, 3], "rab": [4, 5, 6]}
    if is_narwhals_lazyframe(result):
        result = result.collect()
    expected_frame = nw.from_dict(expected, backend=result.implementation)
    assert_frame_equal(result, expected_frame, check_dtypes=False)
