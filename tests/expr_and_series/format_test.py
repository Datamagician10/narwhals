from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.dependencies import is_narwhals_lazyframe
from narwhals.testing import assert_frame_equal

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_format(constructor: Constructor) -> None:
    df = nw.from_native(
        constructor(
            {
                "name": ["bob", "alice", "dodo"],
                "surname": ["builder", "wonderlander", "extinct"],
            }
        )
    )
    result = df.select(fmt=nw.format("hello {} {} wassup", "name", nw.col("surname")))
    expected = {
        "fmt": [
            "hello bob builder wassup",
            "hello alice wonderlander wassup",
            "hello dodo extinct wassup",
        ]
    }
    if is_narwhals_lazyframe(result):
        result = result.collect()
    expected_frame = nw.from_dict(expected, backend=result.implementation)
    assert_frame_equal(result, expected_frame, check_dtypes=False)
    result = df.select(fmt=nw.format("{} {} wassup", "name", nw.col("surname")))
    expected = {
        "fmt": ["bob builder wassup", "alice wonderlander wassup", "dodo extinct wassup"]
    }
    if is_narwhals_lazyframe(result):
        result = result.collect()
    expected_frame = nw.from_dict(expected, backend=result.implementation)
    assert_frame_equal(result, expected_frame, check_dtypes=False)


def test_format_invalid() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw.from_native(
        pd.DataFrame(
            {
                "name": ["bob", "alice", "dodo"],
                "surname": ["builder", "wonderlander", "extinct"],
            }
        )
    )
    with pytest.raises(ValueError, match="Expected 2 arguments, got 1"):
        df.select(fmt=nw.format("hello {} {} wassup", "name"))
