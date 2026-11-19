from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.dependencies import is_narwhals_lazyframe
from narwhals.exceptions import MultiOutputExpressionError
from narwhals.testing import assert_frame_equal

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_double_selected(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7, 8, 9]}
    df = nw.from_native(constructor(data))

    result = df.select(nw.col("a", "b") * 2)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12]}
    if is_narwhals_lazyframe(result):
        result = result.collect()
    expected_frame = nw.from_dict(expected, backend=result.implementation)
    assert_frame_equal(result, expected_frame, check_dtypes=False)

    result = df.select("z", nw.col("a", "b") * 2)
    expected = {"z": [7, 8, 9], "a": [2, 6, 4], "b": [8, 8, 12]}
    if is_narwhals_lazyframe(result):
        result = result.collect()
    expected_frame = nw.from_dict(expected, backend=result.implementation)
    assert_frame_equal(result, expected_frame, check_dtypes=False)

    with pytest.raises(MultiOutputExpressionError):
        df.select("a").select(nw.col("a") + nw.all())
