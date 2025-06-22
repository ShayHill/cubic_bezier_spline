"""Test generating SVG data strings.

:author: Shay Hill
:created: 2024-12-13
"""

# pyright: reportPrivateUsage = false

import itertools as it
from collections.abc import Iterable
from typing import TypeVar

from cubic_bezier_spline import (
    BezierSpline,
    new_closed_approximating_spline,
    new_open_approximating_spline,
)
from cubic_bezier_spline.svg_data import (
    _svgd_join,
    _svgd_split,
    get_cpts_from_svgd,
    get_svgd_from_cpts,
)

_T = TypeVar("_T")


def pairwise(iterable: Iterable[_T]) -> Iterable[tuple[_T, _T]]:
    """Yield pairs of items from an iterable.

    :param iterable: items to pair
    :return: pairs of items from the iterable

    No it.pairwise in Python 3.9.
    """
    a, b = it.tee(iterable)
    _ = next(b, None)
    return zip(a, b)


def assert_svgd_equal(result: str, expect: str):
    """Assert result == expect and test helper functions.

    This is just a method for running bonus circular tests on other test data.
    """
    assert result == expect
    assert _svgd_join(*_svgd_split(expect)) == expect
    assert get_svgd_from_cpts(get_cpts_from_svgd(expect)) == expect


class TestClosedC2Continuous:
    def test_closed_approximating_spline(self):
        spline = new_closed_approximating_spline([(0, 0), (3, 0), (3, 3), (0, 3)])
        assert_svgd_equal(
            spline.svg_data,
            ("M0.5 0.5C1 0 2 0 2.5 0.5S3 2 2.5 2.5 1 3 0.5 2.5 0 1 0.5 0.5Z"),
        )

    def test_open_approximating_spline(self):
        spline = new_open_approximating_spline([(0, 0), (3, 0), (3, 3), (0, 3)])
        assert_svgd_equal(spline.svg_data, ("M0 0C1 0 2 0 2.5 0.5S3 2 2.5 2.5 1 3 0 3"))

    def test_linear_closed(self):
        curves = list(pairwise(((0, 0), (3, 0), (3, 3), (0, 3), (0, 0))))
        spline = BezierSpline(curves)
        assert_svgd_equal(spline.svg_data, "M0 0H3V3H0Z")

    def test_linear_open(self):
        curves = list(pairwise(((0, 0), (3, 0), (3, 3), (0, 3))))
        spline = BezierSpline(curves)
        assert_svgd_equal(spline.svg_data, "M0 0H3V3H0")

    def test_quadratic(self):
        curves = [[(0, 0), (1, 0)], [(1, 0), (2, 0), (3, 1)]]
        spline = BezierSpline(curves)
        assert_svgd_equal(spline.svg_data, ("M0 0H1T3 1"))
