"""Test generating SVG data strings.

:author: Shay Hill
:created: 2024-12-13
"""

import itertools as it
from collections.abc import Iterable
from typing import TypeVar

from cubic_bezier_spline import (
    BezierSpline,
    new_closed_approximating_spline,
    new_open_approximating_spline,
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


class TestClosedC2Continuous:
    def test_closed_approximating_spline(self):
        spline = new_closed_approximating_spline([(0, 0), (3, 0), (3, 3), (0, 3)])
        assert spline.svg_data == (
            "M0.5 0.5C1 0 2 0 2.5 0.5 3 1 3 2 2.5"
            + " 2.5 2 3 1 3 0.5 2.5 0 2 0 1 0.5 0.5Z"
        )

    def test_open_approximating_spline(self):
        spline = new_open_approximating_spline([(0, 0), (3, 0), (3, 3), (0, 3)])
        assert spline.svg_data == ("M0 0C1 0 2 0 2.5 0.5 3 1 3 2 2.5 2.5 2 3 1 3 0 3")

    def test_linear_closed(self):
        curves = list(pairwise(((0, 0), (3, 0), (3, 3), (0, 3), (0, 0))))
        spline = BezierSpline(curves)
        assert spline.svg_data == "M0 0H3V3H0Z"

    def test_linear_open(self):
        curves = list(pairwise(((0, 0), (3, 0), (3, 3), (0, 3))))
        spline = BezierSpline(curves)
        assert spline.svg_data == "M0 0H3V3H0"

    def test_quadratic(self):
        curves = [[(0, 0), (1, 0)], [(1, 0), (2, 0), (3, 1)]]
        spline = BezierSpline(curves)
        assert spline.svg_data == ("M0 0H1Q2 0 3 1")
