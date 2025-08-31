"""Test generating SVG data strings.

:author: Shay Hill
:created: 2024-12-13
"""

# pyright: reportPrivateUsage = false

from typing import TypeVar

from cubic_bezier_spline import (
    BezierSpline,
    new_closed_approximating_spline,
    new_open_approximating_spline,
)
from cubic_bezier_spline.pairwise import pairwise

_T = TypeVar("_T")


class TestClosedC2Continuous:
    def test_closed_approximating_spline(self):
        spline = new_closed_approximating_spline([(0, 0), (3, 0), (3, 3), (0, 3)])
        assert spline.svgd == "M.5 .5C1 0 2 0 2.5 .5s.5 1.5 0 2S1 3 .5 2.5 0 1 .5 .5Z"

    def test_open_approximating_spline(self):
        spline = new_open_approximating_spline([(0, 0), (3, 0), (3, 3), (0, 3)])
        assert spline.svgd == "M0 0C1 0 2 0 2.5 .5s.5 1.5 0 2S1 3 0 3"

    def test_linear_closed(self):
        curves = list(pairwise(((0, 0), (3, 0), (3, 3), (0, 3), (0, 0))))
        spline = BezierSpline(curves)
        assert spline.svgd == "M0 0H3V3H0Z"

    def test_linear_open(self):
        curves = list(pairwise(((0, 0), (3, 0), (3, 3), (0, 3))))
        spline = BezierSpline(curves)
        assert spline.svgd == "M0 0H3V3H0"

    def test_quadratic(self):
        curves = [[(0, 0), (1, 0)], [(1, 0), (2, 0), (3, 1)]]
        spline = BezierSpline(curves)
        assert spline.svgd == "M0 0H1Q2 0 3 1"
