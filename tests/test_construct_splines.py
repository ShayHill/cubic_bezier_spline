"""Test cubic spline construction

:author: Shay Hill
:created: 10/14/2020
"""

import sys
from math import isclose
from typing import Any

import numpy as np
import pytest
from conftest import random_bezier_points
from numpy import typing as npt

from cubic_bezier_spline.construct_splines import (
    new_closed_approximating_spline,
    new_closed_interpolating_spline,
    new_closed_linear_spline,
    new_open_approximating_spline,
    new_open_interpolating_spline,
    new_open_linear_spline,
)

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

FArray: TypeAlias = npt.NDArray[np.floating[Any]]


class TestClosedControlPoints:
    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(2, 10)))
    def test_approximating_open(self, points: FArray) -> None:
        """Returns same value for closed or open control points"""
        open_pts = new_open_approximating_spline(points)
        closed_pts = new_open_approximating_spline(np.concatenate([points, points[:1]]))
        np.testing.assert_allclose(open_pts, closed_pts)


class TestApproximatingOpen:
    def test_linear(self) -> None:
        """Produce one cubic curve from two points."""
        spline = new_open_approximating_spline([[0, 0], [1, 1]])
        assert len(spline) == 1
        np.testing.assert_allclose(spline(0), [0, 0])
        np.testing.assert_allclose(spline(1), [1, 1])

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(2, 10)))
    def test_continuity(self, points: FArray) -> None:
        """Degree 0, 1, and 2 continuous at ends"""
        spline = new_open_approximating_spline(points)
        for i in range(1, len(spline) - 1):
            for j in range(3):
                np.testing.assert_allclose(spline[i](1, j), spline[i + 1](0, j))


class TestApproximatingClosed:
    def test_linear(self) -> None:
        """Produce two cubic curves in a "loop" from two points."""
        spline = new_closed_approximating_spline([[0, 0], [1, 1]])
        assert len(spline) == 2
        np.testing.assert_allclose(spline(0), [1 / 3, 1 / 3])
        np.testing.assert_allclose(spline(1), [2 / 3, 2 / 3])
        np.testing.assert_allclose(spline(2), [1 / 3, 1 / 3])

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(1, 10)))
    def test_continuity(self, points: FArray) -> None:
        """Degree 0, 1, and 2 continuous at ends"""
        spline = new_closed_approximating_spline(points)
        for i in range(len(spline)):
            for j in range(3):
                np.testing.assert_allclose(
                    spline[i](1, j), spline[(i + 1) % len(spline)](0, j)
                )

    def test_symmetrical(self) -> None:
        """Closed square of control points creates symmetrical spline"""
        spline = new_closed_approximating_spline([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        for i in range(5):
            assert isclose(abs(spline(i)[0]), abs(spline(i)[1]))


class TestInterpolatingOpen:
    def test_linear(self) -> None:
        """Produce two cubic curves in a "loop" from two points."""
        spline = new_open_interpolating_spline([[0, 0], [1, 1]])
        assert len(spline) == 1
        np.testing.assert_allclose(spline(0), [0, 0])
        np.testing.assert_allclose(spline(1), [1, 1])

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(1, 10)))
    def test_continuity(self, points: FArray) -> None:
        """Degree 0, 1, and 2 continuous at ends"""
        spline = new_open_interpolating_spline(points)
        for i in range(len(spline) - 1):
            for j in range(3):
                np.testing.assert_allclose(
                    spline[i](1, j), spline[(i + 1) % len(spline)](0, j)
                )

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(1, 10)))
    def test_interpolation(self, points: FArray) -> None:
        """Spline hits control points at knots"""
        spline = new_open_interpolating_spline(points)
        for i in range(len(spline)):
            np.testing.assert_allclose(spline(i), points[i])


class TestInterpolatingClosed:
    def test_linear(self) -> None:
        """Produce two cubic curves in a "loop" from two points."""
        spline = new_closed_interpolating_spline([[0, 0], [1, 1]])
        assert len(spline) == 2
        np.testing.assert_allclose(spline(0), [0, 0])
        np.testing.assert_allclose(spline(1), [1, 1])
        np.testing.assert_allclose(spline(2), [0, 0])

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(1, 10)))
    def test_continuity(self, points: FArray) -> None:
        """Degree 0, 1, and 2 continuous at ends"""
        spline = new_closed_interpolating_spline(points)
        for i in range(len(spline)):
            for j in range(3):
                np.testing.assert_allclose(
                    spline[i](1, j), spline[(i + 1) % len(spline)](0, j)
                )

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(1, 10)))
    def test_interpolation(self, points: FArray) -> None:
        """Degree 0, 1, and 2 continuous at ends"""
        spline = new_closed_interpolating_spline(points)
        for i in range(len(spline)):
            np.testing.assert_allclose(spline(i), points[i])

    def test_symmetrical(self) -> None:
        """Closed square of control points creates symmetrical spline"""
        spline = new_closed_interpolating_spline([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        for i in range(5):
            assert isclose(abs(spline(i)[0]), abs(spline(i)[1]))


class TestLiner:
    def test_linear_open(self) -> None:
        """Produce a linear spline between each pair of points."""
        spline = new_open_linear_spline([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert spline.svg_data == "M0 0H1V1H0"

    def test_linear_closed(self) -> None:
        """Produce a linear spline between each pair of points."""
        spline = new_closed_linear_spline([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert spline.svg_data == "M0 0H1V1H0Z"
