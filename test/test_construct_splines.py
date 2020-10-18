#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
""" Test cubic spline construction

:author: Shay Hill
:created: 10/14/2020
"""
from math import isclose

import numpy as np
import pytest

from bezier.construct_splines import get_approximating_spline, get_interpolating_spline
from .conftest import random_bezier_points


class TestApproximatingOpen:
    def test_linear(self) -> None:
        """Produce one cubic curve from two points."""
        spline = get_approximating_spline([[0, 0], [1, 1]])
        assert len(spline) == 1
        np.testing.assert_allclose(spline(0), [0, 0])
        np.testing.assert_allclose(spline(1), [1, 1])

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(2, 10)))
    def test_continuity(self, points) -> None:
        """degree 0, 1, and 2 continuous at ends"""
        spline = get_approximating_spline(points)
        for i in range(1, len(spline) - 1):
            for j in range(3):
                np.testing.assert_allclose(spline[i](1, j), spline[i + 1](0, j))


class TestApproximatingClosed:
    def test_linear(self) -> None:
        """Produce two cubic curves in a "loop" from two points."""
        spline = get_approximating_spline([[0, 0], [1, 1]], close=True)
        assert len(spline) == 2
        np.testing.assert_allclose(spline(0), [1 / 3, 1 / 3])
        np.testing.assert_allclose(spline(1), [2 / 3, 2 / 3])
        np.testing.assert_allclose(spline(2), [1 / 3, 1 / 3])

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(1, 10)))
    def test_continuity(self, points) -> None:
        """degree 0, 1, and 2 continuous at ends"""
        spline = get_approximating_spline(points, close=True)
        for i in range(0, len(spline)):
            for j in range(3):
                np.testing.assert_allclose(
                    spline[i](1, j), spline[(i + 1) % len(spline)](0, j)
                )

    def test_symmetrical(self) -> None:
        """Closed square of control points creates symmetrical spline"""
        spline = get_approximating_spline(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]], close=True
        )
        for i in range(5):
            assert abs(spline(i)[0]) == abs(spline(i)[1])


class TestInterpolatingOpen:
    def test_linear(self) -> None:
        """Produce two cubic curves in a "loop" from two points."""
        spline = get_interpolating_spline([[0, 0], [1, 1]])
        assert len(spline) == 1
        np.testing.assert_allclose(spline(0), [0, 0])
        np.testing.assert_allclose(spline(1), [1, 1])

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(1, 10)))
    def test_continuity(self, points) -> None:
        """degree 0, 1, and 2 continuous at ends"""
        spline = get_interpolating_spline(points)
        for i in range(0, len(spline) - 1):
            for j in range(3):
                np.testing.assert_allclose(
                    spline[i](1, j), spline[(i + 1) % len(spline)](0, j)
                )

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(1, 10)))
    def test_interpolation(self, points) -> None:
        """spline hits control points at knots"""
        spline = get_interpolating_spline(points)
        for i in range(0, len(spline)):
            np.testing.assert_allclose(spline(i), points[i])


class TestInterpolatingClosed:
    def test_linear(self) -> None:
        """Produce two cubic curves in a "loop" from two points."""
        spline = get_interpolating_spline([[0, 0], [1, 1]], close=True)
        assert len(spline) == 2
        np.testing.assert_allclose(spline(0), [0, 0])
        np.testing.assert_allclose(spline(1), [1, 1])
        np.testing.assert_allclose(spline(2), [0, 0])

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(1, 10)))
    def test_continuity(self, points) -> None:
        """degree 0, 1, and 2 continuous at ends"""
        spline = get_interpolating_spline(points, close=True)
        for i in range(0, len(spline)):
            for j in range(3):
                np.testing.assert_allclose(
                    spline[i](1, j), spline[(i + 1) % len(spline)](0, j)
                )

    @pytest.mark.parametrize("points", random_bezier_points(degree_limits=(1, 10)))
    def test_interpolation(self, points) -> None:
        """degree 0, 1, and 2 continuous at ends"""
        spline = get_interpolating_spline(points, close=True)
        for i in range(0, len(spline)):
            np.testing.assert_allclose(spline(i), points[i])

    def test_symmetrical(self) -> None:
        """Closed square of control points creates symmetrical spline"""
        spline = get_interpolating_spline(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]], close=True
        )
        for i in range(5):
            assert isclose(abs(spline(i)[0]), abs(spline(i)[1]))
