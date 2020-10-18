#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Test functions in ttf_extractor.curve_type.py

:author: Shay Hill
:created: 1/18/2020
"""

import random
from itertools import count

import numpy as np
import pytest

from bezier.bezier_curve import BezierCurve
from bezier.other_solvers import (
    get_bezier_basis,
    get_decasteljau,
    get_split_decasteljau,
)
from bezier.test.conftest import _cbez_d1, _cbez_d2, random_bezier_points, random_times


class TestCall:
    @pytest.mark.parametrize("points,time", zip(random_bezier_points(), random_times()))
    def test_call(self, points, time) -> None:
        """Test against formula"""
        curve = BezierCurve(points)(time)
        decasteljau = get_decasteljau(points, time)
        basis = get_bezier_basis(points, time)
        np.testing.assert_allclose(curve, decasteljau)
        np.testing.assert_allclose(curve, basis)


class TestCubicBezierDerivatives:
    """
    Test derivative argument in __call__ against explicitly defined cubic Bezier
    derivative formulas
    """

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=3), random_times())
    )
    def test_d1(self, points, time) -> None:
        """Test against formula"""
        curve = BezierCurve(points)
        np.testing.assert_allclose(curve(time, 1), _cbez_d1(*curve, time))

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=3), random_times())
    )
    def test_d2(self, points, time) -> None:
        """Test against formula"""
        curve = BezierCurve(points)
        np.testing.assert_allclose(curve(time, 2), _cbez_d2(*curve, time))

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=3), random_times())
    )
    def test_d3(self, points, time) -> None:
        """Test against formula"""
        curve = BezierCurve(points)
        p0, p1, p2, p3 = curve
        np.testing.assert_allclose(curve(time, 3), 6 * (p3 - 3 * p2 + 3 * p1 - p0))

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=3), random_times())
    )
    def test_d4(self, points, time) -> None:
        """Raise ValueError if derivative > degree"""
        curve = BezierCurve([(0, 0), (1, 0), (1, 1), (0, 1)])
        with pytest.raises(ValueError) as excinfo:
            curve(time, 4)
        assert "Bezier curve of degree" in str(excinfo.value)


class TestSplit:
    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=(0, 5)), random_times())
    )
    def test_against_dc(self, points, time) -> None:
        """
        Compare results to decasteljau.
        """
        aaa = get_split_decasteljau(points, time)
        curve = BezierCurve(points)
        bbb = curve.split(time)
        np.testing.assert_allclose(aaa[0], bbb[0]._points)
        np.testing.assert_allclose(aaa[1], bbb[1]._points)

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=(0, 5)), random_times())
    )
    def test_touch(self, points, time) -> None:
        """
        Last point of first curve == first point of second
        """
        curve = BezierCurve(points)
        beg, end = curve.split(time)
        np.testing.assert_array_equal(beg._points[-1], end._points[0])


class TestElevated:
    @pytest.mark.parametrize(
        "points,elevation,time",
        zip(
            random_bezier_points(),
            (random.randint(0, 5) for _ in count()),
            random_times(),
        ),
    )
    def test_elevated(self, points, elevation, time) -> None:
        """Curve is not changed by elevation"""
        curve = BezierCurve(points)
        elevated = curve.elevated(curve.degree + elevation)
        np.testing.assert_allclose(curve(time), elevated(time))

    @pytest.mark.parametrize(
        "points,elevation",
        zip(
            random_bezier_points(),
            (random.randint(0, 5) for _ in count()),
        ),
    )
    def test_degree_raised(self, points, elevation) -> None:
        """Degree increases"""
        curve = BezierCurve(points)
        elevated = curve.elevated(curve.degree + elevation)
        assert elevated.degree - curve.degree == elevation

    @pytest.mark.parametrize(
        "points,elevation,time",
        zip(
            random_bezier_points(degree_limits=(2, 10)),
            (random.randint(0, 5) for _ in count()),
            random_times(),
        ),
    )
    def test_derivative(self, points, elevation, time) -> None:
        """Derivative of curve is not changed by elevation"""
        curve = BezierCurve(points)
        elevated = curve.elevated(curve.degree + elevation)
        np.testing.assert_allclose(curve(time), elevated(time))
        np.testing.assert_allclose(curve(time, 2), elevated(time, 2))
