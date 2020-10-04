#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Test functions in ttf_extractor.curve_type.py

:author: Shay Hill
:created: 1/18/2020
"""
from math import isnan

import numpy as np
import pytest
from typing import Iterator, Any, Tuple, Union
from nptyping import NDArray
import random
from bezier.bezier_curve import BezierCurve, Point
from bezier.other_solvers import (
    get_decasteljau,
    get_bezier_basis,
    get_split_decasteljau,
)
from bezier.matrices import get_pascals
from itertools import count


def random_bezier_points(
    degree_limits: Union[int, Tuple[int, int]] = (0, 10),
    dimension_limits: Union[int, Tuple[int, int]] = (1, 10),
) -> Iterator[NDArray[(Any, Any), float]]:
    """
    Iter sets of Bezier control points

    :yield: (degree + 1, dimensions) array of floats
    """
    if isinstance(degree_limits, int):
        degree_limits = (degree_limits, degree_limits)
    if isinstance(dimension_limits, int):
        dimension_limits = (dimension_limits, dimension_limits)
    for _ in range(100):
        degree = random.randint(*degree_limits)
        dimensions = random.randint(*dimension_limits)
        yield np.array(
            [
                [random.random() * 100 for i in range(dimensions)]
                for j in range(degree + 1)
            ]
        )


def random_times() -> Iterator[float]:
    """
    Infinite random values between 0 and 1
    :return:
    """
    return (random.random() for _ in count())


def get_normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class TestCall:
    @pytest.mark.parametrize("points,time", zip(random_bezier_points(), random_times()))
    def test_call(self, points, time) -> None:
        """Test against formula"""
        curve = BezierCurve(*points)(time)
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
        curve = BezierCurve(*points)
        np.testing.assert_allclose(curve(time, 1), _cbez_d1(*curve, time))

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=3), random_times())
    )
    def test_d2(self, points, time) -> None:
        """Test against formula"""
        curve = BezierCurve(*points)
        np.testing.assert_allclose(curve(time, 2), _cbez_d2(*curve, time))

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=3), random_times())
    )
    def test_d3(self, points, time) -> None:
        """Test against formula"""
        curve = BezierCurve(*points)
        p0, p1, p2, p3 = curve
        np.testing.assert_allclose(curve(time, 3), 6 * (p3 - 3 * p2 + 3 * p1 - p0))

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=3), random_times())
    )
    def test_d4(self, points, time) -> None:
        """Raise ValueError if derivative > degree"""
        curve = BezierCurve((0, 0), (1, 0), (1, 1), (0, 1))
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
        curve = BezierCurve(*points)
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
        curve = BezierCurve(*points)
        beg, end = curve.split(time)
        np.testing.assert_array_equal(beg._points[-1], end._points[0])


def _cbez(p0: Point, p1: Point, p2: Point, p3: Point, time: float) -> Point:
    """
    Cubic Bezier curve.

    :param p0: control point
    :param p1: control point
    :param p2: control point
    :param p3: control point
    :param time: time value on curve, typically 0 to 1
    :return: cubic Bezier curve value at time
    """
    return sum(
        (
            (1 - time) ** 3 * p0,
            3 * (1 - time) ** 2 * time * p1,
            3 * (1 - time) * time ** 2 * p2,
            time ** 3 * p3,
        )
    )


def _cbez_d1(p0: Point, p1: Point, p2: Point, p3: Point, time: float) -> Point:
    """
    First derivative of cubic Bezier at time.

    :param p0: control point
    :param p1: control point
    :param p2: control point
    :param p3: control point
    :param time: time value on curve, typically 0 to 1
    :return: first derivative of cubic Bezier curve at time
    """
    return sum(
        (
            3 * (1 - time) ** 2 * (p1 - p0),
            6 * (1 - time) * time * (p2 - p1),
            3 * time ** 2 * (p3 - p2),
        )
    )


def _cbez_d2(p0: Point, p1: Point, p2: Point, p3: Point, time: float) -> Point:
    """
    Second derivative of cubic Bezier at time.

    :param p0: control point
    :param p1: control point
    :param p2: control point
    :param p3: control point
    :param time: time value on curve, typically 0 to 1
    :return: second derivative of cubic Bezier curve at time
    """
    return sum((6 * (1 - time) * (p2 - 2 * p1 + p0), 6 * time * (p3 - 2 * p2 + p1),))
