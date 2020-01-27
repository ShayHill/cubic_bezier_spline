#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Test functions in ttf_extractor.bezier_curves.py

:author: Shay Hill
:created: 1/18/2020
"""
from math import isnan

import numpy as np
import pytest

from bezier.bezier_curves import (
    BezierCurve,
    Point)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class TestCubicBezier:
    def test_call(self) -> None:
        """Test against formula"""
        for _ in range(100):
            points = [np.random.random(3) for _ in range(4)]
            time = np.random.random()

            curve = BezierCurve(*points)
            np.testing.assert_allclose(curve(time), _cbez(*curve, time))

    def test_d1(self) -> None:
        """Test against formula"""
        for _ in range(100):
            points = [np.random.random(3) for _ in range(4)]
            time = np.random.random()

            curve = BezierCurve(*points)
            np.testing.assert_allclose(curve(time, 1), _cbez_d1(*curve, time))

    def test_d2(self) -> None:
        """Test against formula"""
        for _ in range(100):
            points = [np.random.random(3) for _ in range(4)]
            time = np.random.random()

            curve = BezierCurve(*points)
            np.testing.assert_allclose(curve(time, 2), _cbez_d2(*curve, time))

    def test_d3(self) -> None:
        """Test against formula"""
        for _ in range(100):
            points = [np.random.random(3) for _ in range(4)]
            time = np.random.random()

            curve = BezierCurve(*points)
            p0, p1, p2, p3 = curve
            np.testing.assert_allclose(curve(time, 3), 6 * (p3 - 3 * p2 + 3 * p1 - p0))

    def test_d4(self) -> None:
        """Raise ValueError if derivative > degree"""
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        curve = BezierCurve((0, 0), (1, 0), (1, 1), (0, 1))
        with pytest.raises(ValueError) as excinfo:
            curve(0.5, 4)
        assert "Bezier curve of degree" in str(excinfo.value)

    def test_split(self) -> None:
        """
        New curves are continuous at ends and split.
        """
        for _ in range(100):
            points = [np.random.random(3) for _ in range(4)]
            time = np.random.random()
            curve = BezierCurve(*points)
            beg, end = curve.split(time)
            np.testing.assert_allclose(beg(1), end(0))
            np.testing.assert_allclose(curve(0), beg(0))
            np.testing.assert_allclose(curve(1), end(1))
            for derivative in range(1, 4):
                np.testing.assert_allclose(
                    normalized(beg(1, derivative)), normalized(end(0, derivative))
                )
                np.testing.assert_allclose(
                    normalized(curve(0, derivative)), normalized(beg(0, derivative))
                )
                np.testing.assert_allclose(
                    normalized(curve(1, derivative)), normalized(end(1, derivative))
                )


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