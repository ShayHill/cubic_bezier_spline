#!/usr/bin/env python3
"""Test methods in BezierSpline class

:author: Shay Hill
:created: 10/4/2020
"""

import random
from itertools import count

import numpy as np
import pytest

from cubic_bezier_spline.bezier_spline import BezierSpline, TimeIntervalError

from .conftest import random_bezier_curves, random_bezier_points, random_times

SHORT_SPLINE = BezierSpline([[[0], [1]], [[1], [2]]])


class TestBezierSpline:
    def test_arrayable(self) -> None:
        """Convert to array when passed to np.array()"""
        spline = BezierSpline([[[0, 1], [1, 1]], [[1, 1], [2, 2]], [[2, 2], [3, 3]]])
        assert spline.as_array.shape == (3, 2, 2)

    @pytest.mark.parametrize("points", random_bezier_curves())
    def test_iter(self, points) -> None:
        """Iter spline._curves"""
        spline = BezierSpline(points)
        np.testing.assert_allclose(
            [x(0) for x in spline], [x[0] for x in spline._curves]
        )

    def test_call_low(self) -> None:
        """Raise TimeIntervalError if time < 0"""
        with pytest.raises(TimeIntervalError):
            _ = SHORT_SPLINE(-0.01)

    def test_call_high(self) -> None:
        """Raise TimeIntervalError if time > len(curves)"""
        with pytest.raises(TimeIntervalError):
            _ = SHORT_SPLINE(2.01)

    @pytest.mark.parametrize("time", (random.random() * 2 for _ in range(50)))
    def test_call_simple(self, time) -> None:
        """Value at short spline is--by design--equal to time"""
        assert SHORT_SPLINE(time) == time

    @pytest.mark.parametrize(
        "points, time", zip(random_bezier_curves(), random_times())
    )
    def test_call(self, points, time) -> None:
        """At time n.p, return nth curve at p"""
        spline = BezierSpline(points)
        time = time * len(spline)
        np.testing.assert_allclose(
            spline(time), spline._curves[int(time)](time - int(time))
        )

    @pytest.mark.parametrize("points", random_bezier_curves())
    def test_bottom(self, points):
        """Return spline value at bottom of time interval"""
        np.testing.assert_allclose(BezierSpline(points)(0), points[0][0])

    @pytest.mark.parametrize("points", random_bezier_curves())
    def test_top(self, points):
        """Return spline value at top of time interval"""
        np.testing.assert_allclose(BezierSpline(points)(len(points)), points[-1][-1])

    @pytest.mark.parametrize("time", (0, 0.5, 1, 1.5, 2))
    def test_derivative(self, time) -> None:
        """Return value of spline at derivative"""
        assert SHORT_SPLINE(time, 1) == 1
