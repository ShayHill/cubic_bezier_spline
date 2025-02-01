"""Test methods in BezierSpline class

:author: Shay Hill
:created: 10/4/2020
"""

# pyright: reportPrivateUsage = false


import math
import random
import sys
from typing import Any

import numpy as np
import pytest
from conftest import random_bezier_curves, random_times
from numpy import typing as npt

from cubic_bezier_spline.bezier_spline import BezierSpline

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

FArray: TypeAlias = npt.NDArray[np.floating[Any]]
SHORT_SPLINE = BezierSpline([[[0], [1]], [[1], [2]]])


class TestBezierSpline:
    def test_arrayable(self) -> None:
        """Convert to array when passed to np.array()"""
        spline = BezierSpline([[[0, 1], [1, 1]], [[1, 1], [2, 2]], [[2, 2], [3, 3]]])
        assert spline.as_array.shape == (3, 2, 2)

    @pytest.mark.parametrize("points", random_bezier_curves())
    def test_iter(self, points: FArray) -> None:
        """Iter spline._curves"""
        spline = BezierSpline(points)
        np.testing.assert_allclose(
            [x(0) for x in spline], [x[0] for x in spline._curves]
        )

    def test_call_low_open(self) -> None:
        """Clamp low values for open splines."""
        assert SHORT_SPLINE(-2.01) == SHORT_SPLINE(-800)

    def test_call_high_open(self) -> None:
        """Clamp high values for open splines."""
        assert SHORT_SPLINE(2.01) == SHORT_SPLINE(800)

    def test_call_low_closed(self) -> None:
        """Loop low values for closed splines."""
        spline = BezierSpline([[[0], [1]], [[1], [0]]])
        assert math.isclose(spline(-2.01)[0], spline(1.99)[0])

    def test_call_high_closed(self) -> None:
        """Loop high values for closed splines."""
        spline = BezierSpline([[[0], [1]], [[1], [0]]])
        assert math.isclose(spline(2.01)[0], spline(0.01)[0])

    @pytest.mark.parametrize("time", (random.random() * 2 for _ in range(50)))
    def test_call_simple(self, time: float) -> None:
        """Value at short spline is--by design--equal to time"""
        assert SHORT_SPLINE(time) == time

    @pytest.mark.parametrize(
        "points, time", zip(random_bezier_curves(), random_times())
    )
    def test_call(self, points: FArray, time: float) -> None:
        """At time n.p, return nth curve at p"""
        spline = BezierSpline(points)
        time = time * len(spline)
        np.testing.assert_allclose(
            spline(time), spline._curves[int(time)](time - int(time))
        )

    @pytest.mark.parametrize("points", random_bezier_curves())
    def test_bottom(self, points: FArray):
        """Return spline value at bottom of time interval"""
        np.testing.assert_allclose(BezierSpline(points)(0), points[0][0])

    @pytest.mark.parametrize("points", random_bezier_curves())
    def test_top(self, points: FArray):
        """Return spline value at top of time interval"""
        np.testing.assert_allclose(BezierSpline(points)(len(points)), points[-1][-1])

    @pytest.mark.parametrize("time", (0, 0.5, 1, 1.5, 2))
    def test_derivative(self, time: float) -> None:
        """Return value of spline at derivative"""
        assert SHORT_SPLINE(time, 1) == 1

    def test_uniform_high(self) -> None:
        """Return uniform high value"""
        assert SHORT_SPLINE(2) == 2


class TestSplit:
    def test_split_whole(self) -> None:
        """Split at spline endpoints."""
        spline = SHORT_SPLINE
        split = spline.split(0, 2)
        assert split.control_points == spline.control_points

    def test_split_whole_from_0(self) -> None:
        """Split at spline endpoints."""
        spline = SHORT_SPLINE
        split = spline.split(0, 0)
        assert split.control_points == spline.control_points

    def test_split_whole_from_max(self) -> None:
        """Split at spline endpoints."""
        spline = SHORT_SPLINE
        split = spline.split(2, 2)
        assert split.control_points == spline.control_points

    def test_beg_to_mid(self) -> None:
        """Split at spline endpoints."""
        spline = SHORT_SPLINE
        split = spline.split(0, 1)
        assert split.control_points == (((0.0,), (1.0,)),)

    def test_split_whole_from_middle(self) -> None:
        """Split at spline endpoints."""
        spline = SHORT_SPLINE
        with pytest.raises(ValueError):
            _ = spline.split(1, 0)

    def test_split_whole_loop_from_middle(self) -> None:
        """Split at spline endpoints."""
        spline = BezierSpline([[[0.0], [1.0]], [[1.0], [0.0]]])
        split = spline.split(1, 1)
        assert math.isclose(split(0)[0], spline(1)[0])
        assert math.isclose(split(2)[0], spline(1)[0])
        assert math.isclose(split(1)[0], spline(0)[0])

    def test_same_curve(self) -> None:
        """Split at the same time."""
        spline = SHORT_SPLINE
        split = spline.split(0.4, 0.5)
        assert math.isclose(split(0)[0], spline(0.4)[0])
        assert math.isclose(split(1)[0], spline(0.5)[0])

    def test_different_curves(self) -> None:
        """Split at the same time."""
        spline = SHORT_SPLINE
        split = spline.split(0.5, 1.5)
        assert math.isclose(split(0)[0], spline(0.5)[0])
        assert math.isclose(split(1, normalized=True)[0], spline(1.5)[0])

    def test_split_to_joint(self) -> None:
        """Don't leave any empty curves when splitting exactly at a joint."""
        spline = BezierSpline(
            [[[0.0], [1.0]], [[1.0], [2.0]], [[2.0], [3.0]], [[3.0], [4.0]]]
        )
        split = spline.split(1, 3)
        assert split.control_points == (((1.0,), (2.0,)), ((2.0,), (3.0,)))

    def test_split_as_loop(self) -> None:
        """Split through spline(0) when end is less than start."""
        spline = BezierSpline([[[0.0], [1.0]], [[1.0], [0.0]]])
        split = spline.split(1.8, 0.6)
        assert math.isclose(split(0)[0], spline(1.8)[0])
        assert math.isclose(split(2)[0], spline(0.6)[0])
        assert math.isclose(split(1)[0], spline(0)[0])


class TestReversed:
    @pytest.mark.parametrize(
        "points, time", zip(random_bezier_curves(), random_times())
    )
    def test_reversed(self, points: FArray, time: float) -> None:
        """At time n.p, return nth curve at p"""
        spline = BezierSpline(points)
        reversed_spline = spline.reversed
        time = time * len(spline)
        np.testing.assert_allclose(spline(time), reversed_spline(len(spline) - time))
