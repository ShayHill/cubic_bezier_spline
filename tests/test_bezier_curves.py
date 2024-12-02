"""Test BezierCurve class.

:author: Shay Hill
:created: 1/18/2020
"""

import random
import sys
from itertools import count

import numpy as np
import pytest
from conftest import cbez_d1, cbez_d2, random_bezier_points, random_times
from numpy import typing as npt

from cubic_bezier_spline.bezier_curve import BezierCurve
from cubic_bezier_spline.other_solvers import (
    get_bezier_basis,
    get_decasteljau,
    get_split_decasteljau,
)

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

FArray: TypeAlias = npt.NDArray[np.float64]


@pytest.mark.parametrize("points", random_bezier_points())
def test_arrayable(points: FArray) -> None:
    """Convert to array when passed to np.array()"""
    curve = BezierCurve(points)
    assert curve.as_array.shape == (len(points), len(points[0]))


class TestCall:
    @pytest.mark.parametrize("points,time", zip(random_bezier_points(), random_times()))
    def test_call(self, points: FArray, time: float) -> None:
        """Test against formula"""
        curve = BezierCurve(points)(time)
        decasteljau = get_decasteljau(points, time)
        basis = get_bezier_basis(points, time)
        np.testing.assert_allclose(curve, decasteljau)
        np.testing.assert_allclose(curve, basis)


class TestGetitem:
    @pytest.mark.parametrize("points", random_bezier_points())
    def test_getitem(self, points: FArray) -> None:
        """Test against formula"""
        curve = BezierCurve(points)
        index = random.randint(0, curve.degree)
        np.testing.assert_array_equal(curve[index], points[index])


class TestCubicBezierDerivatives:
    """
    Test derivative argument in __call__ against explicitly defined cubic Bezier
    derivative formulas
    """

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=3), random_times())
    )
    def test_d1(self, points: FArray, time: float) -> None:
        """Test against formula"""
        curve = BezierCurve(points)
        p0, p1, p2, p3 = curve.control_points
        cubic_d1 = cbez_d1(p0, p1, p2, p3, time)
        np.testing.assert_allclose(curve(time, 1), cubic_d1)

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=3), random_times())
    )
    def test_d2(self, points: FArray, time: float) -> None:
        """Test against formula"""
        curve = BezierCurve(points)
        # p0, p1, p2, p3 = p(tuple, curve.control_points))
        p0, p1, p2, p3 = map(tuple, curve.control_points)
        cubic_d2 = cbez_d2(p0, p1, p2, p3, time)
        np.testing.assert_allclose(curve(time, 2), cubic_d2)

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=3), random_times())
    )
    def test_d3(self, points: FArray, time: float) -> None:
        """Test against formula"""
        curve = BezierCurve(points)
        p0, p1, p2, p3 = curve.as_array
        np.testing.assert_allclose(curve(time, 3), 6 * (p3 - 3 * p2 + 3 * p1 - p0))

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=3), random_times())
    )
    def test_d4(self, points: FArray, time: float) -> None:
        """Raise ValueError if derivative > degree"""
        curve = BezierCurve([(0, 0), (1, 0), (1, 1), (0, 1)])
        with pytest.raises(ValueError) as excinfo:
            _ = curve(time, 4)
        assert "Bezier curve of degree" in str(excinfo.value)


class TestSplit:
    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=(0, 5)), random_times())
    )
    def test_against_dc(self, points: FArray, time: float) -> None:
        """
        Compare results to decasteljau.
        """
        point_sequence = [list(map(float, point)) for point in points]
        aaa = get_split_decasteljau(point_sequence, time)
        curve = BezierCurve(points)
        bbb = curve.split(time)
        np.testing.assert_allclose(aaa[0], bbb[0].control_points)
        np.testing.assert_allclose(aaa[1], bbb[1].control_points)

    @pytest.mark.parametrize(
        "points,time", zip(random_bezier_points(degree_limits=(0, 5)), random_times())
    )
    def test_touch(self, points: FArray, time: float) -> None:
        """
        Last point of first curve == first point of second
        """
        curve = BezierCurve(points)
        beg, end = curve.split(time)
        np.testing.assert_array_equal(beg.control_points[-1], end.control_points[0])

    @pytest.mark.parametrize("points", random_bezier_points())
    def test_split_0(self, points: FArray) -> None:
        """Split at 0 returns two curves.

        First is repeated point[0], second is original"""
        curve = BezierCurve(points)
        point, curve_ = curve.split(0)
        assert len(set(point.control_points)) == 1
        assert curve_ is curve

    @pytest.mark.parametrize("points", random_bezier_points())
    def test_split_1(self, points: FArray) -> None:
        """Split at 1 returns two curves.

        First is original, second is repeated point[-1]"""
        curve = BezierCurve(points)
        curve_, point = curve.split(1)
        assert len(set(point.control_points)) == 1
        assert curve_ is curve

    @pytest.mark.parametrize("points", random_bezier_points())
    def test_split_0_1(self, points: FArray) -> None:
        """Split at 0 and 1 returns three curves.

        First is repeated point[0], second is self, third is repeated point[-1]
        """
        curve = BezierCurve(points)
        beg, curve_, end = curve.split(0, 1)
        assert len(set(beg.control_points)) == 1
        assert beg.control_points[0] == curve_.control_points[0]
        assert len(set(end.control_points)) == 1
        assert end.control_points[0] == curve_.control_points[-1]
        assert curve_ is curve

    @pytest.mark.parametrize("points", random_bezier_points())
    def test_split_out_of_range(self, points: FArray) -> None:
        """Clip time values to [0, 1]"""
        curve = BezierCurve(points)
        assert curve.split(-1) == curve.split(0)
        assert curve.split(2) == curve.split(1)


class TestElevated:
    @pytest.mark.parametrize(
        "points,elevation,time",
        zip(
            random_bezier_points(),
            (random.randint(0, 5) for _ in count()),
            random_times(),
        ),
    )
    def test_elevated(self, points: FArray, elevation: int, time: float) -> None:
        """Curve is not changed by elevation"""
        curve = BezierCurve(points)
        elevated = curve.elevated(curve.degree + elevation)
        np.testing.assert_allclose(curve(time), elevated(time))

    @pytest.mark.parametrize("points,time", zip(random_bezier_points(), random_times()))
    def test_default_elevation(self, points: FArray, time: float) -> None:
        """Elevate curve to degree + 1 when None passed to elevated."""
        curve = BezierCurve(points)
        elevated = curve.elevated()
        assert elevated.degree == curve.degree + 1
        np.testing.assert_allclose(curve(time), elevated(time))

    @pytest.mark.parametrize("points", random_bezier_points((2, 10)))
    def test_error_when_decreasing_degree(self, points: FArray):
        curve = BezierCurve(points)
        to_degree = random.randint(0, curve.degree - 1)
        err_msg = (
            f"cannot elevate BezierCurve degree={curve.degree} "
            f"to BezierCurve degree={to_degree}"
        )
        with pytest.raises(ValueError, match=err_msg):
            _ = curve.elevated(to_degree)

    @pytest.mark.parametrize(
        "points,elevation",
        zip(random_bezier_points(), (random.randint(0, 5) for _ in count())),
    )
    def test_degree_raised(self, points: FArray, elevation: int) -> None:
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
    def test_derivative(self, points: FArray, elevation: int, time: float) -> None:
        """Derivative of curve is not changed by elevation"""
        curve = BezierCurve(points)
        elevated = curve.elevated(curve.degree + elevation)
        np.testing.assert_allclose(curve(time), elevated(time))
        np.testing.assert_allclose(curve(time, 2), elevated(time, 2))
