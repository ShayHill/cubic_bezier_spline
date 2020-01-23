#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Bezier curve objects.

:author: Shay Hill
:created: 1/18/2020

I have a lot of Bezier curve code, but most of it is mixed up with other spline
types, rational Bezier, etc., none of which (except perhaps rational Bezier) are
useful for SVG creation. Creating new cubic-Bezier-only functionality here.

Toward the bottom, there are some explicit cubic Bezier functions using formulas. I
wanted these for testing, so I went ahead and made them available in the module. The
Bezier object uses De Casteljau for accuracy / stability.
"""

import numpy as np
from dataclasses import dataclass, astuple
from nptyping import Array
from typing import Iterable, Sequence, List, Iterator, Tuple, TypeVar, Generic
from functools import lru_cache

Point = Array[float]


def iter_decasteljau_steps(
    points: Iterable[Point], time: float
) -> Iterator[List[Point]]:
    """
    Yield De Casteljau iterations.

    :param points: Bezier control points (takes an iterable so a BezierCurve instance
    can be passed.
    :param time: time on Bezier spline
    :yield: each iteration (including the first, which will = the input points) of
    the De Casteljau algorithm

    This is the value of a Bezier spline at time. De Casteljau algorithm works by
    recursively averaging consecutive Bezier control points. Using floats as points,
    De Casteljau would work as:

    1   ,   5   ,   9
        3   ,   8
           5.5

    In this case, the function would yield [1, 5, 9] then [3, 8] then [5.5]
    """
    points = [x for x in points]
    yield points
    while len(points) > 1:
        points = [x * (1 - time) + y * time for x, y in zip(points, points[1:])]
        yield points


def get_derivative_points(points: Sequence[Point], derivative) -> List[Point]:
    """
    Control points for the derivative of a Bezier curve.

    :param points: control points
    :param derivative: 0 -> curve itself, 1 -> 1st derivative -> 2nd derivative
    :return: control points for derivative-th derivative.

    The derivative of a Bezier curve of degree n is a Bezier curve of degree n-1 with
    control points n*(p1-p0), n(p2-p1), n(p3-p2), ...
    """
    if derivative == 0:
        return [x for x in points]
    if derivative >= len(points):
        raise ValueError(
            f"Bezier curve of degree {len(points) - 1} "
            f"does not have a {derivative}th derivative."
        )
    points = get_derivative_points(points, derivative - 1)
    degree = len(points) - 1
    if points[1:]:
        return [(y - x) * degree for x, y in zip(points, points[1:])]


def decasteljau(points: Iterable[Point], time: float, derivative: int = 0) -> Point:
    """
    Value of a non-rational Bezier curve (or its derivative) at time.

    :param points: curve points
    :param time: time on curve
    :param derivative:
    :return:
    """
    points = [x for x in points]
    points = get_derivative_points(points, derivative)
    return tuple(iter_decasteljau_steps(points, time))[-1][-1]


_G = TypeVar("_G", bound=Point)


@dataclass
class BezierCurve(Generic[_G]):
    """
    Bog-standard Bezier.

    This is mostly just a holder for the Bezier points. Most of the functionality is
    available in functions in this module.
    """

    _points: List[_G]

    def __init__(self, *points: Iterable[float]) -> None:
        """
        Convert all points to ndarray.

        This allows for easy math and has the effect of ensuring no references exist
        in Bezier points.
        """
        self._points = [np.array(x) for x in points]

    def __hash__(self) -> int:
        """
        So we can cache method calls
        """
        return id(self)

    def __iter__(self):
        return iter(self._points)

    def __getitem__(self, item: int) -> _G:
        """
        Return item-th point

        :param item: index of [p0, p1, p2, p3]
        :return: Point
        """
        return self._points[item]

    def __call__(self, time: float, derivative: int = 0) -> _G:
        """
        Cubic Bezier calculation at time.

        :param time: time on curve (typically 0 - 1)
        :return: Non-rational cubic Bezier at time
        """
        return decasteljau(self.derivative(derivative), time)

    def split(self, time: float) -> Tuple["BezierCurve", "BezierCurve"]:
        """
        Split a BezierCurve into two Bezier curves of the same degree.

        :param time: time at which to split the curve.
        :return: two new BezierCurve instances
        :raises: ValueError if not 0 <= time <= 1
        """
        steps = tuple(iter_decasteljau_steps(self._points, time))
        return (
            BezierCurve(*(x[0] for x in steps)),
            BezierCurve(*(x[-1] for x in reversed(steps))),
        )

    @lru_cache
    def derivative(self, derivative: int) -> "BezierCurve":
        """
        Control points for the nth derivative of a Bezier curve

        :param derivative: 0 -> the curve itself, 1 -> 1st, 2 -> 2nd, etc.
        :return: points to calculate nth derivative.

        The derivative of a Bezier curve of degree n is a Bezier curve of degree
        n-1 with control points n*(p1-p0), n(p2-p1), n(p3-p2), ...
        """
        return BezierCurve(*get_derivative_points(self._points, derivative))

    def planar_curvature(self, time: float) -> float:
        """
        TODO: probably remove this. I don't think I'm using it.
        Curvature of a planar Bezier spline at time.

        :param time: time on spline
        :return: curvature from 0 to (theoretically) infinity

        This formula for curvature only works on planar curves, so all point values
        except x and y will be ignored. I am not checking against >2 dimensional
        points, because the dimensions >2 may not be holding geometric information.
        For instance, I plan to keep radius in the z dimension of x, y planar curves
        for an ersatz (but more robust) sphere sweep. In that instance, z would not
        contribute to curvature.
        """
        xi, yi = self(time, 1)[:2]
        xii, yii = self(time, 2)[:2]
        num = (xi * yii) - (yi * xii)
        den = (xi ** 2 + yi ** 2) ** (3 / 2)
        return num / den


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
