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

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generic, Iterable, Iterator, List, Optional, Tuple, TypeVar

import numpy as np

Point = Any  # 'np.ndarray[float]'


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


def decasteljau(points: Iterable[Point], time: float) -> Point:
    """
    Value of a non-rational Bezier curve at time.

    :param points: curve points
    :param time: time on curve
    :return:
    """
    return tuple(iter_decasteljau_steps(points, time))[-1][-1]


_G = TypeVar("_G", bound=Point)


CurveT = TypeVar("CurveT", bound="BezierCurve")


@dataclass(frozen=True)
class BezierCurve(Generic[_G]):
    """
    A non-rational Bezier curve.
    """

    _points: Tuple[_G]
    degree: int

    def __init__(self, *points: Iterable[float]) -> None:
        """
        Convert all points to ndarray.

        This allows for easy math and has the effect of ensuring no references exist
        in Bezier points.
        """
        object.__setattr__(self, "_points", tuple(np.array(x) for x in points))
        object.__setattr__(self, "degree", len(points) - 1)

    def __hash__(self) -> int:
        """To cache method calls"""
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
        if derivative == 0:
            return decasteljau(self, time)
        return self.derivative(derivative)(time)

    def split(self: CurveT, time: float) -> Tuple[CurveT, CurveT]:
        """
        Split a BezierCurve into two Bezier curves of the same degree.

        :param time: time at which to split the curve.
        :return: two new BezierCurve instances
        :raises: ValueError if not 0 <= time <= 1
        """
        steps = tuple(iter_decasteljau_steps(self._points, time))
        return (
            type(self)(*(x[0] for x in steps)),
            type(self)(*(x[-1] for x in reversed(steps))),
        )

    def elevated(self: CurveT, to_degree: Optional[int] = None) -> CurveT:
        """
        A new curve, elevated 1 or optionally more degrees.

        :param to_degree:
        :return:
        """
        if to_degree is None:
            to_degree = self.degree + 1
        if to_degree < self.degree:
            raise ValueError(
                "cannot elevate BezierCurve degree={self.degree} "
                "to BezierCurve degree={to_degree}"
            )

        points = self._points
        while len(points) - 1 < to_degree:
            elevated_points = [points[0]]
            for a, b in zip(points, points[1:]):
                time = len(elevated_points) / len(points)
                elevated_points.append(a * time + b * (1 - time))
            points = elevated_points + points[-1:]
        return type(self)(*points)

    @lru_cache
    def derivative(self: CurveT, derivative: int) -> CurveT:
        """
        nth derivative of a Bezier curve

        :param derivative: 0 -> the curve itself, 1 -> 1st, 2 -> 2nd, etc.
        :return: points to calculate nth derivative.

        The derivative of a Bezier curve of degree n is a Bezier curve of degree
        n-1 with control points n*(p1-p0), n(p2-p1), n(p3-p2), ...
        """
        if derivative == 0:
            return self
        if derivative > self.degree:
            raise ValueError(
                f"Bezier curve of degree {self.degree} "
                f"does not have a {derivative}th derivative."
            )
        points = [(y - x) * self.degree for x, y in zip(self, self[1:])]
        return type(self)(*points).derivative(derivative - 1)


