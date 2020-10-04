#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
""" De Casteljau Bezier operations

:author: Shay Hill
:created: 10/2/2020

De Casteljau is Bezier at it's most basic. Here for testing / illustration.

"""
from typing import Any, Iterable, Iterator, List, Tuple

import numpy as np
from nptyping import NDArray

# TODO: fix point type
from bezier.matrices import binom

Point = Any


def _get_curve_as_array(points: Iterable[Point]) -> NDArray[(Any, Any), float]:
    """
    Convert an iterable of points to a numpy array (works with BezierCurve instances)

    I wasn't sure how (or if it's possible) to ``asarray`` a BezierCurve instance.
    This works.
    """
    # noinspection PyTypeChecker
    return np.asarray(tuple(points), dtype=float)


def get_bezier_basis(points: Iterable[Point], time) -> NDArray[(Any,), float]:
    """
    Bezier basis function for testing.

    :param points: Bezier control points (takes an iterable so a BezierCurve instance
    can be passed.
    :param time: time on Bezier spline
    :return: Bezier curve evaluated at time

    Another straightforward way to calculate a point on a Bezier curve. For testing.
    """
    points = _get_curve_as_array(points)
    n = points.shape[0] - 1
    result = np.zeros((points.shape[1],))
    for i in range(n + 1):
        result += binom(n, i) * time ** i * (1 - time) ** (n - i) * points[i]
    # noinspection PyTypeChecker
    return result


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


def get_decasteljau(points: Iterable[Point], time: float) -> Point:
    """
    Value of a non-rational Bezier curve at time.

    :param points: curve points
    :param time: time on curve
    :return:
    """
    return tuple(iter_decasteljau_steps(points, time))[-1][-1]


def get_split_decasteljau(
    points: Iterable[Point], time: float
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Split bezier at time using De Casteljau's algorithm.

    :param points: points in curve
    :param time: time at split
    :return: two curves of same dimensions as input points.
    """
    steps = tuple(iter_decasteljau_steps(points, time))
    return tuple(x[0] for x in steps), tuple(x[-1] for x in reversed(steps))
