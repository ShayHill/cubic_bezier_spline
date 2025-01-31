"""De Casteljau Bezier operations.

:author: Shay Hill
:created: 10/2/2020

De Casteljau is Bezier at it's most basic. Here for testing / illustration.

"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from cubic_bezier_spline.control_point_casting import as_points_array
from cubic_bezier_spline.matrices import binom

Points = Union[Sequence[Sequence[float]], npt.NDArray[np.floating[Any]]]


def get_bezier_basis(points: Points, time: float) -> npt.NDArray[np.floating[Any]]:
    """Bezier basis function for testing.

    :param points: Bezier control points (takes an iterable so a BezierCurve instance
    can be passed.
        [ [x1, y1, z1], [x2, y2, z2], [x3, y3, z3], ...  ]
    :param time: time on Bezier spline
    :return: Bezier curve evaluated at time
    :raises ValueError: if points is not a 2D array of floats

    Another straightforward way to calculate a point on a Bezier curve. For testing.
    """
    points_ = as_points_array(points)
    n = points_.shape[0] - 1
    result = np.zeros((points_.shape[1],))
    for i in range(n + 1):
        point = points_[i]
        result += binom(n, i) * time**i * (1 - time) ** (n - i) * point
    return result


def iter_decasteljau_steps(
    points: Points, time: float
) -> Iterator[list[npt.NDArray[np.floating[Any]]]]:
    """Yield De Casteljau iterations.

    :param points: Bezier control points (takes an iterable so a BezierCurve instance
        can be passed.
    :param time: time on Bezier spline
    :yield: each iteration (including the first, which will = the input points) of
        the De Casteljau algorithm
    :return: None

    This is the value of a Bezier spline at time. De Casteljau algorithm works by
    recursively averaging consecutive Bezier control points. Using floats as points,
    De Casteljau would work as:

    1   ,   5   ,   9
        3   ,   8
           5.5

    In this case, the function would yield [1, 5, 9] then [3, 8] then [5.5]
    """
    points_ = as_points_array(points)
    points_list = [np.asarray(x).astype(float) for x in points_]
    yield points_list
    while len(points_list) > 1:
        points_list = [
            np.asarray(x * (1 - time) + y * time, dtype=float)
            for x, y in zip(points_list, points_list[1:])
        ]
        yield points_list


def get_decasteljau(points: Points, time: float) -> npt.NDArray[np.floating[Any]]:
    """Value of a non-rational Bezier curve at time.

    :param points: curve points
    :param time: time on curve
    :return: point on rational splint at time
    """
    return tuple(iter_decasteljau_steps(points, time))[-1][-1]


def get_split_decasteljau(
    points: Sequence[Sequence[float]], time: float
) -> tuple[list[npt.NDArray[np.floating[Any]]], list[npt.NDArray[np.floating[Any]]]]:
    """Split bezier at time using De Casteljau's algorithm.

    :param points: points in curve
    :param time: time at split
    :return: two curves of same dimensions as input points.
    """
    steps = tuple(iter_decasteljau_steps(points, time))
    return [x[0] for x in steps], [x[-1] for x in reversed(steps)]
