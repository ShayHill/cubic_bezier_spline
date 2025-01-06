"""Configure tests for pytest.

:author: Shay Hill
:created: 2024-12-02
"""

from __future__ import annotations

import random
import sys
from collections.abc import Iterator, Sequence
from itertools import count
from typing import Annotated, Any

import numpy as np
from numpy import typing as npt

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


FArray: TypeAlias = npt.NDArray[np.floating[Any]]
Point = Sequence[float]


def random_bezier_points(
    degree_limits: int | tuple[int, int] = (0, 10),
    dimension_limits: int | tuple[int, int] = (1, 10),
) -> Iterator[Annotated[FArray, (-1, -1)]]:
    """Yield sets of Bezier control points.

    :yield: (degree + 1, dimensions) array of floats
    """
    if isinstance(degree_limits, int):
        degree_limits = (degree_limits, degree_limits)
    if isinstance(dimension_limits, int):
        dimension_limits = (dimension_limits, dimension_limits)
    for _ in range(50):
        degree = random.randint(*degree_limits)
        dimensions = random.randint(*dimension_limits)
        yield np.array(
            [
                [random.random() * 100 for _ in range(dimensions)]
                for _ in range(degree + 1)
            ]
        )


def random_bezier_curves(
    degree_limits: int | tuple[int, int] = (0, 10),
    dimension_limits: int | tuple[int, int] = (1, 10),
    splines_limits: int | tuple[int, int] = (1, 10),
) -> Iterator[Annotated[FArray, (-1, -1)]]:
    """Return a 3-D vector of spline curve points.

    [ curve, curve, curve ]

    :param degree_limits:
    :param dimension_limits:
    :param splines_limits:
    :return:
    """
    if isinstance(degree_limits, int):
        degree_limits = (degree_limits, degree_limits)
    if isinstance(dimension_limits, int):
        dimension_limits = (dimension_limits, dimension_limits)
    if isinstance(splines_limits, int):
        splines_limits = (splines_limits, splines_limits)
    for _ in range(50):
        degree = random.randint(*degree_limits)
        dimensions = random.randint(*dimension_limits)
        splines = random.randint(*splines_limits)
        yield np.array(
            [
                [
                    [random.random() * 100 for _ in range(dimensions)]
                    for _ in range(degree + 1)
                ]
                for _ in range(splines)
            ]
        )


def random_times() -> Iterator[float]:
    """Infinite random values between 0 and 1.

    :return: random values
    """
    return (random.random() for _ in count())


def random_indices() -> Iterator[int]:
    """Return infinite random integers (point indices) 0, 1, 2, 3.

    :return: random values
    """
    return (random.randint(0, 3) for _ in count())


def cbez(p0: Point, p1: Point, p2: Point, p3: Point, time: float) -> Point:
    """Cubic Bezier curve.

    :param p0: control point
    :param p1: control point
    :param p2: control point
    :param p3: control point
    :param time: time value on curve, typically 0 to 1
    :return: cubic Bezier curve value at time
    """
    a0, a1, a2, a3 = map(np.asarray, [p0, p1, p2, p3])
    pt_array = np.sum(
        (
            (1 - time) ** 3 * a0,
            3 * (1 - time) ** 2 * time * a1,
            3 * (1 - time) * time**2 * a2,
            time**3 * a3,
        ),
        axis=0,
    )
    return tuple(map(float, pt_array))


def cbez_d1(p0: Point, p1: Point, p2: Point, p3: Point, time: float) -> Point:
    """First derivative of cubic Bezier at time.

    :param p0: control point
    :param p1: control point
    :param p2: control point
    :param p3: control point
    :param time: time value on curve, typically 0 to 1
    :return: first derivative of cubic Bezier curve at time
    """
    a0, a1, a2, a3 = map(np.asarray, [p0, p1, p2, p3])
    pt_array = np.sum(
        (
            3 * (1 - time) ** 2 * (a1 - a0),
            6 * (1 - time) * time * (a2 - a1),
            3 * time**2 * (a3 - a2),
        ),
        axis=0,
    )
    return tuple(map(float, pt_array))


def cbez_d2(p0: Point, p1: Point, p2: Point, p3: Point, time: float) -> Point:
    """Second derivative of cubic Bezier at time.

    :param p0: control point
    :param p1: control point
    :param p2: control point
    :param p3: control point
    :param time: time value on curve, typically 0 to 1
    :return: second derivative of cubic Bezier curve at time
    """
    a0, a1, a2, a3 = map(np.asarray, [p0, p1, p2, p3])
    pt_array = np.sum(
        (6 * (1 - time) * (a2 - 2 * a1 + a0), 6 * time * (a3 - 2 * a2 + a1)), axis=0
    )
    return tuple(map(float, pt_array))
