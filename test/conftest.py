import os
import random
import sys
from itertools import count
from typing import Any, Iterator, Sequence, Tuple, Union

import numpy as np
from nptyping import NDArray


sys.path.append(os.path.join(__file__, "../.."))

Point = Sequence[float]


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
                [random.random() * 100 for _ in range(dimensions)]
                for _ in range(degree + 1)
            ]
        )


def random_times() -> Iterator[float]:
    """
    Infinite random values between 0 and 1
    :return:
    """
    return (random.random() for _ in count())


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
    return sum(
        (
            6 * (1 - time) * (p2 - 2 * p1 + p0),
            6 * time * (p3 - 2 * p2 + p1),
        )
    )
