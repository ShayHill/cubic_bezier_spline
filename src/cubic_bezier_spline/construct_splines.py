"""Constructors for Bezier splines.

:author: Shay Hill
::created: 10/4/2020
"""

from __future__ import annotations

import enum
import itertools as it
from collections.abc import Iterable, Sequence
from typing import Annotated, Any, TypeVar, Union

import numpy as np
import numpy.typing as npt

from cubic_bezier_spline.bezier_spline import BezierSpline
from cubic_bezier_spline.control_point_casting import (
    as_closed_points_array,
    as_open_points_array,
)

_T = TypeVar("_T")

Points = Union[Sequence[Sequence[float]], npt.NDArray[np.floating[Any]]]
APoints = Annotated[npt.NDArray[np.floating[Any]], "(-1, -1)"]

_LINEAR = 2
_QUADRATIC = 3


def pairwise(iterable: Iterable[_T]) -> Iterable[tuple[_T, _T]]:
    """Yield pairs of items from an iterable.

    :param iterable: items to pair
    :return: pairs of items from the iterable

    No it.pairwise in Python 3.9.
    """
    items_a, items_b = it.tee(iterable)
    _ = next(items_b, None)
    return zip(items_a, items_b)


class _OpenOrClosed(enum.Enum):
    """Enum for whether a spline is open or closed."""

    OPEN = enum.auto()
    CLOSED = enum.auto()


def _get_141_matrix(dim: int, open_or_closed: _OpenOrClosed) -> npt.NDArray[np.bool_]:
    """Create the 141 matrix necessary to get interpolated points from control points.

    :param dim: size of the matrix
    :param open_or_closes: will the spline be open or closed? If closed, the spline
        will wrap around to connect the first and last points.
    :return: a (dim, dim) matrix with 1, 4, 1 on the diagonal and 0 elsewhere.
    """
    ones = [1] * (dim - 1)
    mat_141 = np.diag(ones, -1) + np.diag([4] * dim) + np.diag(ones, 1)
    if open_or_closed == _OpenOrClosed.CLOSED:
        mat_141 += np.diag([1], dim - 1) + np.diag([1], -(dim - 1))
    return mat_141


def _new_linear_spline(cpts: Points, open_or_closed: _OpenOrClosed) -> BezierSpline:
    """Create a linear spline from a sequence of points.

    :param cpts: points to connect with line segments
    :param open_or_closes: will the spline be open or closed? If closed, the spline
        will wrap around to connect the first and last points.
    :return: A spline connecting the points with line segments.
    """
    if open_or_closed == _OpenOrClosed.CLOSED:
        cpts_ = as_closed_points_array(cpts)
    else:
        cpts_ = as_open_points_array(cpts)
    return BezierSpline(pairwise(cpts_))


def new_open_linear_spline(cpts: Points) -> BezierSpline:
    """Create a linear spline from a sequence of points.

    :param cpts: points to connect with line segments
    :return: A spline connecting the points with line segments.
    """
    return _new_linear_spline(cpts, _OpenOrClosed.OPEN)


def new_closed_linear_spline(cpts: Points) -> BezierSpline:
    """Create a linear spline from a sequence of points.

    :param cpts: points to connect with line segments
    :return: A spline connecting the points with line segments.
    """
    return _new_linear_spline(cpts, _OpenOrClosed.CLOSED)


def _new_approximating_spline(
    cpts: Points, open_or_closed: _OpenOrClosed
) -> BezierSpline:
    """Approximate a set of points as a composite Bezier curve (Bezier spline).

    :param cpts: points to approximate
    :param open_or_closes: will the spline be open or closed? If closed, the spline
        will wrap around to connect the first and last points.
    :return: A spline beginning at cpts[0], ending at cpts[-1], and shaped by
        cpts[1:-1] or, if closed, a spline shaped by all control points, beginning
        and ending at the same point.
    """
    if open_or_closed == _OpenOrClosed.CLOSED:
        cpts_ = as_closed_points_array(cpts)
        cpts_ = np.concatenate([cpts_[-2:-1], cpts_, cpts_[1:2]])
    else:
        cpts_ = as_open_points_array(cpts)

    thirds = [
        [x, (2 * x + y) / 3, (x + 2 * y) / 3, y] for x, y in zip(cpts_, cpts_[1:])
    ]
    for prev_curve, next_curve in zip(thirds, thirds[1:]):
        new_point = (prev_curve[2] + next_curve[1]) / 2
        prev_curve[-1] = new_point
        next_curve[0] = new_point

    if open_or_closed == _OpenOrClosed.CLOSED:
        return BezierSpline(thirds[1:-1])
    return BezierSpline(thirds)


def new_open_approximating_spline(cpts: Points) -> BezierSpline:
    """Approximate a set of points as a composite Bezier curve (Bezier spline).

    :param cpts: points to approximate
    :return: A spline beginning at cpts[0], ending at cpts[-1], and shaped by
        cpts[1:-1].
    """
    return _new_approximating_spline(cpts, _OpenOrClosed.OPEN)


def new_closed_approximating_spline(cpts: Points) -> BezierSpline:
    """Approximate a set of points as a composite Bezier curve (Bezier spline).

    :param cpts: points to approximate
    :return: A spline passing through all control points, beginning and ending at
        the same point.
    """
    return _new_approximating_spline(cpts, _OpenOrClosed.CLOSED)


# ===================================================================================
#   Interpolating splines
# ===================================================================================


def _get_b_matrix(cpts: APoints) -> Annotated[npt.NDArray[np.floating[Any]], "(p-2,v)"]:
    """Get the B matrix for a set of points.

    :param cpts: points to interpolate
    :return: A (p-2, v) matrix of scaled control points

    For a set of points (p, v), the B matrix is a (p-2, v) matrix of scaled control
    points needed to compute interpolating control points.

    This is part of solving for B control points that, when approximated, will
    effectively be S control points interpolated.

    [4, 1, 0, 0] [P1]   [6 * (S[1] - S[0])]
    [1, 4, 1, 0] [P2] = [6 *  S[2]        ]
    [0, 1, 4, 1] [P3]   [6 *  S[3]        ]
    [0, 0, 1, 4] [P4]   [6 * (S[4] - S[5])]
    """
    return np.concatenate(
        [[6 * cpts[1] - cpts[0]], 6 * cpts[2:-2], [6 * cpts[-2] - cpts[-1]]]
    )


def get_closed_b_points(cpts: Points) -> APoints:
    """Get points B that, when approximated, will interpolate the control points.

    :param cpts: points to be interpolated
    :return: B points that, when approximated, will interpolate cpts
    """
    cpts_ = as_open_points_array(cpts)
    mat_141 = _get_141_matrix(len(cpts_), open_or_closed=_OpenOrClosed.CLOSED)
    return np.linalg.inv(mat_141) @ cpts_ * 6


def get_open_b_points(cpts: Points) -> APoints:
    """Get points B that, when approximated, will interpolate the control points.

    :param cpts: points to be interpolated
    :return: B points that, when approximated, will interpolate cpts
    """
    cpts_ = as_open_points_array(cpts)
    if len(cpts_) == _LINEAR:
        return cpts_
    if len(cpts_) == _QUADRATIC:
        midpt = (-cpts_[0] + 6 * cpts_[1] - cpts_[2]) / 4
        return np.asarray([cpts_[0], midpt, cpts_[2]])
    mat_141 = _get_141_matrix(len(cpts_) - 2, _OpenOrClosed.OPEN)
    mat_b = _get_b_matrix(cpts_)
    interior_pts = np.linalg.inv(mat_141) @ mat_b
    return np.concatenate([cpts_[:1], interior_pts, cpts_[-1:]])


def new_closed_interpolating_spline(cpts: Points) -> BezierSpline:
    """Get a closed cubic interpolating spline.

    :param cpts: points to interpolate
    :return: A spline passing through all control points, beginning and ending at
        cpts[0]
    """
    b_points = get_closed_b_points(cpts)
    return new_closed_approximating_spline(b_points)


def new_open_interpolating_spline(cpts: Points) -> BezierSpline:
    """Get an open cubic interpolating spline.

    :param cpts: points to interpolate
    :return: A spline passing through all control points.
    """
    b_points = get_open_b_points(cpts)
    return new_open_approximating_spline(b_points)
