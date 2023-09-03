"""Constructors for Bezier splines.

:author: Shay Hill
::created: 10/4/2020
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Annotated

import numpy as np

from .bezier_spline import BezierSpline
from .control_point_casting import as_closed_points_array, as_open_points_array

if TYPE_CHECKING:
    import numpy.typing as npt

    from .type_hints import APoints, FArray, Points

_LINEAR = 2
_QUADRATIC = 3


class _OpenOrClosed(enum.Enum):
    """Enum for whether a spline is open or closed."""

    OPEN = enum.auto()
    CLOSED = enum.auto()


def _get_141_matrix(dim: int, open_or_closed: _OpenOrClosed) -> npt.NDArray[np.bool_]:
    """Create the 141 matrix necessary to get interpolated points from control points.

    :param dim: size of the matrix
    :param close: if True (default False), wrap the 1 values around for the top and
        bottom rows. This will produce points for a closed spline.
    :return: a (dim, dim) matrix with 1, 4, 1 on the diagonal and 0 elsewhere.
    """
    ones = [1] * (dim - 1)
    mat_141 = np.diag(ones, -1) + np.diag([4] * dim) + np.diag(ones, 1)
    if open_or_closed == _OpenOrClosed.CLOSED:
        mat_141 += np.diag([1], dim - 1) + np.diag([1], -(dim - 1))
    return mat_141


def _new_approximating_spline(
    cpts: Points, open_or_closed: _OpenOrClosed
) -> BezierSpline:
    """Approximate a set of points as a composite Bezier curve (Bezier spline).

    :param cpts: points to approximate
    :param close: if True (default False), wrap the ends of the spline around to form
        a closed loop.
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


def new_closed_interpolating_spline(cpts: Points) -> BezierSpline:
    """Get a closed cubic interpolating spline.

    :param cpts: points to interpolate
    :return: A spline passing through all control points, beginning and ending at
        cpts[0]
    """
    cpts_ = as_open_points_array(cpts)
    mat_141 = _get_141_matrix(len(cpts_), open_or_closed=_OpenOrClosed.CLOSED)
    cpts_ = np.linalg.inv(mat_141) @ cpts_ * 6
    return new_closed_approximating_spline(cpts_)


def _get_b_matrix(cpts: APoints) -> Annotated[FArray, "(p-2,v)"]:
    """Get the B matrix for a set of points.

    :param cpts: points to interpolate
    :return: A (p-2, v) matrix of scaled control points

    For a set of points (p, v), the B matrix is a (p-2, v) matrix of scaled control
    points needed to compute interpolating control points.
    """
    beg = [6 * cpts[1] - cpts[0]]
    mid = 6 * cpts[2:-2]
    end = [6 * cpts[-2] - cpts[-1]]
    mat_b = np.insert(mid, 0, beg, axis=0)
    return np.append(mat_b, end, axis=0)


def new_open_interpolating_spline(cpts: Points) -> BezierSpline:
    """Get an open cubic interpolating spline.

    :param cpts: points to interpolate
    :return: A spline passing through all control points.
    """
    cpts_ = as_open_points_array(cpts)
    if len(cpts_) == _LINEAR:
        return new_open_approximating_spline(cpts_)
    if len(cpts_) == _QUADRATIC:
        midpt = (-cpts_[0] + 6 * cpts_[1] - cpts_[2]) / 4
        return new_open_approximating_spline(np.asarray([cpts_[0], midpt, cpts_[2]]))
    mat_141 = _get_141_matrix(len(cpts_) - 2, _OpenOrClosed.OPEN)
    mat_b = _get_b_matrix(cpts_)
    interior_pts = np.linalg.inv(mat_141) @ mat_b
    return new_open_approximating_spline(
        np.concatenate([cpts_[:1], interior_pts, cpts_[-1:]])
    )
