#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Constructors for Bezier splines

:author: Shay Hill
:created: 10/4/2020
"""

from typing import Any, Callable, Sequence, Union

import numpy as np
import numpy.typing as npt

from .bezier_spline import BezierSpline

FArray = npt.NDArray[np.float_]
IArray = npt.NDArray[np.int_]

Cpts = Union[
    Sequence[Sequence[float]], Sequence[FArray], FArray
]


def _remove_matching_ends(
    func: Callable[[Cpts, Any], Any]
) -> Callable[[Cpts, Any], Any]:
    """
    Remove last when first == last.

    :param func: a function taking a sequence of control points as its first argument
    :return: a function which checks that Cpts are not wrapped (end != begin)

    To simplify the code, functions in this module will assume control points are
    never closed (e.g., a square will be defined by four points, whether the spline
    is closed or not.

    Just in case I forget or come across some data with closed (first == last)
    control points, this function will return the data to the expected format.
    """

    def wrapped(cpts: Cpts, *args, **kwargs) -> Any:
        """Remove last control point if wrapped. Cast to array."""
        cpts = np.asarray(cpts, dtype=float)
        if np.array_equal(cpts[0], cpts[-1]):
            cpts = cpts[:-1]
        return func(cpts, *args, **kwargs)

    return wrapped


def _get_141_matrix(dim: int, close: bool) -> IArray:
    """
    Create the 141 matrix necessary to get interpolated points from control points.

    :param dim: size of the matrix
    :param close: if True (default False), wrap the 1 values around for the top and
        bottom rows. This will produce points for a closed spline.
    :return: a (dim, dim) matrix with 1, 4, 1 on the diagonal and 0 elsewhere.
    """
    ones = [1] * (dim - 1)
    mat_141 = np.diag(ones, -1) + np.diag([4] * dim) + np.diag(ones, 1)
    if close:
        mat_141 += np.diag([1], dim - 1) + np.diag([1], -(dim - 1))
    return mat_141


@_remove_matching_ends
def get_approximating_spline(cpts: Cpts, close: bool) -> BezierSpline:
    """
    Approximate a set of points as a composite Bezier curve (Bezier spline)

    :param cpts: points to approximate
    :param close: if True (default False), wrap the ends of the spline around to form a
        closed loop.
    :return: A spline beginning at cpts[0], ending at cpts[-1], and shaped by cpts[1:-1]
        or, if closed, a spline shaped by all control points, beginning and ending at
        the same point.
    """
    if close:
        if not np.array_equal(cpts[0], cpts[-1]):
            cpts = np.concatenate([cpts, cpts[:1]])
        cpts = np.concatenate([cpts[-2:-1], cpts, cpts[1:2]])

    thirds = [[x, (2 * x + y) / 3, (x + 2 * y) / 3, y] for x, y in zip(cpts, cpts[1:])]
    for prev_curve, next_curve in zip(thirds, thirds[1:]):
        new_point = (prev_curve[2] + next_curve[1]) / 2
        prev_curve[-1] = new_point
        next_curve[0] = new_point

    if close:
        return BezierSpline(thirds[1:-1])
    return BezierSpline(thirds)


@_remove_matching_ends
def get_closed_interpolating_spline(cpts: Cpts) -> BezierSpline:
    """
    Get a closed cubic interpolating spline.

    :param cpts: points to interpolate
    :return: A spline passing through all control points, beginning and ending at
        cpts[0]
    """
    cpts = np.linalg.inv(_get_141_matrix(len(cpts), close=True)) @ cpts * 6
    return get_approximating_spline(cpts, True)


@_remove_matching_ends
def get_open_interpolating_spline(cpts: Cpts) -> BezierSpline:
    """
    Get an open cubic interpolating spline.

    :param cpts: points to interpolate
    :return: A spline passing through all control points.
    """
    if len(cpts) == 2:
        return get_approximating_spline(cpts, False)
    if len(cpts) == 3:
        midpt = (-cpts[0] + 6 * cpts[1] - cpts[2]) / 4
        # noinspection PyTypeChecker
        return get_approximating_spline(np.asarray([cpts[0], midpt, cpts[2]]), False)
    mat_141 = _get_141_matrix(len(cpts) - 2, False)
    mat_b = np.concatenate(
        [[6 * cpts[1] - cpts[0]], 6 * cpts[2:-2], [6 * cpts[-2] - cpts[-1]]]
    )
    interior_pts = np.linalg.inv(mat_141) @ mat_b
    # noinspection PyTypeChecker
    return get_approximating_spline(
        np.concatenate([cpts[:1], interior_pts, cpts[-1:]]), False
    )


# noinspection PyArgumentList
def get_interpolating_spline(cpts: Cpts, close: bool) -> BezierSpline:
    """
    Interpolate a set of points as a composite Bezier curve (Bezier spline)

    :param cpts: points to interpolate
    :param close: if True (default False), wrap the ends of the spline around to form a
        closed loop.
    :return: A spline passing through all control points.
    """
    if close:
        return get_closed_interpolating_spline(cpts)
    return get_open_interpolating_spline(cpts)
