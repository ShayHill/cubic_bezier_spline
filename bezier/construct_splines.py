#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Constructors for Bezier splines

:author: Shay Hill
:created: 10/4/2020
"""

from .bezier_spline import BezierSpline
from typing import Iterable, Any
import numpy as np
from nptyping import NDArray


def _get_141_matrix(dim) -> NDArray[(Any, Any), int]:
    """
    Create the 141 matrix necessary to get interpolated points from control points.

    :param dim: size of the matrix
    :return: a (dim, dim) matrix with 1, 4, 1 on the diagonal and 0 elsewhere.
    """
    ones = [1] * (dim - 1)
    return np.diag(ones, -1) + np.diag([4] * dim) + np.diag(ones, 1)


def get_cubic_spline(
    cpts: Iterable[Iterable[float]], close: bool = False
) -> BezierSpline:
    """
    Interpret a set of points as a composite Bezier curve (Bezier spline)

    :param cpts: points to interpret
    :param close: if True (default False), wrap the ends of the spline around to form a
        closed loop.
    :return: A spline beginning at cpts[0], ending at cpts[-1], and shaped by cpts[1:-1]
        or, if closed, a spline shaped by all control points, beginning and ending at
        the same point.
    """
    cpts = np.array(tuple(iter(cpts)))
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

def get_interpolating_spline(
        cpts: Iterable[Iterable[float]], close: bool = False
) -> BezierSpline:
    # cpts = np.array([[1, -1], [-1, 2], [1, 4], [4, 3], [7, 5]])
    cpts = np.array(cpts)
    cpts = np.concatenate([cpts] * 10)
    aaa = _get_141_matrix(len(cpts)-2)
    bbb = np.array([6*cpts[1] - cpts[0]] + [6*x for x in cpts[2:-2]] + [6*cpts[-2]-cpts[-1]])
    bbbb = np.array([6*x for x in cpts[2:-2]])

    # bbb = aaa @ cpts[1:-1]
    ccc = np.linalg.inv(aaa) @ bbb
    ddd = np.linalg.inv(aaa)[:,1:-1] @ bbb[1:-1]
    breakpoint()
    cccc = np.linalg.inv(aaa)[1:-1] @ bbbb
    if close:
        ddd = get_cubic_spline(ccc, close=True)
    else:
        ddd = get_cubic_spline(np.concatenate([cpts[:1], ccc, cpts[-1:]]))
    breakpoint()



aaa = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
bbb = get_cubic_spline(aaa)
bbb = get_cubic_spline(aaa, close=True)
get_interpolating_spline(aaa, close=True)
breakpoint()
