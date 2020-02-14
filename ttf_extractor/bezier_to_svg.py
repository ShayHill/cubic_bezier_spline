#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Convert a list of Bezier spline control points to an svg path

:author: Shay Hill
:created: 2/11/2020
"""

from typing import List, Tuple, Iterator
from ..bezier import BezierCurve
import numpy as np
from ..bezier.continuity import Continuity, get_continuity


def _iter_loops(curves: List[BezierCurve]) -> Iterator[List[BezierCurve]]:
    """
    Split a list of BezierCurves into closed loops.

    :param curves: A list of curves representing some number of closed loops.
    :return: curves separated into loops
    """
    loop = []
    for curve in curves:
        loop.append(curve)
        if np.array_equal(curve[-1], loop[0][0]):
            yield loop
            loop = []
    if loop:
        raise ValueError("curves contain open splines")


def _get_points_string(xs, ys):
    """Format xs and ys into space-delimited string"""
    pts = [" ".join(xy) for xy in zip(xs, ys)]
    return " ".join(pts)


def _loop_to_d(curves: List[BezierCurve]) -> str:
    ds = ["M" + " ".join(str(x) for x in curves[0][0])]
    for i, curve in enumerate(curves):
        xs = [str(x) for x, _ in curve]
        ys = [str(y) for _, y in curve]
        if curve.degree == 1:
            if ys[0] == ys[1]:
                ds.append("H" + xs[1])
            elif xs[0] == xs[1]:
                ds.append("V" + ys[1])
            else:
                ds.append("L" + _get_points_string(xs[1:], ys[1:]))
        elif curve.degree == 2:
            if i == 0 or curves[i - 1].degree != 2:
                if np.array_equal(curve[0], curve[1]):
                    ds.append("T" + _get_points_string(xs[2:], ys[2:]))
                else:
                    ds.append("Q" + _get_points_string(xs[1:], ys[1:]))
            elif get_continuity(*curves[i - 1 : i + 1]) >= Continuity.c1:
                ds.append("T" + _get_points_string(xs[2:], ys[2:]))
            else:
                ds.append("Q" + _get_points_string(xs[1:], ys[1:]))
        elif curve.degree == 3:
            if i == 0 or curves[i - 1].degree != 3:
                if np.array_equal(curve[0], curve[1]):
                    ds.append("S" + _get_points_string(xs[2:], ys[2:]))
                else:
                    ds.append("C" + _get_points_string(xs[1:], ys[1:]))
            elif get_continuity(*curves[i - 1 : i + 1]) >= Continuity.c1:
                ds.append("S" + _get_points_string(xs[2:], ys[2:]))
            else:
                ds.append("C" + _get_points_string(xs[1:], ys[1:]))
    if np.array_equal(curves[-1][-1], curves[0][0]) and ds[-1][0] in 'LHV':
        ds.pop(-1)
    ds.append("z")
    for i, (prev, this) in enumerate(zip(ds, ds[1:]), start=1):
        if prev[0] == this[0] or prev[0] == 'M' and this[0] in 'lL':
            ds[i] = ' ' + ds[i][1:]

    return "".join(ds)


def bezier_to_svg(curves: List[BezierCurve]):
    """
    Convert a list of BezierCurves into an svg 'd' path string.

    :param curves: list of BezierCurve objects
    :return: svg 'd' string 'M ... z'
    """
    return ''.join([_loop_to_d(x) for x in _iter_loops(curves)])
