"""Closely approximate the length of a Bezier curve.

Sum the lengths of the control-point segments. This will be equal to (for a straight
line) or greater than the length between the first and last control points. If the
difference is very small, return the length. If the difference is meaningful (uses
default tolerance values of np.isclose), divide the curve at 0.5 and repeat the
process on each half.

This isn't the fastest way to get an approximate length, but the approximation is
much better than Gaussian quadrature or other numerical methods I tried.

:author: Shay Hill
:created: 2025-05-21
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from cubic_bezier_spline.bezier_curve import BezierCurve


def _get_cp_length(control_points: Sequence[Sequence[float]]) -> float:
    """Get the combined length of the control-point segments.

    :param control_points: control points
    :return: combined length of the control-point segments
    """
    cp_array = np.asarray(control_points, dtype=float)
    pairwise = zip(cp_array, cp_array[1:])
    return float(sum(np.linalg.norm(p1 - p0) for p0, p1 in pairwise))


def _get_ep_length(control_points: Sequence[Sequence[float]]) -> np.floating[Any]:
    """Get the length between the first and last control points.

    :param control_points: control points
    :return: length between the first and last control points
    """
    cp_array = np.asarray(control_points, dtype=float)
    return np.linalg.norm(cp_array[-1] - cp_array[0])


def _get_cp_length_and_error(curve: BezierCurve) -> tuple[float, np.floating[Any]]:
    """Get length of cp segments and delta from length between endpoints.

    :param curve: Bezier curve
    :return: a tuple: length of cp segments, error
    """
    cpts = curve.control_points
    ep_norm = _get_ep_length(cpts)
    cp_norm = _get_cp_length(cpts)
    return cp_norm, cp_norm - ep_norm


def _iter_sub_lengths(curve: BezierCurve) -> Iterator[float]:
    """Get the approximate length of a Bezier curve.

    :param curve: Bezier curve
    :return: approximate length of the Bezier curve

    This is a good appriximation.
    """
    if curve.degree == 0:
        return
    if curve.degree == 1:
        yield _get_cp_length(curve.control_points)
        return

    curves = [curve]
    errors = [_get_cp_length_and_error(curve)]
    while curves:
        if np.isclose(errors[-1][1], 0):
            _ = curves.pop()
            yield errors.pop()[0]
            continue
        curves.extend(curves.pop().split(0.5))
        errors[-1:] = [_get_cp_length_and_error(x) for x in curves[-2:]]


def get_approximate_curve_length(curve: BezierCurve) -> float:
    """Get the approximate length of a Bezier curve.

    :param curve: Bezier curve
    :return: approximate length of the Bezier curve

    This is a simple approximation that calculates the length of the control-point
    segments. It is not very accurate, but it is fast.
    """
    return sum(_iter_sub_lengths(curve))
