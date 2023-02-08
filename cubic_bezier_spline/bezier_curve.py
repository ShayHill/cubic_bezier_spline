"""BezierCurve object.

:author: Shay Hill
:created: 1/18/2020

This uses matrix math to evaluate the Bezier curve. The math benefits from cacheing, so
the curve object is immutable.
"""

from __future__ import annotations

import dataclasses
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, TypeVar

import numpy as np
from paragraphs import par

from .control_point_casting import as_nested_tuple, as_points_array
from .matrices import get_mix_matrix

if TYPE_CHECKING:
    from .type_hints import APoints, FArray, Points, TPoints


_BezierCurveT = TypeVar("_BezierCurveT", bound="BezierCurve")


def _interp_floats(at_0: float, at_1: float, time: float) -> float:
    """Interpolate between two floats.

    :param at_0: value at time 0
    :param at_1: value at time 1
    :param time: time in [0, 1]
    :return: interpolated value
    """
    return at_0 + (at_1 - at_0) * time


@dataclasses.dataclass(frozen=True)
class BezierCurve:
    """A non-rational Bezier curve.

    Array annotations are used to document the shape of the arrays. I've used the
    convention `p` for the number of points (degree + 1) and `v` for the
    dimensionality of the space.

    For instance
    * the `as_array` array is a 2D array of shape (p, v). [point, point, point, ...]
    * the `tmat` array is a 1D array of shape (p,). [t^0, t^1, t^2, ...]
    * the `zmat` array is a 2D array of shape (p, p).
    """

    control_points: TPoints
    as_array: APoints = dataclasses.field(init=False, repr=False, compare=False)

    def __init__(self, points: Points) -> None:
        """Initialize BezierCurve.

        :param points: control points, each a sequence of floats

        Stores points internally as a tuple of tuples (`.control_points`) as a
        canonical representation. Also stores an array of the same information
        (`.points`), for convenience and to give argument validation through
        `as_points_array`.
        """
        object.__setattr__(self, "as_array", as_points_array(points))  # validates
        object.__setattr__(self, "control_points", as_nested_tuple(self.as_array))

    @property
    def degree(self) -> int:
        """Degree of curve.

        :return: len(self.control_points) - 1
        """
        return len(self.control_points) - 1

    def __getitem__(self, item: int) -> FArray:
        """Return item-th point.

        :param item: index of [p0, p1, p2, p3]
        :return: One control point as an array
        """
        return self.as_array[item]

    def __array__(self) -> Points:
        """Return self.as_array (it's an array anyway).

        :return: self.as_array when encountering np.array(self)
        """
        return self.as_array

    def __call__(self, time: float, derivative: int = 0) -> FArray:
        """Cubic Bezier calculation at time.

        :param time: time on curve (typically 0 - 1)
        :return: Non-rational Bezier at time
        """
        if derivative == 0:
            return self._get_tmat(time) @ self._mixed_points
        return self.derivative(derivative)(time)

    def _get_tmat(self, time: float) -> Annotated[FArray, "(p,)"]:
        """Get the t matrix for time.

        :param time: time on curve (typically 0 - 1)
        :return: t matrix for time: [1, t, t^2, t^3]
        """
        return np.array([time**x for x in range(self.degree + 1)])

    def _get_zmat(self, time: float) -> Annotated[FArray, "(p, p)"]:
        """Get a 2D zero matrix with tmat on the diagonal.

        :param time: time on curve (typically 0 - 1)
        :return: 2D zero matrix with tmat on the diagonal
        [
            [1, 0, 0, 0],
            [0, t, 0, 0],
            [0, 0, t^2, 0],
            [0, 0, 0, t^3]
        ]
        """
        return np.diagflat(self._get_tmat(time))

    @cached_property
    def _mmat(self) -> Annotated[FArray, "(p, p)"]:
        """Get the mix matrix for this curve.

        :return: mix matrix for this curve
        """
        return get_mix_matrix(self.degree + 1)

    @cached_property
    def _mixed_points(self) -> Annotated[FArray, "(p,v)"]:
        """Points scaled by binomial coefficients.

        :return: Points scaled by binomial coefficients

        Scale this by time matrix to evaluate curve at time.
        """
        return self._mmat @ self.as_array

    def split(self: _BezierCurveT, *time_args: float) -> list[_BezierCurveT]:
        """Split a BezierCurve into two Bezier curves of the same degree.

        :param time_args: time at which to split the curve. Multiple args accepted
        :return: two new BezierCurve instances
        :raises: ValueError if not 0 <= time <= 1
        """
        curves = [self]
        time_at = 0.0
        for time in time_args:
            time_prime = _interp_floats(time_at, 1, time)
            qmat = (
                np.linalg.inv(curves[-1]._mmat)
                @ curves[-1]._get_zmat(time_prime)
                @ curves[-1]._mmat
            )

            qmat_prime = np.zeros_like(qmat)
            for i in range(qmat.shape[0]):
                j = i + 1
                qmat_prime[-j, -j:] = qmat[i, :j]
            curves[-1:] = [
                type(self)(qmat @ curves[-1].as_array),
                type(self)(qmat_prime @ curves[-1].as_array),
            ]
            time_at = time
        return curves

    def elevated(self, to_degree: int | None = None) -> BezierCurve:
        """Create a new curve, elevated 1 or optionally more degrees.

        :param to_degree: final degree of Bezier curve
        :return: Bezier curve of identical shape with degree increased
        :raise ValueError: if to_degree is less than current degree
        """
        if to_degree == self.degree:
            return self
        if to_degree is None:
            to_degree = self.degree + 1
        elif to_degree < self.degree:
            msg = par(
                f"""cannot elevate BezierCurve degree={self.degree} to BezierCurve
                degree={to_degree}"""
            )
            raise ValueError(msg)
        nn, ps = len(self.control_points), self.as_array
        # mypy likes linspace more that arange
        rats = np.linspace(1, nn - 1, nn - 1)[:, None] / nn
        return type(self)(
            np.concatenate([ps[:1], ps[:-1] * rats + ps[1:] * (1 - rats), ps[-1:]])
        ).elevated(to_degree)

    def derivative(self, derivative: int) -> BezierCurve:
        """nth derivative of a Bezier curve.

        :param derivative: 0 -> the curve itself, 1 -> 1st, 2 -> 2nd, etc.
        :return: points to calculate nth derivative.
        :raise ValueError: if derivative is greater than degree

        The derivative of a Bezier curve of degree n is a Bezier curve of degree
        n-1 with control points n*(p1-p0), n*(p2-p1), n*(p3-p2), ...
        """
        if derivative == 0:
            return self
        if derivative > self.degree:
            msg = par(
                f"""Bezier curve of degree {self.degree} does not have a
                {derivative}th derivative."""
            )
            raise ValueError(msg)
        points = (self.as_array[1:] - self.as_array[:-1]) * self.degree
        return type(self)(points).derivative(derivative - 1)
