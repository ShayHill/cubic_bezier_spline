#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Bezier curve objects.

:author: Shay Hill
:created: 1/18/2020

I have a lot of Bezier curve code, but most of it is mixed up with other spline
types, rational Bezier, etc., none of which (except perhaps rational Bezier) are
useful for SVG creation. Creating new Bezier functionality here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Optional, Sequence, Tuple

import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore

from .matrices import get_mix_matrix


@dataclass(frozen=True)
class BezierCurve:
    """
    A non-rational Bezier curve.
    """

    _points: NDArray[(Any, Any), float]
    degree: int

    def __init__(self, points: Sequence[Sequence[float]]) -> None:
        """
        Convert all points to ndarray.

        This allows for easy math and has the effect of ensuring no references exist
        in Bezier points.

        The `tuple` in `np.array(tuple(points))` allows a BezierCurve to be constructed
        from an iterable of iterables.
        """
        object.__setattr__(self, "_points", np.array(points, dtype=float))
        object.__setattr__(self, "degree", len(self._points) - 1)

    def __hash__(self) -> int:
        """To cache method calls"""
        return id(self)

    def __iter__(self):
        return iter(self._points)

    def __getitem__(self, item: int) -> NDArray[(Any,), float]:
        """
        Return item-th point

        :param item: index of [p0, p1, p2, p3]
        :return: One control point as an array
        """
        return self._points[item]

    def __array__(self) -> NDArray[(Any, Any), float]:
        """
        Return self._points (it's an array anyway)

        :return: self._points when encountering np.array(self)
        """
        return self._points

    def __call__(self, time: float, derivative: int = 0) -> NDArray[(Any,), float]:
        """
        Cubic Bezier calculation at time.

        :param time: time on curve (typically 0 - 1)
        :return: Non-rational Bezier at time
        """
        if derivative == 0:
            return self._get_tmat(time) @ self._mixed_points
        return self.derivative(derivative)(time)

    def _get_tmat(self, time) -> NDArray[(Any,), float]:
        # noinspection PyTypeChecker
        return np.array([1] + [time ** x for x in range(1, self.degree + 1)])

    def _get_zmat(self, time) -> NDArray[(Any, Any), float]:
        """ 2D zero matrix with tmat on the diagonal """
        # noinspection PyTypeChecker
        return np.diagflat(self._get_tmat(time))

    @cached_property
    def _mmat(self) -> NDArray[(Any, Any), float]:
        return get_mix_matrix(self.degree + 1)

    @cached_property
    def _mixed_points(self) -> NDArray[(Any, Any), float]:
        """
        Points scaled by binomial coefficients

        Scale this by time matrix to evaluate curve at time.
        """
        return self._mmat @ self._points

    def split(self, time: float) -> Tuple[BezierCurve, BezierCurve]:
        """
        Split a BezierCurve into two Bezier curves of the same degree.

        :param time: time at which to split the curve.
        :return: two new BezierCurve instances
        :raises: ValueError if not 0 <= time <= 1
        """
        qmat = np.linalg.inv(self._mmat) @ self._get_zmat(time) @ self._mmat
        qmat_prime = np.zeros_like(qmat)
        for i in range(qmat.shape[0]):
            j = i + 1
            qmat_prime[-j, -j:] = qmat[i, :j]
        return (
            type(self)(qmat @ self._points),
            type(self)(qmat_prime @ self._points),
        )

    def elevated(self, to_degree: Optional[int] = None) -> BezierCurve:
        """
        A new curve, elevated 1 or optionally more degrees.

        :param to_degree: final degree of Bezier curve
        :return: Bezier curve of identical shape with degree increased
        """
        if to_degree == self.degree:
            return self
        if to_degree is None:
            to_degree = self.degree + 1
        elif to_degree < self.degree:
            raise ValueError(
                "cannot elevate BezierCurve degree={self.degree} "
                "to BezierCurve degree={to_degree}"
            )
        nn, pp = len(self._points), self._points
        rats = np.arange(1, nn)[:, None] / nn
        return type(self)(
            np.concatenate([pp[:1], pp[:-1] * rats + pp[1:] * (1 - rats), pp[-1:]])
        ).elevated(to_degree)

    @lru_cache
    def derivative(self, derivative: int) -> BezierCurve:
        """
        nth derivative of a Bezier curve

        :param derivative: 0 -> the curve itself, 1 -> 1st, 2 -> 2nd, etc.
        :return: points to calculate nth derivative.

        The derivative of a Bezier curve of degree n is a Bezier curve of degree
        n-1 with control points n*(p1-p0), n*(p2-p1), n*(p3-p2), ...
        """
        if derivative == 0:
            return self
        if derivative > self.degree:
            raise ValueError(
                f"Bezier curve of degree {self.degree} "
                f"does not have a {derivative}th derivative."
            )
        points = (self._points[1:] - self._points[:-1]) * self.degree
        return type(self)(points).derivative(derivative - 1)
