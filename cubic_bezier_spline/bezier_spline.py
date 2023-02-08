"""Bezier splines (composite curves).

:author: Shay Hill
:created: 10/4/2020

A dead-simple container for lists of Bezier curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import TYPE_CHECKING

import numpy as np

from .bezier_curve import BezierCurve

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from .type_hints import FArray


class TimeIntervalError(Exception):
    """Time value out of range in BezierSpline.__call__."""


@dataclass
class BezierSpline:
    """A list of non-rational Bezier curves."""

    _curves: list[BezierCurve]

    def __init__(self, curves: Iterable[Sequence[Sequence[float]]]) -> None:
        """Create a spline from a list of Bezier curves."""
        self._curves = [BezierCurve(x) for x in curves]

    def __iter__(self) -> Iterator[BezierCurve]:
        """Iterate over the curves in the spline.

        :return: iterator over Bezier curves
        """
        return iter(self._curves)

    def __getitem__(self, item: int) -> BezierCurve:
        """Get a curve from the spline.

        :param item: index of curve
        :return: Bezier curve
        """
        return self._curves[item]

    def __len__(self) -> int:
        """Get the number of curves in the spline.

        :return: number of curves
        """
        return len(self._curves)

    def __array__(self) -> FArray:
        """Get the spline as a numpy array.

        :return: numpy array of curves
        """
        return np.array([np.array(x) for x in self._curves])

    def __call__(self, time: float, derivative: int = 0) -> FArray:
        """Given x.y, call curve x at time y.

        :param time: x.y -> curve index x and time on curve y
            between 0 and len(curves)
        :param derivative: optional derivative at time
        :return: xth non-rational Bezier at time
        :raise TimeIntervalError: if time is not in [0, len(curves)]

        For a spline with 3 curves, spline(3) will return curve 2 at time=1
        """
        if not 0 <= time <= len(self):
            msg = f"{time} not in interval [0, {len(self)}]"
            raise TimeIntervalError(msg)
        try:
            return self._curves[floor(time)](time % 1, derivative)
        except IndexError:
            return self._curves[floor(time) - 1](1, derivative)
