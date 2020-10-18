#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
""" Bezier splines (composite curves)

:author: Shay Hill
:created: 10/4/2020

A dead-simple container for lists of Bezier curves.
"""

from dataclasses import dataclass
from math import floor
from typing import Any, Iterable, Iterator, List, Sequence

import numpy as np
from nptyping import NDArray

from .bezier_curve import BezierCurve


class TimeIntervalError(Exception):
    """ Time value out of range in BezierSpline.__call__ """


@dataclass
class BezierSpline:
    """
    A list of non-rational Bezier curves.
    """

    _curves: List[BezierCurve]

    def __init__(self, curves: Iterable[Sequence[Sequence[float]]]) -> None:
        self._curves = [BezierCurve(x) for x in curves]

    def __iter__(self) -> Iterator[BezierCurve]:
        return iter(self._curves)

    def __getitem__(self, item: int) -> BezierCurve:
        return self._curves[item]

    def __len__(self) -> int:
        return len(self._curves)

    def __array__(self) -> NDArray[(Any, Any, Any), float]:
        # noinspection PyTypeChecker
        return np.array([np.array(x) for x in self._curves])

    def __call__(self, time: float, derivative: int = 0) -> NDArray[(Any,), float]:
        """
        Given x.y, call curve x at time y.

        :param time: x.y -> curve index x and time on curve y
            between 0 and len(curves)
        :param derivative: optional derivative at time
        :return: xth non-rational Bezier at time

        For a spline with 3 curves, spline(3) will return curve 2 at time=1
        """
        if not 0 <= time <= len(self):
            raise TimeIntervalError(f"{time} not in interval [0, {len(self)}]")
        try:
            return self._curves[floor(time)](time % 1, derivative)
        except IndexError:
            # time == len(self)
            return self._curves[floor(time) - 1](1, derivative)
