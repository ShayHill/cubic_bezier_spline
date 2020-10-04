#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
""" Bezier splines (composite curves)

:author: Shay Hill
:created: 10/4/2020

A dead-simple container for lists of Bezier curves.
"""

from typing import List, Iterable, Iterator, Any
from .bezier_curve import BezierCurve
from dataclasses import dataclass
from nptyping import NDArray
from math import floor

# TODO: something better with Point type
Point = Any


class TimeIntervalError(Exception):
    """ Time value out of range in BezierSpline.__call__ """


@dataclass
class BezierSpline:
    """
    A list of non-rational Bezier curves.
    """

    _curves: List[BezierCurve]

    def __init__(self, curves: Iterable[Iterable[Point]]) -> None:
        self._curves = [BezierCurve(x) for x in curves]

    def __iter__(self) -> Iterator[BezierCurve]:
        return iter(self._curves)

    def __getitem__(self, item: int) -> BezierCurve:
        return self._curves[item]

    def __len__(self) -> int:
        return len(self._curves)

    def __call__(self, time: float, derivative: int = 0) -> NDArray[(Any,), float]:
        """
        Given x.y, call curve x and time y

        :param time: x.y -> curve index x and time on curve y
            between 0 and len(curves)
        :param derivative: optional derivative at time
        :return: xth non-rational Bezier at time
        """
        if not 0 <= time <= len(self):
            raise TimeIntervalError(f"{time} not in interval [0, {len(self)}]")
        try:
            return self._curves[floor(time)](time % 1, derivative)
        except IndexError:
            # time == len(self)
            return self._curves[floor(time)-1](1, derivative)
