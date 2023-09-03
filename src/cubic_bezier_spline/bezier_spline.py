"""Bezier splines (composite curves).

:author: Shay Hill
:created: 10/4/2020

A dead-simple container for lists of Bezier curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import floor
from typing import TYPE_CHECKING, Annotated

import numpy as np

from .bezier_curve import BezierCurve

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import numpy.typing as npt

    from .type_hints import Point


class TimeIntervalError(Exception):
    """Time value out of range in BezierSpline.__call__."""


def _get_tuple_strings(tuples: Iterable[Sequence[float]]) -> list[str]:
    """Get a list of strings for a list of tuples.

    :param tuples: list of tuples
    :param precision: number of digits after decimal point
    :return: list of strings

    This limits the precision of float strings to 6 digits, which is appropriate for
    svg.
    """
    return [f"{','.join(f'{x:.6f}' for x in t)}" for t in tuples]


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

    def __array__(self) -> Annotated[npt.NDArray[np.float_], (-1, -1, -1)]:
        """Get the spline as a numpy array.

        :return: numpy array of curves
        """
        return np.array([np.array(x) for x in self._curves])

    def _yield_svg_commands(self) -> Iterator[str]:
        """Get the SVG data for the spline.

        :return: SVG data
        """
        prev_pnt = ""
        for curve in self._curves:
            pnt_0, pnt_1, pnt_2, pnt_3 = _get_tuple_strings(curve.control_points)
            if prev_pnt != pnt_0:
                yield f"M {pnt_0}"
            yield f"C {pnt_1} {pnt_2} {pnt_3}"
            prev_pnt = pnt_3

    @cached_property
    def svg_data(self) -> str:
        """Get the SVG data for the spline.

        :return: SVG data string (the d="" attribute of an svg "path" element)
        """
        return " ".join(self._yield_svg_commands())

    def __call__(self, time: float, derivative: int = 0) -> Point:
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
