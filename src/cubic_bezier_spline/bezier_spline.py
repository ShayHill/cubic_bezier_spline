"""Bezier splines (composite curves).

:author: Shay Hill
:created: 10/4/2020

A dead-simple container for lists of Bezier curves.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from math import floor
from typing import TYPE_CHECKING, Annotated, Any, Callable, TypeVar, Union

import numpy as np
import numpy.typing as npt

from cubic_bezier_spline.bezier_curve import BezierCurve

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


Point = Union[Sequence[float], npt.NDArray[np.floating[Any]]]
Points = Union[Sequence[Sequence[float]], npt.NDArray[np.floating[Any]]]


def _svg_d_join(*parts: str) -> str:
    """Join SVG path data parts.

    :param parts: parts of an SVG path data string
    :return: joined SVG path data string

    Svg datastrings don't need a lot of whitespace.
    """
    joined = " ".join(parts)
    joined = re.sub(r"\s+", " ", joined)
    joined = re.sub(r" -", "-", joined)
    return re.sub(r"\s*([A-Za-z])\s*", r"\1", joined)


class TimeIntervalError(Exception):
    """Time value out of range in BezierSpline.__call__."""


def _format_number(num: float | str) -> str:
    """Format strings at limited precision.

    :param num: anything that can print as a float.
    :return: str

    I've read articles that recommend no more than four digits before and two digits
    after the decimal point to ensure good svg rendering. I'm being generous and
    giving six. Mostly to eliminate exponential notation, but I'm "rstripping" the
    strings to reduce filesize and increase readability

    * reduce fp precision to 6 digits
    * remove trailing zeros
    * remove trailing decimal point
    * convert "-0" to "0"
    """
    as_str = f"{float(num):0.6f}".rstrip("0").rstrip(".")
    if as_str == "-0":
        as_str = "0"
    return as_str


@dataclasses.dataclass
class _StrPoint:
    """A point with string representation."""

    x: str
    y: str

    def __init__(self, point: Sequence[float]) -> None:
        """Create a point with string representation."""
        self.x, self.y = map(_format_number, point)

    @property
    def xy(self) -> str:
        """Get the svg representation of the point.

        :return: x,y as a string
        """
        return _svg_d_join(self.x, self.y)


def _new_svg_command_issuer() -> Callable[..., str]:
    """Format an SVG command without unnecessary repetition.

    :return: function that formats SVG commands
    """
    prev_cmd: str | None = None

    def issue_cmd(cmd: str, *pnts: str) -> str:
        """Format a command with points.

        :param cmd: command, e.g. "M", "L", "C"
        :param pnts: points for the command
        :return: formatted command
        """
        nonlocal prev_cmd
        cmd_ = cmd if cmd != prev_cmd else ""
        prev_cmd = cmd
        return _svg_d_join(cmd_, *pnts)

    return issue_cmd


_BezierSplineT = TypeVar("_BezierSplineT", bound="BezierSpline")


@dataclass
class BezierSpline:
    """A list of non-rational Bezier curves."""

    _curves: list[BezierCurve]

    def __init__(self, curves: Iterable[Sequence[Sequence[float]]]) -> None:
        """Create a spline from a list of Bezier curves."""
        self._curves = [BezierCurve(x) for x in curves]
        self.control_points = tuple(x.control_points for x in self._curves)

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

    def __array__(self) -> Annotated[npt.NDArray[np.floating[Any]], (-1, -1, -1)]:
        """Get the spline as a numpy array.

        :return: numpy array of curves
        """
        return np.array([x.as_array for x in self._curves])

    def _yield_svg_commands(self) -> Iterator[str]:
        """Get the SVG data for the spline.

        :return: SVG data
        :raise NotImplementedError: if the number of control points is not 1 or 3
        """
        beg_path: _StrPoint | None = None
        prev_pnt: _StrPoint | None = None

        issue_cmd = _new_svg_command_issuer()

        for curve in self._curves:
            pnt, *pnts = map(_StrPoint, curve.control_points)
            if prev_pnt != pnt:
                if pnt == beg_path:
                    yield issue_cmd("Z")
                yield issue_cmd("M", pnt.xy)
                beg_path = pnt
            if len(pnts) == 1 and pnts[0] == beg_path:
                # linear spline closing the path
                yield issue_cmd("Z")
            elif len(pnts) == 1 and pnts[0].x == pnt.x:
                yield issue_cmd("V", pnts[0].y)
            elif len(pnts) == 1 and pnts[0].y == pnt.y:
                yield issue_cmd("H", pnts[0].x)
            elif len(pnts) == 1:
                yield issue_cmd("L", pnts[0].xy)
            elif len(pnts) == 3:
                yield issue_cmd("C", *(p.xy for p in pnts))
            else:
                msg = f"Unexpected number of control points: {len(pnts)}"
                raise NotImplementedError(msg)
            prev_pnt = pnts[-1]
        if prev_pnt == beg_path:
            yield issue_cmd("Z")

    @property
    def as_array(self) -> Annotated[npt.NDArray[np.floating[Any]], (-1, -1, -1)]:
        """Get the spline as a numpy array.

        :return: numpy array of curves
        """
        return self.__array__()

    @cached_property
    def svg_data(self) -> str:
        """Get the SVG data for the spline.

        :return: SVG data string (the d="" attribute of an svg "path" element)
        """
        return "".join(self._yield_svg_commands())

    def _divmod_time(self, time: float) -> tuple[int, float]:
        """Divmod a time value into curve index and time on curve.

        :param time: time value
        :return: curve index, time on curve
        """
        time = min(max(0, time), len(self))
        floor_, fractional = int(time), time % 1
        if floor_ == len(self):
            return floor_ - 1, 1
        return floor_, fractional

    def reversed(self: _BezierSplineT) -> _BezierSplineT:
        """Reverse the spline.

        :return: reversed BezierSpline
        """
        return type(self)(x.reversed().control_points for x in reversed(self._curves))

    def _split_to_curves(self, beg_time: float, end_time: float) -> list[BezierCurve]:
        """Split a BezierSpline into multiple Bezier curves.

        :param beg_time: time at which to start the new spline
        :param end_time: time at which to end the new spline
        :return: list of Bezier curves
        :raises ValueError: if the spline is open and the times are reversed
        """
        beg_time = min(max(0, beg_time), len(self))
        end_time = min(max(0, end_time), len(self))

        if beg_time in {0, len(self)} and end_time in {0, len(self)}:
            return self._curves

        beg_idx, beg_val = self._divmod_time(beg_time)
        end_idx, end_val = self._divmod_time(end_time)

        if beg_time >= end_time:
            if self.control_points[0][0] != self.control_points[-1][-1]:
                msg = "Cannot split an open spline from high to low time"
                raise ValueError(msg)
            curves: list[BezierCurve] = []
            if beg_time != len(self):
                curves.extend(self._split_to_curves(beg_time, len(self)))
            if end_time != 0:
                curves.extend(self._split_to_curves(0, end_time))
            return curves

        if beg_idx == end_idx:
            return self._curves[beg_idx].split(beg_val, end_val)[1:-1]

        head = [] if beg_val == 1 else self._curves[beg_idx].split(beg_val)[1:]
        body = self._curves[beg_idx + 1 : end_idx]
        tail = [] if end_val == 0 else self._curves[end_idx].split(end_val)[:1]
        return head + body + tail

    def split(self: _BezierSplineT, beg_time: float, end_time: float) -> _BezierSplineT:
        """Split a BezierSpline into two Bezier splines.

        :param beg_time: time at which to start the new spline
        :param end_time: time at which to end the new spline
        :return: new BezierSpline

        A split BezierSpline will be another spline with some number <, =, or 1
        greater than the number of curves of the parent spline. The curves at the
        beginning of the spline may be very short, even 0 dimensional, so a split
        BezierSpline won't necessarily be useful for plotting or additional
        splitting, but it may be useful for drawing.

        If the time values are equal, or the end time is less than the begin time,
        this method will assume the spline is closed, and return a spline from begin
        to end *through* spline(0).
        """
        curves = self._split_to_curves(beg_time, end_time)
        return type(self)([x.control_points for x in curves])

    def __call__(
        self,
        time: float,
        derivative: int = 0,
        *,
        normalized_time_interval: bool = False,
    ) -> Point:
        """Given x.y, call curve x at time y.

        :param time: x.y -> curve index x and time on curve y
            between 0 and len(curves)
        :param derivative: optional derivative at time
        :param normalized_time_interval: if True, time is in [0, 1]
        :return: xth non-rational Bezier at time
        :raise TimeIntervalError: if time is not in [0, len(curves)]

        For a spline with 3 curves, spline(3) will return curve 2 at time=1
        """
        if normalized_time_interval:
            time = time * len(self)
        if not 0 <= time <= len(self):
            msg = f"{time} not in interval [0, {len(self)}]"
            raise TimeIntervalError(msg)
        try:
            return self._curves[floor(time)](time % 1, derivative)
        except IndexError:
            return self._curves[floor(time) - 1](1, derivative)
