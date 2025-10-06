"""Bezier splines (composite curves).

:author: Shay Hill
:created: 10/4/2020

A dead-simple container for lists of Bezier curves.
"""

from __future__ import annotations

import itertools as it
from dataclasses import dataclass
from functools import cached_property
from math import floor
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

import numpy as np
import numpy.typing as npt
from svg_path_data import get_svgd_from_cpts

from cubic_bezier_spline.bezier_curve import BezierCurve
from cubic_bezier_spline.control_point_casting import as_nested_tuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


_BezierSplineT = TypeVar("_BezierSplineT", bound="BezierSpline")


@dataclass(frozen=True)
class BezierSpline:
    """A sequence of non-rational Bezier curves."""

    cpts_: Iterable[Iterable[Iterable[float]]]

    def __post_init__(self) -> None:
        """Set cpts_ to cpts for __str___."""
        object.__setattr__(self, "cpts_", self.cpts)

    @cached_property
    def cpts(self) -> tuple[tuple[tuple[float, ...], ...], ...]:
        """Get the control points of the spline.

        :return: tuple of control points for each curve
        """
        return tuple(as_nested_tuple(x) for x in self.cpts_)

    @cached_property
    def curves(self) -> list[BezierCurve]:
        """Get the curves in the spline.

        :return: list of Bezier curves
        """
        return [BezierCurve(x) for x in self.cpts]

    def __iter__(self) -> Iterator[BezierCurve]:
        """Iterate over the curves in the spline.

        :return: iterator over Bezier curves
        """
        return iter(self.curves)

    def __getitem__(self, item: int) -> BezierCurve:
        """Get a curve from the spline.

        :param item: index of curve
        :return: Bezier curve
        """
        return self.curves[item]

    def __len__(self) -> int:
        """Get the number of curves in the spline.

        :return: number of curves
        """
        return len(self.curves)

    def __array__(self) -> Annotated[npt.NDArray[np.floating[Any]], (-1, -1, -1)]:
        """Get the spline as a numpy array.

        :return: numpy array of curves
        """
        return np.array([x.as_array for x in self.curves])

    @cached_property
    def is_closed(self) -> bool:
        """Check if the spline is closed.

        :return: True if the spline is closed, False otherwise
        """
        return self.cpts[0][0] == self.cpts[-1][-1]

    @cached_property
    def svgd(self) -> str:
        """Get the SVG data for the spline.

        :return: SVG data string (the d="" attribute of an svg "path" element)
        """
        return get_svgd_from_cpts(self.cpts, 6)

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

    @cached_property
    def reversed(self: _BezierSplineT) -> _BezierSplineT:
        """Reverse the spline.

        :return: reversed BezierSpline
        """
        return type(self)(x.reversed.cpts for x in reversed(self.curves))

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
            return self.curves

        beg_idx, beg_val = self._divmod_time(beg_time)
        end_idx, end_val = self._divmod_time(end_time)

        if beg_time >= end_time:
            if self.cpts[0][0] != self.cpts[-1][-1]:
                msg = "Cannot split an open spline from high to low time"
                raise ValueError(msg)
            curves: list[BezierCurve] = []
            if beg_time != len(self):
                curves.extend(self._split_to_curves(beg_time, len(self)))
            if end_time != 0:
                curves.extend(self._split_to_curves(0, end_time))
            return curves

        if beg_idx == end_idx:
            return self.curves[beg_idx].split(beg_val, end_val)[1:-1]

        head = [] if beg_val == 1 else self.curves[beg_idx].split(beg_val)[1:]
        body = self.curves[beg_idx + 1 : end_idx]
        tail = [] if end_val == 0 else self.curves[end_idx].split(end_val)[:1]
        return head + body + tail

    def split(
        self: _BezierSplineT,
        beg_time: float,
        end_time: float,
        *,
        normalized: bool | None = None,
        uniform: bool | None = None,
    ) -> _BezierSplineT:
        """Split a BezierSpline into two Bezier splines.

        :param beg_time: time at which to start the new spline
        :param end_time: time at which to end the new spline
        :param normalized: if True (default False), time is in [0, 1]
            instead of [0, len(curves)]
        :param uniform: if True (default), treat all curves as equal in length,
            else longer curves will take up more of the time interval.
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
        bool_kwargs = {"uniform": uniform, "normalized": normalized}
        beg_time = sum(self.divmod_time(beg_time, **bool_kwargs))
        end_time = sum(self.divmod_time(end_time, **bool_kwargs))
        curves = self._split_to_curves(beg_time, end_time)
        return type(self)([x.cpts for x in curves])

    @cached_property
    def lengths(self) -> list[float]:
        """Get the lengths of the curves in the spline.

        :return: list of curve lengths
        """
        return [x.length for x in self.curves]

    @cached_property
    def seams(self) -> list[float]:
        """Get the time value at each curve seam.

        :return: list of seam times
        """
        return [0, *it.accumulate(self.lengths)]

    def divmod_time(
        self,
        time: float,
        *,
        normalized: bool | None = None,
        uniform: bool | None = None,
    ) -> tuple[int, float]:
        """Return the curve index and time on curve for a given time value.

        :param time: time value
        :param normalized: if True (default False), time is in [0, 1]
            instead of [0, len(curves)]
        :param uniform: if True (default), treat all curves as equal in length,
            else longer curves will take up more of the time interval.
        :return: curve index, time on curve

        For the default uniform, non-normalized case, time n.t will return the
        evaluation of `self.curves[n](t)`.
        time = 2.5 -> curve 2 evaluated at time=0.5

        The uniform, normalized case will scale n.t in [0, 1] to [0, len(self)]
        then return the same `self.curves[n](t)`.
        time = 0.5 with 5 curves -> curve 2 evaluated at time=0.5

        The non-uniform, normalized case will scale time from n.t in [0, 1] to [0,
        spline_len] where spline_len is the sum of the lengths of all curves in the
        spline. n.t will be evaluated such that n.t lies on the curve in the time
        interval [<=n.t, >=n+1.t].

        The non-uniform, non-normalized case would require knowing the spline_len
        before passing time, but would find the interval without scaling time. I
        don't expect to ever need this case.
        """
        uniform = uniform if uniform in {True, False} else True
        normalized = normalized if normalized in {True, False} else not uniform

        total_length = len(self) if uniform else self.seams[-1]
        time = time * total_length if normalized else time

        if self.is_closed:
            time = time % total_length
        else:
            time = min(max(0, time), total_length)

        if uniform:
            if floor(time) == len(self):  # time == len(self)
                return len(self) - 1, 1
            return floor(time), time % 1

        curve_ix = _find_curve_index(self.seams, time)
        interval = self.seams[curve_ix : curve_ix + 2]
        return curve_ix, (time - interval[0]) / (interval[1] - interval[0])

    def __call__(
        self,
        time: float,
        derivative: int = 0,
        *,
        normalized: bool | None = None,
        uniform: bool | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Given x.y, call curve x at time y.

        :param time: x.y -> curve index x and time on curve y
            between 0 and len(curves)
        :param derivative: optional derivative at time
        :param normalized: if True (default False), time is in [0, 1]
            instead of [0, len(curves)]
        :param uniform: if True (default), treat all curves as equal in length,
            else longer curves will take up more of the time interval.
        :return: xth non-rational Bezier at time

        For a spline with 3 curves, spline(3) will return curve 2 at time=1
        """
        curve_idx, time = self.divmod_time(time, normalized=normalized, uniform=uniform)
        return self.curves[curve_idx](time, derivative)


def _find_curve_index(seams: Sequence[float], time: float) -> int:
    """Find the lowest gap where target could be inserted.

    :param seams: a list of sorted numbers representing the time intervals at which
        curves meet.
    :param time: the time value for which you are seeking an interval.
    :return: int, the index of the first curve where time value is on that curve.
        Time will like on the interval [seams[index], seams[index+1]]
    :raises ValueError: if time is out of bounds (this should not happen)
    """
    if seams[0] > time > seams[-1]:
        msg = "The time value is out of bounds of the seams."
        raise ValueError(msg)

    left, right = 0, len(seams) - 1
    result = 0  # for case where target is exactly sorted_values[0]
    while left <= right:
        mid = (left + right) // 2
        if seams[mid] < time:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    return result
