#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Convert svg paths to tuples of control points

:author: Shay Hill
:created: 1/11/2020

One quasi-public function: svg_to_bezier
TODO: update docstrings. No longer issues only cubic Bezier
"""


import re
from string import ascii_lowercase
from typing import Iterator, List, Optional, Tuple

import numpy as np
from nptyping import Array
from ..bezier import BezierCurve

Point = Array[float, 2]
Curve = Tuple[Point, ...]

def _split_d(svg_d_path: str) -> List[Tuple[str, str]]:
    """
    Split a path or glyph d argument into individual curve paths.

    :param svg_d_path: spline path from an svg path or glyph element
    :return: each line or curve as its own tuple
        (command, parameters)

        >>> _split_d('M1 2l-4 3h5')
        [('M', '1 2'), ('l', '-4 3'), ('h', '5')]

    Strip() the parameters, as there is an occasional trailing space in font files.
    """
    paths = re.split("([a-zA-Z])", svg_d_path)
    return [(paths[x], paths[x + 1].strip()) for x in range(1, len(paths), 2)]


class _PathConverter:
    def __init__(self):
        self._path_beg: Optional[Point] = None
        self._prev_curve = (np.array([0, 0]),)

    @property
    def _current_point(self):
        #TODO: I believe it's safe to factor out this .copy()
        return self._prev_curve[-1].copy()

    def __call__(self, svg_d_path: str) -> List[Curve]:
        """
        Translate a
        :param path:
        :return:
        """
        curves = []
        for command, points in _split_d(svg_d_path):
            if command in "hH":
                points = points.replace(" ", f" 0 ") + " 0"
            elif command in "vV":
                points = "0 " + points.replace(" ", " 0 ")
            points = list(np.fromstring(points, sep=' ').reshape((-1, 2)))
            points = self._relative_to_absolute(command, points)
            if command in "zZ":
                command = "L"
                if not np.array_equal(self._prev_curve[-1], self._path_beg):
                    points.append(self._path_beg)
            if command in "mM":
                command = "L"
                self._path_beg = points.pop(0)
                self._prev_curve = (self._path_beg,)
            curves += list(self.exec_command(command.upper(), points))
        return curves

    def _issue_curve(self, points: List[Point], degree: int) -> Curve:
        curve = [self._prev_curve[-1]] + [points.pop(0) for _ in range(degree)]
        self._prev_curve = BezierCurve(*curve)
        return self._prev_curve

    def _relative_to_absolute(self, command: str, points: List[Point]) -> List[Point]:
        """
        Convert relative points to absolute.

        :param command: svg d command from a path or glyph (e.g., V, C, s)
        :param points: parameters to svg command
        :return: points converted to absolute

        Lowercase svg path commands are relative to current point (last point prior
        to command), uppercase are absolute. Actually, it's not exactly that
        straightforward. Commands H and V are uppercase, but the y and x values
        (respectively) are relative, as they inherit from the current point.
        """
        if command in ascii_lowercase + "H":
            points = [x + (0, self._current_point[1]) for x in points]
        if command in ascii_lowercase + "V":
            points = [x + (self._current_point[0], 0) for x in points]
        return points

    def exec_command(self, command: str, points: List[Point]) -> Iterator[Curve]:
        """
        Groom arguments and call the appropriate _exec_ method

        :param command: svg d element command (e.g., M, m, L, l, C, c, ...)
        :param points: svg d element arguments converted to x, y np.arrays
        :return: yield Curve control-point tuples

        """
        if not points:
            return
        if command in "LHV":
            yield self._issue_curve(points, 1)
            yield from self.exec_command("L", points)
            return
        if command in "ST":
            first_point = self._current_point
            if self._prev_curve.degree == {"S": 3, "T": 2}[command]:
                first_point = self._prev_curve[-1] * 2 - self._prev_curve[-2]
            points.insert(0, first_point)
        if command in "QT":
            yield self._issue_curve(points, 2)
            yield from self.exec_command(command, points)
            return
        if command in "CS":
            """All points after transformation"""
            yield self._issue_curve(points, 3)
            yield from self.exec_command(command, points)
            return
        raise NotImplementedError(f"no provision for command {command}")


_CONVERT_PATH = _PathConverter()



def svg_to_bezier(svg_d_path: str) -> List[Curve]:
    """
    Convert the d argument from an svg path or glyph to a list of cubic Bezier points.

    :param svg_d_path: d argument from an svg path or glyph. E.g., 'M1 2l3 5t3 7'
    :return: a list of cubic Bezier control point tuples.
        [
            (endpoint_0, control_point_0, control_point_1, endpoint_1),
            (endpoint_0, control_point_0, control_point_1, endpoint_1),
            (endpoint_0, control_point_0, control_point_1, endpoint_1),
            (endpoint_0, control_point_0, control_point_1, endpoint_1)
        ]
    """
    return _CONVERT_PATH(svg_d_path)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
