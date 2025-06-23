"""Convert between control points and SVG path data.

:author: Shay Hill
:created: 2025-06-18
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Callable, TypeVar
import itertools as it

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

# Number of places after the decimal point to write numbers when converting from
# float values to svg path data string floats.
PRECISION = 6

_T = TypeVar("_T")


def _pairwise(iterable: Iterable[_T]) -> Iterator[tuple[_T, _T]]:
    """Return an iterator of pairs from an iterable.

    :param iterable: an iterable
    :return: an iterator of pairs
    """
    iter1, iter2 = it.tee(iterable)
    _ = next(iter2, None)
    return zip(iter1, iter2)


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
    as_str = f"{float(num):0.{PRECISION}f}".rstrip("0").rstrip(".")
    if as_str == "-0":
        as_str = "0"
    return as_str


# Match an svg path data string command or number.
COMMAND_OR_NUMBER = re.compile(r"[A-Za-z]|-?\d+\.\d+|-?\d+")


def _svgd_split(svgd: str) -> list[str]:
    """Split an svg data string into commands and numbers.

    :param svgd: An svg path element d string
    :return: a list of all commands (single letters) and numbers
    """
    return COMMAND_OR_NUMBER.findall(svgd)


def _svgd_join(*parts: str) -> str:
    """Join SVG path data parts.

    :param parts: parts of an SVG path data string
    :return: joined SVG path data string

    Svg datastrings don't need a lot of whitespace.
    """
    joined = " ".join(parts)
    joined = re.sub(r"\s+", " ", joined)
    joined = re.sub(r" -", "-", joined)
    return re.sub(r"\s*([A-Za-z])\s*", r"\1", joined)


def _new_svg_command_issuer() -> Callable[..., str]:
    """Format an SVG command without unnecessary repetition.

    :return: function that formats SVG commands
    """
    prev_cmd: str | None = None

    def issue_cmd(cmd: str, *pnts: str) -> str:
        """Format a command with points.

        :param cmd: command, e.g. "M", "L", "Q", "C"
        :param pnts: points for the command
        :return: formatted command
        """
        nonlocal prev_cmd
        cmd_ = cmd if cmd != prev_cmd else ""
        prev_cmd = cmd
        return _svgd_join(cmd_, *pnts)

    return issue_cmd


@dataclasses.dataclass
class _StrPoint:
    """A point with string representation."""

    x: str
    y: str

    def __init__(self, point: Sequence[float] | Sequence[str]) -> None:
        """Create a point with string representation."""
        self.x, self.y = map(_format_number, point)

    @property
    def xy(self) -> str:
        """Get the svg representation of the point.

        :return: x,y as a string
        """
        return _svgd_join(self.x, self.y)

    def __sub__(self, other: _StrPoint) -> _StrPoint:
        """Subtract two points.

        :param other: another point
        :return: a new point with the difference
        """
        return _StrPoint(
            (float(self.x) - float(other.x), float(self.y) - float(other.y))
        )

    def __add__(self, other: _StrPoint) -> _StrPoint:
        """Add two points.

        :param other: another point
        :return: a new point with the sum
        """
        return _StrPoint(
            (float(self.x) + float(other.x), float(self.y) + float(other.y))
        )


def _is_c1_continuous(curve_a: list[_StrPoint], curve_b: list[_StrPoint]) -> bool:
    """Check if two curves are C1 continuous.

    :param curve_a: first curve
    :param curve_b: second curve
    :return: True if the curves are C1 continuous, False otherwise
    """
    if curve_a[-1] != curve_b[0]:
        return False
    vec_a = curve_a[-1] - curve_a[-2]
    vec_b = curve_b[1] - curve_b[0]
    return vec_a == vec_b


def _is_c1_continuous2(curve_a: list[_StrPoint], curve_b: list[_StrPoint]) -> bool:
    """Check if two curves are C1 continuous.

    :param curve_a: first curve
    :param curve_b: second curve
    :return: True if the curves are C1 continuous, False otherwise
    """
    *_, pnt_a, pnt_b = curve_a
    pnt_c, *_ = curve_b
    return (pnt_b - pnt_a) == (pnt_c - pnt_b)


def _do_allow_curve_shorthand(cmd_a: _CmdPts, cmd_b: _CmdPts) -> bool:
    if cmd_a.cmd not in {"Q", "T", "C", "S"}:
        return False
    if cmd_b.cmd not in {"Q", "C"}:
        return False
    *_, pnt_a, pnt_b = cmd_a.pts
    pnt_c, *_ = cmd_b.pts
    if cmd_b.cmd == "Q" and cmd_a.cmd in {"Q", "T"}:
        return (pnt_b - pnt_a) == (pnt_c - pnt_b)
    if cmd_b.cmd == "C" and cmd_a.cmd in {"C", "S"}:
        return (pnt_b - pnt_a) == (pnt_c - pnt_b)
    return pnt_c == pnt_b


@dataclasses.dataclass
class _CmdPts:
    """A command with points."""

    cmd: str
    pts: list[_StrPoint]
    _prv: _CmdPts | None = dataclasses.field(default=None, init=False)
    _nxt: _CmdPts | None = dataclasses.field(default=None, init=False)

    @property
    def str_cmd(self) -> Iterator[str]:
        """Get the SVG command for this command."""
        if self._prv is None:
            yield self.cmd
            return
        if self.cmd == "L" and self._prv.cmd == "M":
            return
        if self.cmd != self._prv.cmd:
            yield self.cmd

    @property
    def nxt(self) -> _CmdPts | None:
        """Get the previous command."""
        return self._nxt

    @nxt.setter
    def nxt(self, value: _CmdPts | None) -> None:
        """Set the previous command."""
        if value is None:
            self._nxt = None
            return
        if value.cmd in "MLHV":
            if value.pts[0].x == self.pts[0].x:
                value.cmd = "V"
            elif value.pts[0].y == self.pts[0].y:
                value.cmd = "H"
        elif value.cmd == "T" and len(value.pts) == 1:
            if self.cmd in {"Q", "T"}:
                *_, pnt_a, pnt_b = self.pts
                value.pts.insert(0, pnt_b + (pnt_b - pnt_a))
            else:
                value.pts.insert(0, self.pts[-1])
        elif value.cmd == "S" and len(value.pts) == 2:
            if self.cmd in {"C", "S"}:
                *_, pnt_a, pnt_b = self.pts
                value.pts.insert(0, pnt_b + (pnt_b - pnt_a))
            else:
                value.pts.insert(0, self.pts[-1])
        elif value.cmd == "Q":
            value.cmd = "T" if _do_allow_curve_shorthand(self, value) else "Q"
        elif value.cmd == "C":
            value.cmd = "S" if _do_allow_curve_shorthand(self, value) else "C"
        value._prv = self
        self._nxt = value

    @property
    def str_pts(self) -> Iterator[str]:
        """Get the points that will be used in the SVG data."""
        if self.cmd == "H":
            yield self.pts[0].x
        elif self.cmd == "V":
            yield self.pts[0].y
        elif self.cmd in {"T", "S"}:
            yield from (p.xy for p in self.pts[1:])
        else:
            yield from (p.xy for p in self.pts)

    @property
    def str(self) -> Iterator[str]:
        """Get the SVG command and points for this command."""
        yield from self.str_cmd
        yield from self.str_pts


def cmd_pts_from_spline(spline: list[list[_StrPoint]]) -> _CmdPts:
    """Create a linked list of _CmdPts from a list of control points.

    :param spline: a list of curves, each curve is a list of _StrPoint
    :return: a linked list of _CmdPts
    """
    n2cmd = {2: "L", 3: "Q", 4: "C"}

    commanded = [
        _CmdPts("M", [spline[0][0]]),
        *(_CmdPts(n2cmd[len(c)], c[1:]) for c in spline),
    ]

    if spline[-1][-1] == spline[0][0]:
        if len(spline[-1]) == 2:
            commanded[-1] = _CmdPts("Z", [])
        else:
            commanded.append(_CmdPts("Z", []))

    for prev, this in _pairwise(commanded):
        prev.nxt = this

    return commanded[0]


def _yield_svgd_spline(cpts: list[list[_StrPoint]]) -> Iterator[str]:

    this_cmd: _CmdPts | None = cmd_pts_from_spline(cpts)
    while this_cmd is not None:
        yield from this_cmd.str
        this_cmd = this_cmd.nxt


def _yield_svgd(
    cpts: Iterable[Iterable[Sequence[float]]],
) -> Iterator[str]:
    """Determine the SVG command for each curve. Convert control points to _StrPoint.

    :param cpts: control points
        [
            [(x0, y0), (x1, y1)],  # linear Bezier curve
            [(x0, y0), (x1, y1), (x2, y2)],  # quadratic Bezier curve
            [(x0, y0), (x1, y1), (x2, y2), (x3, y3)],  # cubic Bezier curve
        ]
    :return: list of tuples with command and control points
        [
            ("M", [StrPoint(x0, y0)]),
            ("L", [StrPoint(x1, y1)]),
            ("Q", [StrPoint(x1, y1), StrPoint(x2, y2)]),
            ("C", [StrPoint(x1, y1), StrPoint(x2, y2), StrPoint(x3, y3)]),
        ]
    """
    cpts_sp = [[_StrPoint(p) for p in curve] for curve in cpts]
    spline: list[list[_StrPoint]] = []
    while cpts_sp:
        spline.append(cpts_sp.pop(0))
        if not cpts_sp:
            yield from _yield_svgd_spline(spline)
        elif spline[-1][-1] != cpts_sp[0][0]:
            yield from _yield_svgd_spline(spline)
            spline.clear()




def get_svgd_from_cpts(cpts: Iterable[Sequence[Sequence[float]]]) -> str:
    """Get an SVG path data string for a list of list of Bezier control points.

    :param cpts: control points
    :return: SVG path data string
    """
    return _svgd_join(*_yield_svgd(cpts))


def _pop_coordinate(svgd_parts: list[str]) -> tuple[float, float]:
    """Pop a coordinate from the SVG data parts.

    :param svgd_parts: SVG data parts
    :return: coordinate as a tuple of floats
    """
    x = float(svgd_parts.pop(0))
    y = float(svgd_parts.pop(0))
    return x, y


def _pop_coordinates(svgd_parts: list[str], num: int) -> Iterator[tuple[float, float]]:
    """Pop a number of coordinates from the SVG data parts.

    :param svgd_parts: SVG data parts
    :param num: number of coordinates to pop
    :return: list of coordinates
    """
    for _ in range(num):
        yield _pop_coordinate(svgd_parts)


def get_cpts_from_svgd(svgd: str) -> list[list[tuple[float, float]]]:
    """Get a list of lists (curves) of xy coordinates from SVG data.

    :param svgd: an ABSOLUTE SVG path data string
    :return: list of lists of xy coordinates
    [
        [(x0, y0), (x1, y1)],  # linear Bezier curve
        [(x0, y0), (x1, y1), (x2, y2)],  # quadratic Bezier curve
        [(x0, y0), (x1, y1), (x2, y2), (x3, y3)],  # cubic Bezier curve
    ]
    :raise ValueError: if the SVG data string contains arc commands does not start
        with "M"
    :raise NotImplementedError: if the SVG data string contains an unexpected command
    """
    if "a" in svgd.lower():
        msg = f"Arc commands cannot be converted to Bezier control points in {svgd}."
        raise ValueError(msg)
    parts = _svgd_split(svgd)
    if parts[0].lower() != "m":
        msg = f"Incomplete or invalid svg path element data string: {svgd}."
        raise ValueError(msg)
    cpts: list[list[tuple[float, float]]] = []
    cmd: str | None = None
    queued_pnt: tuple[float, float] | None = None
    while parts:
        if re.match("[A-Za-z]", parts[0]):
            cmd = parts.pop(0)
        if cmd and cmd in "mM":
            queued_pnt = _pop_coordinate(parts)
            cmd = "L"
            continue

        x0, y0 = queued_pnt or cpts[-1][-1]
        queued_pnt = None
        if cmd == "H":
            x1 = float(parts.pop(0))
            cpts.append([(x0, y0), (x1, y0)])
        elif cmd == "V":
            y1 = float(parts.pop(0))
            cpts.append([(x0, y0), (x0, y1)])
        elif cmd == "L":
            cpts.append([(x0, y0), _pop_coordinate(parts)])
        elif cmd == "T":
            x_delta = cpts[-1][-1][0] - cpts[-1][-2][0]
            y_delta = cpts[-1][-1][1] - cpts[-1][-2][1]
            cpts.append(
                [(x0, y0), (x0 + x_delta, y0 + y_delta), _pop_coordinate(parts)]
            )
        elif cmd == "Q":
            cpts.append([(x0, y0), *_pop_coordinates(parts, 2)])
        elif cmd == "S":
            x_delta = cpts[-1][-1][0] - cpts[-1][-2][0]
            y_delta = cpts[-1][-1][1] - cpts[-1][-2][1]
            cpts.append(
                [(x0, y0), (x0 + x_delta, y0 + y_delta), *_pop_coordinates(parts, 2)]
            )
        elif cmd == "C":
            cpts.append([(x0, y0), *_pop_coordinates(parts, 3)])
        elif cmd in {"Z", "z"}:
            if (x0, y0) != cpts[0][0]:
                cpts.append([(x0, y0), cpts[0][0]])
        else:
            msg = f"Unexpected command in svg path data string: {cmd} in {svgd}."
            raise NotImplementedError(msg)
    return cpts


if __name__ == "__main__":
    # TODO: remove this test code.
    aaa = "M0.5 0.5C1 0 2 0 2.5 0.5 3 1 3 2 2.5 2.5 2 3 1 3 0.5 2.5 0 2 0 1 0.5 0.5Z"
    bbb = get_cpts_from_svgd(aaa)
    ccc = get_svgd_from_cpts(bbb)
    ddd = get_cpts_from_svgd(ccc)
    print(aaa)
    print(ccc)
    print(bbb)
