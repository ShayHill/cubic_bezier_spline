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
_COMMAND_OR_NUMBER = re.compile(r"[A-Za-z]|-?\d+\.\d+|-?\d+")


def _svgd_split(svgd: str) -> list[str]:
    """Split an svg data string into commands and numbers.

    :param svgd: An svg path element d string
    :return: a list of all commands (single letters) and numbers
    """
    return _COMMAND_OR_NUMBER.findall(svgd)


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


def _do_use_curve_shorthand(cmd_a: _CmdPts, cmd_b: _CmdPts) -> bool:
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
        if value.cmd in "MLHV" and self.cmd != "Z":
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
            value.cmd = "T" if _do_use_curve_shorthand(self, value) else "Q"
        elif value.cmd == "C":
            value.cmd = "S" if _do_use_curve_shorthand(self, value) else "C"
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


def cmd_pts_from_string(svgd: str) -> list[_CmdPts]:
    """Create a linked list of _CmdPts from an SVG path data string.

    :param svgd: an ABSOLUTE SVG path data string
    :return: a linked list of _CmdPts
    """
    if "a" in svgd.lower():
        msg = f"Arc commands cannot be converted to Bezier control points in {svgd}."
        raise ValueError(msg)

    parts = _svgd_split(svgd)
    cmd2len = {"m": 2, "l": 2, "h": 1, "v": 1, "q": 4, "t": 2, "c": 6, "s": 4, "z": 0}
    commanded: list[tuple[str, list[str]]] = []

    cmd = parts.pop(0).upper()
    while parts:
        if parts[0].lower() in cmd2len.keys():
            cmd = parts.pop(0)
        num = cmd2len[cmd.lower()]
        commanded.append((cmd, [parts.pop(0) for _ in range(num)]))

    for prev, this in _pairwise(commanded):
        if this[0].lower() == "v":
            this[1].insert(0, prev[1][-2])
        elif this[0].lower() == "h":
            this[1].append(prev[1][-1])

    cmd_pts = [_CmdPts(x[0], [_StrPoint(x) for x in chunk_pairs(x[1])]) for x in commanded]

    for prev, this in _pairwise(cmd_pts):
        prev.nxt = this
        if this.cmd == this.cmd.lower():
            this.cmd = this.cmd.upper()
            this.pts = [x + prev.pts[-1] for x in this.pts]

    return cmd_pts


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

# def yield_cpts(svgd: str) -> Iterator[list[_StrPoint]]:



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

def chunk_pairs(items: Sequence[_T]) -> Iterator[tuple[_T, _T]]:
    if len(items) % 2 != 0:
        msg = f"Expected an even number of items, got {len(items)}."
        raise ValueError(msg)
    for i in range(0, len(items), 2):
        yield (items[i], items[i+1])

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
    cmd_pts = cmd_pts_from_string(svgd)
    lists: list[list[_StrPoint]] = []
    beg_idx = 0
    for cmd in cmd_pts:
        if cmd.cmd == "M":
            beg_idx = len(lists)
            continue
        if cmd.cmd == "Z":
            beg = lists[beg_idx][0]
            end = lists[-1][-1]
            if beg != end:
                lists.append([end, beg])
            continue
        lists.append([cmd._prv.pts[-1], *cmd.pts])

    return [[(float(p.x), float(p.y)) for p in curve] for curve in lists]   



from paragraphs import par
if __name__ == "__main__":
    potrace_output = par(
        """M338 236 c-5 -3 -6 -6 -3 -6 1 -1 2 -2 2 -3 0 -2 1 -2 2 -2 2 0 3 0 4 -1 2
        -2 2 -2 4 -1 1 2 2 2 3 1 2 -3 6 0 6 6 1 8 -4 9 -11 3 l-3 -3 0 4 c0 3 -1 4 -4
        2z M170 235 c-2 0 -5 -1 -5 -1 -1 -1 -3 -1 -4 -1 -3 0 -13 -5 -14 -6 -1 -1 -2
        -2 -4 -2 -3 0 -6 -2 -4 -3 1 -1 1 -1 0 -1 -1 -1 -1 -1 -1 0 0 1 -1 1 -1 1 -2 0
        -5 -4 -4 -5 0 -1 -1 -1 -2 -2 -1 0 -4 -3 -8 -6 -4 -4 -9 -8 -11 -9 -6 -5 -15
        -14 -14 -15 1 -1 0 -1 -2 -2 -4 0 -8 -4 -11 -10 -4 -7 -1 -6 3 1 2 4 3 5 2 3 0
        -2 -1 -4 -2 -5 -1 0 -1 -1 -1 -1 1 -1 5 1 5 2 0 1 0 1 1 1 1 0 1 0 1 -1 -2 -2 2
        -8 4 -8 0 1 2 1 2 1 1 0 1 1 1 1 0 1 2 4 4 7 5 6 5 6 -2 7 l-4 1 5 0 c4 -1 5 0
        7 2 2 2 4 3 4 3 1 0 0 -1 -2 -3 -3 -3 -3 -3 -1 -5 1 -1 1 -1 0 -1 -2 1 -11 -10
        -9 -12 2 -3 6 -2 9 3 3 2 5 4 6 3 1 0 0 -1 -3 -3 -6 -5 -8 -8 -6 -10 2 -1 3 -1
        4 2 3 6 9 9 12 6 2 -1 6 -2 6 0 0 1 -6 6 -7 6 -3 0 2 5 7 8 3 1 4 6 3 9 -1 1 8
        5 11 5 1 0 0 -1 -2 -2 -7 -2 -11 -9 -7 -10 4 -2 12 5 12 10 0 2 0 2 1 1 0 -1 1
        -2 0 -3 0 -1 0 -1 1 0 2 1 1 4 -2 5 -2 0 -2 0 0 1 1 1 3 3 4 4 0 1 1 3 2 3 0 0
        1 0 2 0 0 1 0 1 -1 1 0 -1 -1 -1 -1 0 0 0 2 1 4 2 2 1 4 3 4 3 0 1 0 1 1 0 2 -1
        8 2 8 4 0 1 2 3 4 4 2 1 4 2 4 2 0 -1 -1 -2 -3 -3 -2 0 -3 -1 -3 -2 1 0 0 -2 -2
        -3 -3 -2 -2 -4 2 -2 4 3 5 2 1 0 -4 -3 -10 -9 -9 -9 0 0 1 1 3 1 1 1 3 2 4 2 2
        0 4 1 6 4 3 3 5 4 5 3 1 -1 2 0 4 1 l2 3 -2 -3 c-1 -2 -2 -3 -3 -2 -2 0 -9 -6
        -9 -8 1 -3 4 -2 7 1 2 2 4 3 4 2 1 -1 1 -1 1 0 1 0 2 1 2 0 2 0 17 13 17 14 -1
        1 6 5 8 5 2 1 10 3 12 4 3 1 5 1 5 0 0 -1 2 -2 6 -3 3 -1 8 -3 10 -5 3 -2 5 -3
        6 -3 1 0 1 -1 1 -1 0 -1 1 -2 1 -3 1 0 3 -4 5 -8 2 -4 4 -7 5 -7 0 0 1 -1 2 -2
        0 -2 1 -2 1 -1 1 1 0 2 -2 5 -1 2 -2 3 -1 2 1 -1 2 -1 2 0 0 0 1 1 1 1 1 0 1 0
        1 1 0 2 1 2 2 1 3 -2 4 0 1 2 -1 1 -2 3 -2 3 0 1 -1 2 -1 3 -2 0 -2 3 0 3 2 1 2
        1 1 -1 0 -3 5 -10 9 -11 1 0 2 1 1 1 0 0 1 1 2 2 0 1 1 2 1 3 0 2 3 3 16 3 5 1
        6 1 4 0 -12 0 -14 -1 -14 -3 1 -3 4 -5 6 -3 1 1 4 1 6 2 1 0 4 0 5 1 1 0 2 0 2
        -1 0 -1 0 -2 -1 -2 -1 0 -1 0 -1 -1 0 -1 1 0 2 1 2 1 3 2 2 2 0 1 1 1 2 0 2 -1
        2 -1 0 -3 -2 -1 -3 -4 -1 -3 0 1 2 0 3 -1 2 -1 3 -1 3 0 0 1 1 1 3 1 4 0 5 1 2
        3 -2 1 -2 1 0 1 2 0 3 0 4 1 0 1 1 1 1 1 1 0 0 -1 -1 -2 -1 -2 -1 -2 0 -3 1 -1
        1 -1 2 0 1 1 1 1 2 0 2 -2 5 0 5 3 -1 4 0 6 1 4 1 -1 1 -1 1 1 0 2 -1 3 -1 2 -1
        0 -1 2 -2 4 0 2 -1 3 -2 3 0 -1 -1 0 -2 1 -1 0 -1 1 -1 1 1 0 0 1 0 3 -2 3 -5 4
        -5 2 0 -1 -1 -1 -1 1 0 1 -1 1 -1 0 0 -1 0 -1 -2 0 -1 2 -4 2 -17 2 -8 0 -15 0
        -16 1 -2 0 -15 -3 -19 -4 -2 -2 -3 -1 -8 0 -4 1 -7 2 -8 1 -1 0 -2 0 -2 1 0 1
        -6 3 -8 3 -1 0 -2 0 -2 1 -1 1 -1 1 -1 0 0 -1 -2 -1 -11 0 -2 0 -2 0 1 1 3 1 2
        1 -2 1 -4 -1 -7 -1 -7 -2 0 -1 -1 -1 -2 0 -1 1 -2 1 -3 0 -2 -2 -5 -3 -3 -1 0 1
        -4 1 -9 -1 -3 -1 -5 -1 -5 0 -1 1 -3 1 -5 1 -2 0 -6 1 -9 2 -4 0 -7 1 -8 1 -1 0
        -4 -1 -6 -1z"""
    )
    # TODO: remove this test code.
    aaa = "M0.5 0.5C1 0 2 0 2.5 0.5 3 1 3 2 2.5 2.5 2 3 1 3 0.5 2.5 0 2 0 1 0.5 0.5Z"
    aaa = potrace_output
    bbb = get_cpts_from_svgd(aaa)
    ccc = get_svgd_from_cpts(bbb)
    ddd = get_cpts_from_svgd(ccc)
    eee = get_svgd_from_cpts(ddd)
    limit = 75
    print(aaa[:limit])
    print(ccc[:limit])
    print(eee[:limit])
