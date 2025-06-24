"""Convert between control points and SVG path data.

Splines in this library are represented as lists of, at a miminum, c0 continuous
Bezier curves, with each curve represented as a list of control points.

When starting from such a list, the svg data string representation will start with an
"M" and perhaps end with a "Z" if the last point is the same as the first. When
working from the other direction, SVG data strings are descriptions of (mostly)
Bezier curves, but they are frequently *not* continuous. There may be several "M" and
"Z" commands in one data string. If you convert this to control points, they may not
function well at a spline, because evaluating the spline at a discontinuous point
would have two possible values.

The functions you may need:

`get_svgd_from_cpts(cpts: Iterable[Sequence[Sequence[float]]]) -> str`
    - Convert a list of lists of Bezier control points to an SVG path data string.

`get_cpts_from_svgd(svgd: str) -> list[list[tuple[float, float]]`
    - Convert an SVG path data string to a list of lists of Bezier control points.

`make_relative(svgd: str) -> str`
    - Convert an absolute SVG path data string to a relative one.

`make_absolute(svgd: str) -> str`
    - Convert a relative SVG path data string to an absolute one.

:author: Shay Hill
:created: 2025-06-18
"""

from __future__ import annotations

import dataclasses
import itertools as it
import re
from typing import TYPE_CHECKING, TypeVar

from cubic_bezier_spline.pairwise import pairwise

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

# Number of places after the decimal point to write numbers when converting from
# float values to svg path data string floats.
PRECISION = 6

_T = TypeVar("_T")


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
    """A point with string representation.

    Stay in this format to compare "floats" by their less-precise string
    representations. This improves consistency when moving between formats.
    """

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
    """Determine if the curve shorthand can be used.

    :param cmd_a: the previous command
    :param cmd_b: the current command
    :return: True if the curve shorthand can be used, False otherwise
    """
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
    """A command with points.

    The nxt setter of this class does most of the work of identifying
    character-saving steps to create an SVG path data string.

    The str properties strip out unnecessary commands and points.
    """

    cmd: str
    pts: list[_StrPoint]
    prv: _CmdPts | None = dataclasses.field(default=None, init=False)
    _nxt: _CmdPts | None = dataclasses.field(default=None, init=False)

    @property
    def str_cmd(self) -> Iterator[str]:
        """Get the SVG command for this command.

        :return: None
        :yield: the SVG command (e.g. "M", "L", "Q", "C") if it differs from the
            previous command
        """
        if self.prv is None:
            yield self.cmd
            return
        if self.cmd == "L" and self.prv.cmd == "M":
            return
        if self.cmd != self.prv.cmd:
            yield self.cmd

    @property
    def nxt(self) -> _CmdPts | None:
        """Get the previous command.

        :return: the next command or None if this is the last command
        """
        return self._nxt

    @nxt.setter
    def nxt(self, value: _CmdPts | None) -> None:
        """Set the previous command.

        :param value: the next instance or None if this is the last command
        """
        if value is None:
            self._nxt = None
            return
        if value.cmd in "LHV" and self.cmd != "Z":
            if value.pts[0].x == self.pts[0].x:
                value.cmd = "V"
            elif value.pts[0].y == self.pts[0].y:
                value.cmd = "H"
        elif value.cmd in "tT" and len(value.pts) == 1:
            if self.cmd in "qQtT":
                *_, pnt_a, pnt_b = self.pts
                start = pnt_b if value.cmd == "T" else _StrPoint((0, 0))
                value.pts.insert(0, start + (pnt_b - pnt_a))
            else:
                value.pts.insert(0, self.pts[-1])
        elif value.cmd in "sS" and len(value.pts) == 2:
            if self.cmd in "cCsS":
                *_, pnt_a, pnt_b = self.pts
                start = pnt_b if value.cmd == "S" else _StrPoint((0, 0))
                value.pts.insert(0, start + (pnt_b - pnt_a))
            else:
                value.pts.insert(0, self.pts[-1])

        elif value.cmd == "Q":
            value.cmd = "T" if _do_use_curve_shorthand(self, value) else "Q"
        elif value.cmd == "C":
            value.cmd = "S" if _do_use_curve_shorthand(self, value) else "C"
        value.prv = self
        self._nxt = value

    @property
    def str_pts(self) -> Iterator[str]:
        """Get the points that will be used in the SVG data.

        :return: None
        :yield: the points as strings (e.g. "x y")
        """
        if self.cmd in "hH":
            yield self.pts[0].x
        elif self.cmd in "vV":
            yield self.pts[0].y
        elif self.cmd in "tTsS":
            yield from (p.xy for p in self.pts[1:])
        else:
            yield from (p.xy for p in self.pts)

    @property
    def str(self) -> Iterator[str]:
        """Get the SVG command and points for this command.

        :return: None
        :yield: the SVG command and points as strings
        """
        yield from self.str_cmd
        yield from self.str_pts


def _iter_cmds(cmd: _CmdPts, *, rev: bool = False) -> Iterator[_CmdPts]:
    """Iterate over commands in a linked list.

    :param cmd: the first command
    :param rev: if True, iterate in reverse order
    :return: an iterator over commands
    """
    if rev:
        while cmd.nxt is not None:
            cmd = cmd.nxt
    else:
        while cmd.prv is not None:
            cmd = cmd.prv

    cmd_: _CmdPts | None = cmd
    while cmd_ is not None:
        yield cmd_
        cmd_ = cmd_.prv if rev else cmd_.nxt


def _cmd_pts_from_spline(spline: list[list[_StrPoint]]) -> _CmdPts:
    """Create a linked list of _CmdPts from a list of control points.

    :param spline: a list of curves, each curve is a list of _StrPoint. These must be
        at least c0 continuous.
    :return: a linked list of _CmdPts
    """
    n2cmd = {2: "L", 3: "Q", 4: "C"}

    cmds = [
        _CmdPts("M", [spline[0][0]]),
        *(_CmdPts(n2cmd[len(c)], c[1:]) for c in spline),
    ]

    if spline[-1][-1] == spline[0][0]:
        if len(spline[-1]) == 2:
            cmds[-1] = _CmdPts("Z", [])
        else:
            cmds.append(_CmdPts("Z", []))

    for prev, this in pairwise(cmds):
        prev.nxt = this

    return cmds[0]


def _cmd_pts_from_string(svgd: str) -> _CmdPts:
    """Create a linked list of _CmdPts from an SVG path data string.

    :param svgd: an ABSOLUTE SVG path data string
    :return: a linked list of _CmdPts
    :raises ValueError: if the SVG data string contains arc commands
    """
    if "a" in svgd.lower():
        msg = f"Arc commands cannot be converted to Bezier control points in {svgd}."
        raise ValueError(msg)

    parts = _svgd_split(svgd)  # e.g., ["M", "0", "0", "H", "1", "V", "2"]
    cmd2len = {"m": 2, "l": 2, "h": 1, "v": 1, "q": 4, "t": 2, "c": 6, "s": 4, "z": 0}

    str_groups: list[list[str]] = []  # e.g., [["M", "0", "0"], ["H", "1"]]
    cmd = parts.pop(0).upper()
    while parts:
        if parts[0].lower() in cmd2len:
            cmd = parts.pop(0)
        num = cmd2len[cmd.lower()]
        str_groups.append([cmd, *(parts.pop(0) for _ in range(num))])

    # Fill in inferred values for H and V commands.
    for prev, this in pairwise(str_groups):
        if this[0] == "V":
            this.insert(1, prev[-2])
        elif this[0] == "H":
            this.append(prev[-1])
        elif this[0] == "v":
            this.insert(1, "0")
        elif this[0] == "h":
            this.append("0")

    cmd_pts = [
        _CmdPts(x, list(map(_StrPoint, _chunk_pairs(xs)))) for x, *xs in str_groups
    ]

    for prev_cmd, this_cmd in pairwise(cmd_pts):
        prev_cmd.nxt = this_cmd
        if this_cmd.cmd == this_cmd.cmd.lower():
            this_cmd.cmd = this_cmd.cmd.upper()
            this_cmd.pts = [x + prev_cmd.pts[-1] for x in this_cmd.pts]

    return cmd_pts[0]


def _yield_svgd_spline_cmds(cpts: list[list[_StrPoint]]) -> Iterator[str]:
    """Yield SVG commands and points from a list of control points.

    :param cpts: a list of curves, each curve is a list of _StrPoint
    :return: None
    :yield: SVG commands and points
        "Mx0 y0",
        "Lx1 y1",
        "Qx1 y1 x2 y2",
        "Cx1 y1 x2 y2 x3 y3",

    A subroutine of _yield_svgd.
    """
    cmd = _cmd_pts_from_spline(cpts)
    yield from it.chain.from_iterable(x.str for x in _iter_cmds(cmd))


def _yield_svgd_cmds(cpts: Iterable[Iterable[Sequence[float]]]) -> Iterator[str]:
    """Determine the SVG command for each curve. Convert control points to _StrPoint.

    :param cpts: control points
        [
            [(x0, y0), (x1, y1)],  # linear Bezier curve
            [(x0, y0), (x1, y1), (x2, y2)],  # quadratic Bezier curve
            [(x0, y0), (x1, y1), (x2, y2), (x3, y3)],  # cubic Bezier curve
        ]
    :return: None
    :yield: SVG commands and points
        "Mx0 y0",
        "Lx1 y1",
        "Qx1 y1 x2 y2",
        "Cx1 y1 x2 y2 x3 y3",

    Split the curves into continuous splines and pass each spline to
    yield_svgd_spline.
    """
    cpts_sp = [[_StrPoint(p) for p in curve] for curve in cpts]
    spline: list[list[_StrPoint]] = []
    while cpts_sp:
        spline.append(cpts_sp.pop(0))
        if not cpts_sp:
            yield from _yield_svgd_spline_cmds(spline)
        elif spline[-1][-1] != cpts_sp[0][0]:
            yield from _yield_svgd_spline_cmds(spline)
            spline.clear()


def get_svgd_from_cpts(cpts: Iterable[Sequence[Sequence[float]]]) -> str:
    """Get an SVG path data string for a list of list of Bezier control points.

    :param cpts: control points
    :return: SVG path data string
    """
    return _svgd_join(*_yield_svgd_cmds(cpts))


def _chunk_pairs(items: Sequence[_T]) -> Iterator[tuple[_T, _T]]:
    """Yield pairs of items from a sequence.

    :param items: a sequence of items
    :return: None
    :yield: pairs (without overlap) of items from the sequence
    :raises ValueError: if the number of items is not even
    """
    if len(items) % 2 != 0:
        msg = f"Expected an even number of items, got {len(items)}."
        raise ValueError(msg)
    for i in range(0, len(items), 2):
        yield (items[i], items[i + 1])


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
    head_cmd = _cmd_pts_from_string(svgd)
    lists: list[list[_StrPoint]] = []
    beg_idx = 0
    for cmd in _iter_cmds(head_cmd):
        if cmd.cmd == "M":
            beg_idx = len(lists)
        elif cmd.cmd == "Z":
            beg = lists[beg_idx][0]
            end = lists[-1][-1]
            if beg != end:
                lists.append([end, beg])
        elif cmd.prv is not None:
            lists.append([cmd.prv.pts[-1], *cmd.pts])

    return [[(float(p.x), float(p.y)) for p in curve] for curve in lists]


def make_relative(svgd: str) -> str:
    """Convert an absolute SVG path data string to a relative one.

    :param svgd: an ABSOLUTE SVG path data string
    :return: a RELATIVE SVG path data string
    """
    head_cmd = _cmd_pts_from_string(svgd)
    for cmd in _iter_cmds(head_cmd, rev=True):
        if cmd.cmd == "M" or cmd.prv is None:
            continue
        cmd.cmd = cmd.cmd.lower()
        cmd.pts = [p - cmd.prv.pts[-1] for p in cmd.pts]
    return _svgd_join(*it.chain.from_iterable(x.str for x in _iter_cmds(head_cmd)))


def make_absolute(svgd: str) -> str:
    """Convert a relative SVG path data string to an absolute one.

    :param svgd: a RELATIVE SVG path data string
    :return: an ABSOLUTE SVG path data string
    """
    cmd = _cmd_pts_from_string(svgd)
    return _svgd_join(*it.chain.from_iterable(x.str for x in _iter_cmds(cmd)))
