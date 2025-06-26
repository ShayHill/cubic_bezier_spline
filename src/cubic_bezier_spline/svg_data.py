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

import enum
import itertools as it
import math
import re
from string import ascii_lowercase
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

# Number of places after the decimal point to write numbers when converting from
# float values to svg path data string floats.
PRECISION = 6

_T = TypeVar("_T")


class RelativeOrAbsolute(str, enum.Enum):
    """Enum to indicate whether a path is relative or absolute."""

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


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


def _nums_print_equal(*numbers: float | str) -> bool:
    """Check if first half of the numbers are equal to the second half when printed.

    :param numbers: values to compare
    :return: True if all values are equal when printed, False otherwise
    :raises ValueError: if the number of numbers is not even
    """
    if len(numbers) // 2 != len(numbers) / 2:
        msg = f"Expected an even number of numbers, got {len(numbers)}."
        raise ValueError(msg)
    nos_a = [_format_number(n) for n in numbers[: len(numbers) // 2]]
    nos_b = [_format_number(n) for n in numbers[len(numbers) // 2 :]]
    return all(a == b for a, b in zip(nos_a, nos_b))


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


def _get_curve_shorthand_command(cmd: PathCommand) -> str:
    """If a curve command can be shortened, return the shorthand SVG command.

    :param cmd: the command to check
    :return: the input cmd.cmd or a shorthand replacement ("H", "V", "T", "S", "Z")
    """
    if cmd.cmd in "QC":
        if cmd.prev and cmd.prev.cmd == cmd.cmd:
            tan_x = cmd.prev.xs[-1] - cmd.prev.xs[-2]
            tan_y = cmd.prev.ys[-1] - cmd.prev.ys[-2]
            if _nums_print_equal(tan_x, tan_y, cmd.xsr[0], cmd.ysr[0]):
                return "T" if cmd.cmd == "Q" else "S"
        elif _nums_print_equal(cmd.xsr[0], cmd.ysr[0], 0, 0):
            return "T" if cmd.cmd == "Q" else "S"
    if cmd.cmd == "L":
        eq_x = _nums_print_equal(cmd.xs[0], cmd.prev_x)
        eq_y = _nums_print_equal(cmd.ys[0], cmd.prev_y)
        if eq_x:
            return "V"
        if eq_y:
            return "H"
    return cmd.cmd


# what is the degree of each basic command?
N_2_CMD = {1: "L", 2: "Q", 3: "C"}

# how many floats does each command take?
CMD_2_LEN = {"m": 2, "l": 2, "h": 1, "v": 1, "q": 4, "t": 2, "c": 6, "s": 4, "z": 0}


class PathCommand:
    """A command with points.

    haracter-saving steps to create an SVG path data string.

    The str properties strip out unnecessary commands and points.
    """

    def __init__(
        self,
        cmd: str | None,
        xs: Iterable[float] | None = None,
        ys: Iterable[float] | None = None,
        prev: PathCommand | None = None,
    ) -> None:
        """Create a command with points.

        :param cmd: the SVG command (e.g. "M", "L", "Q", "C")
        :param pts: the points for this command
        """
        xs = None if xs is None else list(xs)
        ys = None if ys is None else list(ys)
        max_len = max(len(xs) if xs else 0, len(ys) if ys else 0)
        self._xsa = None
        self._ysa = None
        self._xsr = None
        self._ysr = None

        if cmd and cmd in ascii_lowercase:
            self._xsr = xs
            self._ysr = ys
            self.cmd = cmd.upper()
        else:
            self._xsa = xs
            self._ysa = ys
            self.cmd = cmd or N_2_CMD[max_len]
        self.prev = prev
        self.next: PathCommand | None = None
        if self.prev is not None:
            self.prev.next = self

    def __repr__(self) -> str:
        """Get the SVG command and points for this command.

        :return: the SVG command and points as a string
        """
        return f"Command('{self.cmd}', {self.xs}, {self.ys})"

    @property
    def prev_x(self) -> float:
        """Get the current x coordinate.

        :return: the x coordinate of the last point in the previous command
        """
        if self.prev is None:
            return 0.0
        return self.prev.xs[-1]

    @property
    def prev_y(self) -> float:
        """Get the current y coordinate.

        :return: the y coordinate of the last point in the previous command
        """
        if self.prev is None:
            return 0.0
        return self.prev.ys[-1]

    @property
    def xs(self) -> list[float]:
        """Get the x coordinates of the points.

        :return: the x coordinates of the points
        """
        if self._xsa is None:
            self._xsa = [self.prev_x + x for x in self.xsr]
        return self._xsa

    @property
    def ys(self) -> list[float]:
        """Get the y coordinates of the points.

        :return: the y coordinates of the points
        """
        if self._ysa is None:
            self._ysa = [self.prev_y + y for y in self.ysr]
        return self._ysa

    @property
    def xsr(self) -> list[float]:
        """Get the relative x coordinates of the points as strings.

        :return: the relative x coordinates of the points
        :raises ValueError: if no x coordinates can be inferred
        """
        if self._xsr:
            return self._xsr
        if self._xsa:
            self._xsr = [x - self.prev_x for x in self._xsa]
            return self._xsr
        ys = self._ysr or self._ysa
        if ys is not None:
            self._xsr = [0.0 for _ in ys]
            return self._xsr
        msg = "No y coordiates available to assume 0 relative x."
        raise ValueError(msg)

    @property
    def ysr(self) -> list[float]:
        """Get the relative y coordinates of the points as strings.

        :return: the relative y coordinates of the points
        :raises ValueError: if no y coordinates can be inferred
        """
        if self._ysr:
            return self._ysr
        if self._ysa:
            self._ysr = [y - self.prev_y for y in self._ysa]
            return self._ysr
        xs = self._xsr or self._xsa
        if xs is not None:
            self._ysr = [0.0 for _ in xs]
            return self._ysr
        msg = "No x coordiates available to assume 0 relative y."
        raise ValueError(msg)

    def set_relative(self, xs: Iterable[float], ys: Iterable[float]) -> None:
        """Set the relative x and y coordinates of the points.

        :param xs: the relative x coordinates of the points
        :param ys: the relative y coordinates of the points
        """
        self._xsr = list(xs)
        self._ysr = list(ys)
        self._xsa = None
        self._ysa = None

    @property
    def str_cmd(self) -> str:
        """Get the SVG command for this command as it will be used in the SVG data.

        :return: the SVG command (e.g. "M", "L", "Q", "C", "V", "H", ...)
        """
        return _get_curve_shorthand_command(self)

    def iter_str_pts(self, relative_or_absolute: RelativeOrAbsolute) -> Iterator[str]:
        """Iterate over the points in this command as strings.

        :param relative_or_absolute: whether to return relative or absolute coordinates
        :return: an iterator over the points as strings
        :raises ValueError: if the relative_or_absolute value is unknown
        """
        if relative_or_absolute == RelativeOrAbsolute.ABSOLUTE:
            xs = self.xs
            ys = self.ys
        elif relative_or_absolute == RelativeOrAbsolute.RELATIVE:
            xs = self.xsr
            ys = self.ysr
        else:
            msg = f"Unknown relative_or_absolute value: {relative_or_absolute}"
            raise ValueError(msg)
        if self.str_cmd == "Z":
            return
        if self.str_cmd == "V":
            yield _format_number(ys[0])
        elif self.str_cmd == "H":
            yield _format_number(xs[0])
        elif self.str_cmd in "TS":
            xys = zip(xs[1:], ys[1:])
            yield from map(_format_number, it.chain(*xys))
        else:
            xys = zip(xs, ys)
            yield from map(_format_number, it.chain(*xys))


class PathCommands:
    """A linked list of commands.

    This class is used to create a linked list of _Command objects. It is used to
    convert a list of control points to an SVG path data string.
    """

    def __init__(self, cmd: PathCommand) -> None:
        """Create a linked list of commands.

        :param cmd: the first command in the linked list
        """
        self.head = cmd

    def __iter__(self) -> Iterator[PathCommand]:
        """Iterate over the commands in the linked list.

        :return: an iterator over the commands in the linked list
        """
        cmd: PathCommand | None = self.head
        while cmd is not None:
            yield cmd
            cmd = cmd.next

    @classmethod
    def from_cpts(
        cls, cpts: Iterable[Iterable[Iterable[float]]], cmd: str | None = None
    ) -> PathCommands:
        """Create a linked list of commands from a list of tuples.

        :param cpts: a list of curves, each a list of xy control points
        :param cmd: the command to use for the first point, defaults to "M"
        :return: an instance of PathCommands linked list
        :raises ValueError: if no commands can be created from the control points
        """
        cpts_ = [[(x, y) for x, y in c] for c in cpts]
        node: None | PathCommand = None
        path_open: tuple[float, float] = (math.inf, math.inf)
        for curve in cpts_:
            xs, ys = zip(*curve)
            if node is None or not _nums_print_equal(
                node.xs[-1], node.ys[-1], *curve[0]
            ):
                node = PathCommand("M", xs[:1], ys[:1], node)
                path_open = curve[0]
            cmd = (
                "Z"
                if len(curve) == 2 and _nums_print_equal(*curve[-1], *path_open)
                else None
            )
            node = PathCommand(cmd, xs[1:], ys[1:], node)
            if len(curve) > 2 and _nums_print_equal(*curve[-1], *path_open):
                node = PathCommand("z", [0], [0], node)
        if node is None:
            msg = "No commands created from control points."
            raise ValueError(msg)
        while node.prev is not None:
            node = node.prev
        return cls(node)

    @classmethod
    def from_svgd(cls, svgd: str) -> PathCommands:
        """Create a linked list of commands from an SVG path data string.

        :param svgd: an ABSOLUTE SVG path data string
        :return: the first command in the linked list
        :raises ValueError: if the SVG data string contains arc commands
        """
        if "a" in svgd.lower():
            msg = (
                f"Arc commands cannot be converted to Bezier control points in {svgd}."
            )
            raise ValueError(msg)

        parts = _svgd_split(svgd)  # e.g., ["M", "0", "0", "H", "1", "V", "2"]
        cmd_str = parts.pop(0)
        path_open = (float(parts.pop(0)), float(parts.pop(0)))
        node = PathCommand(cmd_str, [path_open[0]], [path_open[1]], None)
        cmd_str = {"m": "l", "M": "L"}[cmd_str]
        while parts:
            if parts[0].lower() in CMD_2_LEN:
                cmd_str = parts.pop(0)
            num = CMD_2_LEN[cmd_str.lower()]
            nums = [float(parts.pop(0)) for _ in range(num)]
            if cmd_str in "mM":
                path_open = (nums[0], nums[1])
                node = PathCommand(cmd_str, [path_open[0]], [path_open[1]], node)
            elif cmd_str in "vV":
                node = PathCommand(cmd_str, None, nums, node)
            elif cmd_str in "hH":
                node = PathCommand(cmd_str, nums, None, node)
            elif cmd_str in "Zz":
                node = PathCommand("Z", [path_open[0]], [path_open[1]], node)
            else:
                xs, ys = zip(*list(_chunk_pairs(nums)))
                node = PathCommand(cmd_str, xs, ys, node)
        while node.prev is not None:
            node = node.prev
        cmds = PathCommands(node)

        for cmd in cmds:
            if cmd.cmd not in "TS":
                continue
            cmd.cmd = {"T": "Q", "S": "C"}.get(cmd.cmd, cmd.cmd)
            if cmd.prev and cmd.cmd == cmd.prev.cmd:
                vx = cmd.prev.xs[-1] - cmd.prev.xs[-2]
                vy = cmd.prev.ys[-1] - cmd.prev.ys[-2]
                cmd.set_relative([vx, *cmd.xsr], [vy, *cmd.ysr])
            else:
                cmd.set_relative([0, *cmd.xsr], [0, *cmd.ysr])
        return cmds

    @property
    def abs_svgd(self) -> str:
        """Get the SVG path data string for the commands in the linked list.

        :return: an ABSOLUTE SVG path data string
        """
        bits: list[str] = []
        for cmd in self:
            if cmd.prev is None or cmd.str_cmd != cmd.prev.str_cmd:
                bits.append(cmd.str_cmd)
            bits.extend(cmd.iter_str_pts(RelativeOrAbsolute.ABSOLUTE))
        return _svgd_join(*bits)

    @property
    def rel_svgd(self) -> str:
        """Get the SVG path data string for the commands.

        :return: a RELATIVE SVG path data string
        """
        bits: list[str] = []
        for cmd in self:
            if cmd.prev is None or cmd.str_cmd != cmd.prev.str_cmd:
                bits.append(cmd.str_cmd.lower())
            bits.extend(cmd.iter_str_pts(RelativeOrAbsolute.RELATIVE))
        return _svgd_join(*bits)

    @property
    def cpts(self) -> list[list[tuple[float, float]]]:
        """Get the control points from the commands in the linked list.

        :return: a list of lists of control points
        :raises ValueError: if the first command is not a move command
        """
        cpts: list[list[tuple[float, float]]] = []
        path_open = (math.inf, math.inf)
        for cmd in self:
            if cmd.cmd == "M":
                path_open = (cmd.xs[0], cmd.ys[0])
            elif cmd.cmd == "Z":
                if not _nums_print_equal(cmd.prev_x, cmd.prev_y, *path_open):
                    xs = [cmd.prev_x, *cmd.xs]
                    ys = [cmd.prev_y, *cmd.ys]
                    cpts.append(list(zip(xs, ys)))
            elif cmd.prev is None:
                msg = "First command is not a move command."
                raise ValueError(msg)
            else:
                xs = [cmd.prev_x, *cmd.xs]
                ys = [cmd.prev_y, *cmd.ys]
                cpts.append(list(zip(xs, ys)))
        cmd = self.head
        while cmd.next is not None:
            cmd = cmd.next

        return cpts


def make_relative(svgd: str) -> str:
    """Convert an absolute SVG path data string to a relative one.

    :param svgd: an ABSOLUTE SVG path data string
    :return: a RELATIVE SVG path data string
    """
    return PathCommands.from_svgd(svgd).rel_svgd


def make_absolute(svgd: str) -> str:
    """Convert a relative SVG path data string to an absolute one.

    :param svgd: a RELATIVE SVG path data stming
    :return: an ABSOLUTE SVG path data string
    """
    return PathCommands.from_svgd(svgd).abs_svgd


def get_cpts_from_svgd(svgd: str) -> list[list[tuple[float, float]]]:
    """Get a list of lists of Bezier control points from an SVG path data string.

    :param svgd: an absolute or relative SVG path data string
    :return: a list of curves, each a list of xy tuples.
    """
    return PathCommands.from_svgd(svgd).cpts


def get_svgd_from_cpts(cpts: Iterable[Iterable[Iterable[float]]]) -> str:
    """Get an SVG path data string for a list of list of Bezier control points.

    :param cpts: a list of curves, each a list of xy control points
    :return: SVG path data string
    """
    return PathCommands.from_cpts(cpts).abs_svgd
