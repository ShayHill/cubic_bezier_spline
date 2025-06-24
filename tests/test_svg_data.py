"""Test generating SVG data strings.

:author: Shay Hill
:created: 2024-12-13
"""

# pyright: reportPrivateUsage = false

from typing import TypeVar

from paragraphs import par

from cubic_bezier_spline import (
    BezierSpline,
    new_closed_approximating_spline,
    new_open_approximating_spline,
)
from cubic_bezier_spline.pairwise import pairwise
from cubic_bezier_spline.svg_data import (
    _CmdPts,
    _do_use_curve_shorthand,
    _format_number,
    _StrPoint,
    _svgd_join,
    _svgd_split,
    get_cpts_from_svgd,
    get_svgd_from_cpts,
    make_absolute,
    make_relative,
)

_T = TypeVar("_T")


def assert_svgd_equal(result: str, expect: str):
    """Assert result == expect and test helper functions.

    This is just a method for running bonus circular tests on other test data.
    """
    assert result == expect
    assert _svgd_join(*_svgd_split(expect)) == expect
    assert get_svgd_from_cpts(get_cpts_from_svgd(expect)) == make_absolute(expect)
    cpts = get_cpts_from_svgd(expect)
    assert cpts == get_cpts_from_svgd(make_relative(expect))
    assert make_relative(make_absolute(expect)) == expect


class TestFormatNumber:
    def test_negative_zero(self):
        """Test that negative zero is formatted as zero."""
        assert _format_number(-1 / 1e10) == "0"


class TestCptsWithMidClose:
    def test_mid_close(self):
        """Insert multiple m commands if a path is closed in the middle."""
        cpts = [
            [(0, 0), (1, 0), (2, 0)],
            [(2, 0), (3, 0), (4, 0)],
            [(4, 0), (5, 0), (0, 0)],  # Close the path
            [(0, 5), (1, 5), (2, 5)],  # Another segment starting with M
            [(2, 5), (3, 5), (4, 5)],  # another disjoint segment, but returned to M
            [(3, 9), (4, 9), (5, 9)],  # Another segment starting with M
            [(5, 9), (6, 9), (3, 9)],  # Close the path
        ]
        expect = "M0 0q1 0 2 0t2 0-4 0zM0 5q1 0 2 0t2 0M3 9q1 0 2 0t-2 0z"
        result = make_relative(get_svgd_from_cpts(cpts))
        assert_svgd_equal(result, expect)


class TestDoUseCurveShorthand:
    def test_first_arg_not_a_curve(self):
        """Silently return False for non-curve commands."""
        cmd_a, cmd_b = (_CmdPts("L", [_StrPoint((0, 0))]) for _ in range(2))
        assert _do_use_curve_shorthand(cmd_a, cmd_b) is False

    def test_second_arg_not_a_curve(self):
        """Silently return False for non-curve commands."""
        cmd_a = _CmdPts("Q", [_StrPoint((x, 0)) for x in range(2)])
        cmd_b = _CmdPts("L", [_StrPoint((0, 0))])
        assert _do_use_curve_shorthand(cmd_a, cmd_b) is False


class TestClosedC2Continuous:
    def test_closed_approximating_spline(self):
        spline = new_closed_approximating_spline([(0, 0), (3, 0), (3, 3), (0, 3)])
        assert_svgd_equal(
            spline.svg_data,
            ("M0.5 0.5c0.5-0.5 1.5-0.5 2 0s0.5 1.5 0 2-1.5 0.5-2 0-0.5-1.5 0-2z"),
        )

    def test_open_approximating_spline(self):
        spline = new_open_approximating_spline([(0, 0), (3, 0), (3, 3), (0, 3)])
        assert_svgd_equal(
            spline.svg_data, ("M0 0c1 0 2 0 2.5 0.5s0.5 1.5 0 2-1.5 0.5-2.5 0.5")
        )

    def test_linear_closed(self):
        curves = list(pairwise(((0, 0), (3, 0), (3, 3), (0, 3), (0, 0))))
        spline = BezierSpline(curves)
        assert_svgd_equal(spline.svg_data, "M0 0h3v3h-3z")

    def test_linear_open(self):
        curves = list(pairwise(((0, 0), (3, 0), (3, 3), (0, 3))))
        spline = BezierSpline(curves)
        assert_svgd_equal(spline.svg_data, "M0 0h3v3h-3")

    def test_quadratic(self):
        curves = [[(0, 0), (1, 0)], [(1, 0), (2, 0), (3, 1)]]
        spline = BezierSpline(curves)
        assert_svgd_equal(spline.svg_data, ("M0 0h1q1 0 2 1"))


potrace_output = par(
    """M338 236 c-5 -3 -6 -6 -3 -6 1 -1 2 -2 2 -3 0 -2 1 -2 2 -2 2 0 3 0 4 -1 2 -2 2
    -2 4 -1 1 2 2 2 3 1 2 -3 6 0 6 6 1 8 -4 9 -11 3 l-3 -3 0 4 c0 3 -1 4 -4 2z M170
    235 h1v2l0 6c-2 0 -5 -1 -5 -1 -1 -1 -3 -1 -4 -1 -3 0 -13 -5 -14 -6 -1 -1 -2 -2 -4
    -2 -3 0 -6 -2 -4 -3 1 -1 1 -1 0 -1 -1 -1 -1 -1 -1 0 0 1 -1 1 -1 1 -2 0 -5 -4 -4
    -5 0 -1 -1 -1 -2 -2 -1 0 -4 -3 -8 -6 -4 -4 -9 -8 -11 -9 -6 -5 -15 -14 -14 -15 1
    -1 0 -1 -2 -2 -4 0 -8 -4 -11 -10 -4 -7 -1 -6 3 1 2 4 3 5 2 3 0 -2 -1 -4 -2 -5 -1
    0 -1 -1 -1 -1 1 -1 5 1 5 2 0 1 0 1 1 1 1 0 1 0 1 -1 -2 -2 2 -8 4 -8 0 1 2 1 2 1 1
    0 1 1 1 1 0 1 2 4 4 7 5 6 5 6 -2 7 l-4 1 5 0 c4 -1 5 0 7 2 2 2 4 3 4 3 1 0 0 -1
    -2 -3 -3 -3 -3 -3 -1 -5 1 -1 1 -1 0 -1 -2 1 -11 -10 -9 -12 2 -3 6 -2 9 3 3 2 5 4
    6 3 1 0 0 -1 -3 -3 -6 -5 -8 -8 -6 -10 2 -1 3 -1 4 2 3 6 9 9 12 6 2 -1 6 -2 6 0 0
    1 -6 6 -7 6 -3 0 2 5 7 8 3 1 4 6 3 9 -1 1 8 5 11 5 1 0 0 -1 -2 -2 -7 -2 -11 -9 -7
    -10 4 -2 12 5 12 10 0 2 0 2 1 1 0 -1 1 -2 0 -3 0 -1 0 -1 1 0 2 1 1 4 -2 5 -2 0 -2
    0 0 1 1 1 3 3 4 4 0 1 1 3 2 3 0 0 1 0 2 0 0 1 0 1 -1 1 0 -1 -1 -1 -1 0 0 0 2 1 4
    2 2 1 4 3 4 3 0 1 0 1 1 0 2 -1 8 2 8 4 0 1 2 3 4 4 2 1 4 2 4 2 0 -1 -1 -2 -3 -3
    -2 0 -3 -1 -3 -2 1 0 0 -2 -2 -3 -3 -2 -2 -4 2 -2 4 3 5 2 1 0 -4 -3 -10 -9 -9 -9 0
    0 1 1 3 1 1 1 3 2 4 2 2 0 4 1 6 4 3 3 5 4 5 3 1 -1 2 0 4 1 l2 3 -2 -3 s1 2 3 4s1
    2 3 4t1 2t8 5 c-1 -2 -2 -3 -3 -2 -2 0 -9 -6 -9 -8 1 -3 4 -2 7 1 2 2 4 3 4 2 1 -1
    1 -1 1 0 1 0 2 1 2 0 2 0 17 13 17 14 -1 1 6 5 8 5 2 1 10 3 12 4 3 1 5 1 5 0 0 -1
    2 -2 6 -3 3 -1 8 -3 10 -5 3 -2 5 -3 6 -3 1 0 1 -1 1 -1 0 -1 1 -2 1 -3 1 0 3 -4 5
    -8 2 -4 4 -7 5 -7 0 0 1 -1 2 -2 0 -2 1 -2 1 -1 1 1 0 2 -2 5 -1 2 -2 3 -1 2 1 -1 2
    -1 2 0 0 0 1 1 1 1 1 0 1 0 1 1 0 2 1 2 2 1 3 -2 4 0 1 2 -1 1 -2 3 -2 3 0 1 -1 2
    -1 3 -2 0 -2 3 0 3 2 1 2 1 1 -1 0 -3 5 -10 9 -11 1 0 2 1 1 1 0 0 1 1 2 2 0 1 1 2
    1 3 0 2 3 3 16 3 5 1 6 1 4 0 -12 0 -14 -1 -14 -3 1 -3 4 -5 6 -3 1 1 4 1 6 2 1 0 4
    0 5 1 1 0 2 0 2 -1 0 -1 0 -2 -1 -2 -1 0 -1 0 -1 -1 0 -1 1 0 2 1 2 1 3 2 2 2 0 1 1
    1 2 0 2 -1 2 -1 0 -3 -2 -1 -3 -4 -1 -3 0 1 2 0 3 -1 2 -1 3 -1 3 0 0 1 1 1 3 1 4 0
    5 1 2 3 -2 1 -2 1 0 1 2 0 3 0 4 1 0 1 1 1 1 1 1 0 0 -1 -1 -2 -1 -2 -1 -2 0 -3 1
    -1 1 -1 2 0 1 1 1 1 2 0 2 -2 5 0 5 3 -1 4 0 6 1 4 1 -1 1 -1 1 1 0 2 -1 3 -1 2 -1
    0 -1 2 -2 4 0 2 -1 3 -2 3 0 -1 -1 0 -2 1 -1 0 -1 1 -1 1 1 0 0 1 0 3 -2 3 -5 4 -5
    2 0 -1 -1 -1 -1 1 0 1 -1 1 -1 0 0 -1 0 -1 -2 0 -1 2 -4 2 -17 2 -8 0 -15 0 -16 1
    -2 0 -15 -3 -19 -4 -2 -2 -3 -1 -8 0 -4 1 -7 2 -8 1 -1 0 -2 0 -2 1 0 1 -6 3 -8 3
    -1 0 -2 0 -2 1 -1 1 -1 1 -1 0 0 -1 -2 -1 -11 0 -2 0 -2 0 1 1 3 1 2 1 -2 1 -4 -1
    -7 -1 -7 -2 0 -1 -1 -1 -2 0 -1 1 -2 1 -3 0 -2 -2 -5 -3 -3 -1 0 1 -4 1 -9 -1 -3 -1
    -5 -1 -5 0 -1 1 -3 1 -5 1 -2 0 -6 1 -9 2 -4 0 -7 1 -8 1 -1 0 -4 -1 -6 -1Q1 2 3 4Q4 2 5 3z"""
)


class TestPotraceOutput:
    def test_cycle(self) -> None:
        iterations: list[str] = []
        iterations.append(make_relative(potrace_output))
        iterations.append(make_relative(iterations[-1]))
        assert iterations[0] == iterations[1]
