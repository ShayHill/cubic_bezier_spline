#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Test functions in bezier_to_d

:author: Shay Hill
:created: 2/11/2020
"""

from my_bezier.ttf_extractor.bezier_to_svg import _loop_to_d
from my_bezier import BezierCurve


class TestLoopToD:
    def test_H(self) -> None:
        """"""
        aaa = _loop_to_d([BezierCurve((0, 0), (1, 0))])
        assert aaa == "M0 0H1z"

    def test_V(self) -> None:
        """"""
        aaa = _loop_to_d([BezierCurve((0, 0), (0, 1))])
        assert aaa == "M0 0V1z"

    def test_L(self) -> None:
        """"""
        aaa = _loop_to_d([BezierCurve((0, 0), (1, 1))])
        assert aaa == "M0 0 1 1z"

    def test_Q(self) -> None:
        """"""
        aaa = _loop_to_d([BezierCurve((0, 0), (1, 1), (1, 0))])
        assert aaa == "M0 0Q1 1 1 0z"

    def test_T(self) -> None:
        """"""
        aaa = _loop_to_d(
            [BezierCurve((0, 0), (1, 1), (1, 0)), BezierCurve((1, 0), (1, -1), (2, 2))]
        )
        assert aaa == "M0 0Q1 1 1 0T2 2z"

    def test_T_knot(self) -> None:
        """"""
        aaa = _loop_to_d(
            [BezierCurve((0, 0), (1, 1)), BezierCurve((1, 1), (1, 1), (2, 2))]
        )
        assert aaa == "M0 0 1 1T2 2z"

    def test_C(self) -> None:
        """"""
        aaa = _loop_to_d([BezierCurve((0, 0), (0, 1), (1, 1), (0, 1))])
        assert aaa == "M0 0C0 1 1 1 0 1z"

    def test_S(self) -> None:
        """"""
        aaa = _loop_to_d(
            [
                BezierCurve((0, 0), (0, 1), (1, 1), (0, 1)),
                BezierCurve((0, 1), (-1, 1), (-1, 0), (0, 0)),
            ]
        )
        assert aaa == "M0 0C0 1 1 1 0 1S-1 0 0 0z"

    def test_S_knot(self) -> None:
        """"""
        aaa = _loop_to_d([BezierCurve((0, 0), (0, 0), (1, 0), (1, 1))])
        assert aaa == "M0 0S1 0 1 1z"

    def test_chain(self) -> None:
        """
        Remove unnecessary path codes
        :return:
        """
        aaa = _loop_to_d(
            [
                BezierCurve((0, 0), (2, 1)),
                BezierCurve((2, 1), (1, 1)),
                BezierCurve((1, 1), (0, 1)),
                BezierCurve((0, 1), (0, 0)),
                BezierCurve((0, 0), (0, 1), (1, 1), (0, 1)),
                BezierCurve((0, 1), (0, 0), (1, 0), (1, 1)),
                BezierCurve((1, 1), (0, 0)),
            ]
        )
        assert aaa == "M0 0 2 1H1 0V0C0 1 1 1 0 1 0 0 1 0 1 1z"
