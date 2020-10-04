#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Test methods in BezierSpline class

:author: Shay Hill
:created: 10/4/2020
"""

import random

import pytest

from bezier.bezier_spline import BezierSpline, TimeIntervalError

SHORT_SPLINE = BezierSpline([[[0], [1]], [[1], [2]]])


class TestBezierSpline:
    def test_call_low(self) -> None:
        """Raise TimeIntervalError if time < 0"""
        with pytest.raises(TimeIntervalError):
            _ = SHORT_SPLINE(-0.01)

    def test_call_high(self) -> None:
        """Raise TimeIntervalError if time > len(curves)"""
        with pytest.raises(TimeIntervalError):
            _ = SHORT_SPLINE(2.01)

    @pytest.mark.parametrize("time", (random.random() * 2 for _ in range(50)))
    def test_call(self, time):
        assert SHORT_SPLINE(time) == time

    def test_bottom(self):
        """Return spline value at bottom of time interval"""
        assert SHORT_SPLINE(0) == 0

    def test_top(self):
        """Return spline value at top of time interval"""
        assert SHORT_SPLINE(2) == 2

    @pytest.mark.parametrize("time", (0, 0.5, 1, 1.5, 2))
    def test_derivative(self, time) -> None:
        """Return value of spline at derivative"""
        assert SHORT_SPLINE(time, 1) == 1
