#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Test functions in ttf_extractor.bezier_curves.py

:author: Shay Hill
:created: 1/18/2020
"""
from math import isnan

import numpy as np
import pytest

from bezier.bezier_curves import (
    BezierCurve,
    _cbez,
    _cbez_d1,
    _cbez_d2,
)

from ttf_extractor import extract_glyphs

class TestCubicBezier:
    def test_call(self) -> None:
        """Test against formula"""
        for _ in range(100):
            points = [np.random.random(3) for _ in range(4)]
            time = np.random.random()

            curve = BezierCurve(*points)
            np.testing.assert_allclose(curve(time), _cbez(*curve, time))

    def test_d1(self) -> None:
        """Test against formula"""
        for _ in range(100):
            points = [np.random.random(3) for _ in range(4)]
            time = np.random.random()

            curve = BezierCurve(*points)
            np.testing.assert_allclose(curve(time, 1), _cbez_d1(*curve, time))

    def test_d2(self) -> None:
        """Test against formula"""
        for _ in range(100):
            points = [np.random.random(3) for _ in range(4)]
            time = np.random.random()

            curve = BezierCurve(*points)
            np.testing.assert_allclose(curve(time, 2), _cbez_d2(*curve, time))

    def test_d3(self) -> None:
        """Test against formula"""
        for _ in range(100):
            points = [np.random.random(3) for _ in range(4)]
            time = np.random.random()

            curve = BezierCurve(*points)
            p0, p1, p2, p3 = curve
            np.testing.assert_allclose(curve(time, 3), 6 * (p3 - 3 * p2 + 3 * p1 - p0))

    def test_d4(self) -> None:
        """Raise ValueError if derivative > degree"""
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        curve = BezierCurve((0, 0), (1, 0), (1, 1), (0, 1))
        with pytest.raises(ValueError) as excinfo:
            curve(0.5, 4)
        assert "Bezier curve of degree" in str(excinfo.value)

    def test_split(self) -> None:
        """
        New curves are continuous at ends and split.
        """
        for _ in range(100):
            points = [np.random.random(3) for _ in range(4)]
            time = np.random.random()
            curve = BezierCurve(*points)
            beg, end = curve.split(time)
            np.testing.assert_allclose(
                curve.planar_curvature(0), beg.planar_curvature(0)
            )
            np.testing.assert_allclose(
                curve.planar_curvature(time), beg.planar_curvature(1)
            )
            np.testing.assert_allclose(
                curve.planar_curvature(time), end.planar_curvature(0)
            )
            np.testing.assert_allclose(
                curve.planar_curvature(1), end.planar_curvature(1)
            )

    def test_curvature_min(self) -> None:
        """
        Return 0 when curve is flat.
        """
        curve = BezierCurve((0, 0), (1, 0), (2, 0))
        assert curve.planar_curvature(0.5) == 0

    def test_curvature_max(self) -> None:
        """
        Return nan when curvature is infinite (curve reverses on self).
        """
        curve = BezierCurve((0, 0), (1, 0), (0, 0))
        assert isnan(curve.planar_curvature(0.5))
