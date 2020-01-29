#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Test functions in ttf_extractor.path_conversion.py

:author: Shay Hill
:created: 1/13/2020
"""

import numpy as np
import pytest

from ttf_extractor.path_converter import _PathConverter


class TestPathConverter:
    """Only recursive tests

    I might factor out PathConverter._yield at some point, so test everything for
    every command.
    """

    def test_M(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3 6 9")
        np.testing.assert_equal(curves, [((0, 3), (6, 9))])

    def test_m(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3m6 9 12 15")
        np.testing.assert_equal(curves, [((6, 12), (12, 18))])

    def test_L(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3L6 9 12 15")
        np.testing.assert_equal(
            curves, [((0, 3), (6, 9)), ((6, 9), (12, 15))],
        )

    def test_l(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3l6 9 12 15")
        np.testing.assert_equal(
            curves, [((0, 3), (6, 12)), ((6, 12), (12, 18))],
        )

    def test_H(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3H6 9 12 15")
        np.testing.assert_equal(
            curves,
            [
                ((0, 3), (6, 3)),
                ((6, 3), (9, 3)),
                ((9, 3), (12, 3)),
                ((12, 3), (15, 3)),
            ],
        )

    def test_h(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3h6 9 12 15")
        np.testing.assert_equal(
            curves,
            [
                ((0, 3), (6, 3)),
                ((6, 3), (9, 3)),
                ((9, 3), (12, 3)),
                ((12, 3), (15, 3)),
            ],
        )

    def test_V(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3V6 9 12 15")
        np.testing.assert_equal(
            curves,
            [
                ((0, 3), (0, 6)),
                ((0, 6), (0, 9)),
                ((0, 9), (0, 12)),
                ((0, 12), (0, 15)),
            ],
        )

    def test_v(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3v6 9 12 15")
        np.testing.assert_equal(
            curves,
            [
                ((0, 3), (0, 9)),
                ((0, 9), (0, 12)),
                ((0, 12), (0, 15)),
                ((0, 15), (0, 18)),
            ],
        )

    def test_C(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3C6 9 12 15 18 21 24 27 30 33 36 39")
        np.testing.assert_equal(
            curves,
            [
                ((0, 3), (6, 9), (12, 15), (18, 21)),
                ((18, 21), (24, 27), (30, 33), (36, 39)),
            ],
        )

    def test_c(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3c6 9 12 15 18 21 24 27 30 33 36 39")
        np.testing.assert_equal(
            curves,
            [
                ((0, 3), (6, 12), (12, 18), (18, 24)),
                ((18, 24), (24, 30), (30, 36), (36, 42)),
            ],
        )

    def test_S(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3S12 15 18 21 30 33 36 39")
        np.testing.assert_equal(
            curves,
            [
                ((0, 3), (0, 3), (12, 15), (18, 21)),
                ((18, 21), (24, 27), (30, 33), (36, 39)),
            ],
        )

    def test_s(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3s12 15 18 21 30 33 36 39")
        np.testing.assert_equal(
            curves,
            [
                ((0, 3), (0, 3), (12, 18), (18, 24)),
                ((18, 24), (24, 30), (30, 36), (36, 42)),
            ],
        )

    def test_Q(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3Q6 9 12 15 18 21 24 27")
        np.testing.assert_equal(
            curves, [((0, 3), (6, 9), (12, 15)), ((12, 15), (18, 21), (24, 27))],
        )

    def test_q(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3q6 9 12 15 18 21 24 27")
        np.testing.assert_equal(
            curves, [((0, 3), (6, 12), (12, 18)), ((12, 18), (18, 24), (24, 30))],
        )

    def test_T(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3T12 15 124 127")
        np.testing.assert_equal(
            curves, [((0, 3), (0, 3), (12, 15)), ((12, 15), (24, 27), (124, 127))],
        )

    def test_t(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3t12 15 24 27")
        np.testing.assert_equal(
            curves, [((0, 3), (0, 3), (12, 18)), ((12, 18), (24, 33), (24, 30))],
        )

    def test_z_open(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3 6 9z")
        np.testing.assert_equal(
            curves,
            [
                ((0, 3), (6, 9)),
                ((6, 9), (0, 3)),
            ],
        )

    def test_z_close(self) -> None:
        convert_path = _PathConverter()
        curves = convert_path("M0 3 6 9 0 3z")
        np.testing.assert_equal(
            curves,
            [
                ((0, 3), (6, 9)),
                ((6, 9), (0, 3)),
            ],
        )

    def test_not_implemented(self) -> None:
        """Raise NotImplementedError for other commands"""
        convert_path = _PathConverter()
        with pytest.raises(NotImplementedError):
            curves = convert_path("M0 3A6 9 0 3z")
