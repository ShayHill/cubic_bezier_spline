#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Test functions in ttf_extractor.path_transformations.py

:author: Shay Hill
:created: 1/17/2020
"""

from bezier.path_transformations import translate_path, path_over
from bezier.bezier_curves import BezierCurve
from ttf_extractor.ttf_extractor.path_converter import svg_to_bezier
import numpy as np


class TestPathOver:
    def test_closed(self) -> None:
        """A closed path is still closed."""
        path_a = svg_to_bezier(
            BezierCurve(x) for x in svg_to_bezier("M-1 -1 1 -1 1 1 -1 1z")
        )
        path_b = _path_over(path_a, .5)



class TestTranslatePath:
    """
    Something isn't working here. Keep testing till the problem is found.
    """

    def test_box_open(self) -> None:
        """Box path transforms correctly."""
        path_a = svg_to_bezier("M0 0 3 0 3 3 0 3z")
        path_b = translate_path(path_a, np.array([5, 0]))
        np.testing.assert_allclose(
            path_b,
            [
                ((5.0, 0.0), (6.0, 0.0), (7.0, 0.0), (8.0, 0.0),),
                ((8.0, 0.0), (8.0, 1.0), (8.0, 2.0), (8.0, 3.0),),
                ((8.0, 3.0), (7.0, 3.0), (6.0, 3.0), (5.0, 3.0),),
                ((5.0, 3.0), (5.0, 2.0), (5.0, 1.0), (5.0, 0.0),),
            ],
        )

    def test_box_closed(self) -> None:
        """Box path transforms correctly."""
        path_a = svg_to_bezier("M0 0 3 0 3 3 0 3 0 0z")
        path_b = translate_path(path_a, np.array([5, 0]))
        np.testing.assert_allclose(
            path_b,
            [
                ((5.0, 0.0), (6.0, 0.0), (7.0, 0.0), (8.0, 0.0),),
                ((8.0, 0.0), (8.0, 1.0), (8.0, 2.0), (8.0, 3.0),),
                ((8.0, 3.0), (7.0, 3.0), (6.0, 3.0), (5.0, 3.0),),
                ((5.0, 3.0), (5.0, 2.0), (5.0, 1.0), (5.0, 0.0),),
            ],
        )

    def test_Q(self) -> None:
        """Box path transforms correctly."""
        path_a = svg_to_bezier("M0 0Q3 0 3 3z")
        path_b = translate_path(path_a, np.array([5, 3]))
        points_a = [x for y in path_a for x in y]
        points_b = [x for y in path_b for x in y]
        assert all(x[0] + 5 == y[0] for x, y in zip(points_a, points_b))
        assert all(x[1] + 3 == y[1] for x, y in zip(points_a, points_b))

    def test_q(self) -> None:
        """Box path transforms correctly."""
        path_a = svg_to_bezier("M0 0q3 0 3 3z")
        path_b = translate_path(path_a, np.array([5, 3]))
        points_a = [x for y in path_a for x in y]
        points_b = [x for y in path_b for x in y]
        assert all(x[0] + 5 == y[0] for x, y in zip(points_a, points_b))
        assert all(x[1] + 3 == y[1] for x, y in zip(points_a, points_b))

    def test_t(self) -> None:
        """Box path transforms correctly."""
        path_a = svg_to_bezier("M0 0t3 0 3 3 0 3 0 0z")
        path_b = translate_path(path_a, np.array([5, 3]))
        points_a = [x for y in path_a for x in y]
        points_b = [x for y in path_b for x in y]
        assert all(x[0] + 5 == y[0] for x, y in zip(points_a, points_b))
        assert all(x[1] + 3 == y[1] for x, y in zip(points_a, points_b))

    def test_C(self) -> None:
        """Box path transforms correctly."""
        path_a = svg_to_bezier("M0 0C3 0 3 3 0 3z")
        path_b = translate_path(path_a, np.array([5, 3]))
        points_a = [x for y in path_a for x in y]
        points_b = [x for y in path_b for x in y]
        assert all(x[0] + 5 == y[0] for x, y in zip(points_a, points_b))
        assert all(x[1] + 3 == y[1] for x, y in zip(points_a, points_b))

    def test_c(self) -> None:
        """Box path transforms correctly."""
        path_a = svg_to_bezier("M0 0c3 0 3 3 0 3z")
        path_b = translate_path(path_a, np.array([5, 3]))
        points_a = [x for y in path_a for x in y]
        points_b = [x for y in path_b for x in y]
        assert all(x[0] + 5 == y[0] for x, y in zip(points_a, points_b))
        assert all(x[1] + 3 == y[1] for x, y in zip(points_a, points_b))

    def test_S(self) -> None:
        """Box path transforms correctly."""
        path_a = svg_to_bezier("M0 0S3 3 0 3z")
        path_b = translate_path(path_a, np.array([5, 3]))
        points_a = [x for y in path_a for x in y]
        points_b = [x for y in path_b for x in y]
        assert all(x[0] + 5 == y[0] for x, y in zip(points_a, points_b))
        assert all(x[1] + 3 == y[1] for x, y in zip(points_a, points_b))

    def test_s(self) -> None:
        """Box path transforms correctly."""
        path_a = svg_to_bezier("M0 0s3 3 0 3z")
        path_b = translate_path(path_a, np.array([5, 3]))
        points_a = [x for y in path_a for x in y]
        points_b = [x for y in path_b for x in y]
        assert all(x[0] + 5 == y[0] for x, y in zip(points_a, points_b))
        assert all(x[1] + 3 == y[1] for x, y in zip(points_a, points_b))
