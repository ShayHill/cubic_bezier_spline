#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Transformations for lists of cubic splines

:author: Shay Hill
:created: 1/13/2020

TODO: perhaps ditch this completely
"""

from typing import List

from nptyping import Array

from .bezier_curves import BezierCurve


def reverse_path(path: List[BezierCurve]) -> List[BezierCurve]:
    return [BezierCurve(*reversed(x)) for x in reversed(path)]


def _translate_curve(curve: Curve, translation: Array[float, 2]) -> Curve:
    """
    Move the curve

    :param curve: Tuple of cubic Bezier control points
    :param translation: x, y relocation vector
    :return: a new Tuple (x + x, y + y)
    """
    return (
        curve[0] + translation,
        curve[1] + translation,
        curve[2] + translation,
        curve[3] + translation,
    )


def translate_path(path: List[Curve], translation: Array[float, 2]) -> List[Curve]:
    """
    Move the path.

    :param path: List of cubic Bezier control-point tuples
    :param translation: x, y relocation vector
    :return: each curve translated by translation
    """
    return [_translate_curve(x, translation) for x in path]


