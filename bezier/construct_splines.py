#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Constructors for Bezier splines

:author: Shay Hill
:created: 10/4/2020
"""

from .bezier_spline import BezierSpline
from typing import Iterable
import numpy as np


def get_cubic_spline(cpts: Iterable[Iterable[float]]):
    cpts = np.array(tuple(iter(cpts)))
    thirds = [[x, (2 * x + y) / 3, (x + 2 * y) / 3, y] for x, y in zip(cpts, cpts[1:])]
    for aaa, bbb in zip(thirds, thirds[1:]):
        new_point = (aaa[2] + bbb[1]) / 2
        aaa[-1] = new_point
        bbb[0] = new_point
    return BezierSpline(thirds)

aaa = [[0, 0], [1, 0], [1, 1], [0, 1]]
bbb = get_cubic_spline(aaa)
breakpoint()
