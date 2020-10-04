#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Continuity of two Bezier curves

:author: Shay Hill
:created: 1/29/2020

I hesitate to include this in public code, because of the guesswork required to
produce the most useful answer, but here it is. See the docstring for get_continuity.
"""
from typing import Optional

import numpy as np

from .bezier_curve import BezierCurve


class _Continuity:
    """
    Assign an integer value to a continuity string.

    An instance will be like a function taking a string ('g\\d+' or r'c\\d+') and
    returning an integer, EXCEPT the "function" is called with getattr rather than
    call. This is because I'm refactoring an IntEnum. The only purpose is to compare
    continuity strings, and IntEnum is the common way to do that. For that purpose,
    an instance of _Continuity will look and work like an infinite IntEnum.
    """

    def __getattr__(self, continuity: str) -> int:
        if "g" in continuity:
            return int(continuity[1:]) * 2
        return 1 + int(continuity[1:]) * 2


Continuity = _Continuity()


def get_continuity(curve_a: BezierCurve, curve_b: BezierCurve) -> Optional[int]:
    """
    What is the continuity where a joins b?

    :param curve_a: curve before potential discontinuity
    :param curve_b: curve after potential discontinuity
    :return: continuity of curve_a and curve_b as an int, or None if none

    c0 -> 1
    g1 -> 2
    c1 -> 3

    The specific integer values are only important for comparison:

    if retval < Continuity.c1: ...

    There are various naming conventions for continuity. Here's the one I'm using:

    If equal in DIRECTION at derivative n:
    - gn continuous

    If ADDITIONALLY equal in MAGNITUDE at derivative n:
    - cn continuous

    Two curves that touch at the endpoints but are otherwise not continuous:
    - c0 continuous

    This isn't as well defined as I'd want it to be. Take these two curves:
    ((0, 0), (1, 0), (1, 0))
    ((1, 0), (1, 0), (1, 1))

    When drawn, they will appear as two straight lines at a right angle, but the
    curves are arguably c1 continuous because they have matching tangent vectors
    [(0, 0)] at (0, 1). This function will call that c0 continuous. Those types of
    curves (double-endpoints) DO appear in fonts.

    In this function, matching zero-length derivatives to not imply tangency. Of
    course, now that we have TWO rules (equal AND not 0 at derivative), there's going
    to be some gray area.

    Take two cubic Bezier curves that happen to be straight line segments. If these
    are touching and parallel, they can end up with anything from g1 to c3 continuity
    depending on the spacing of their control points. Like the zero-length derivative
    case, this makes mathematical sense but not perceptual sense. Unlike the
    zero-length derivative case, there is no simple heuristic (of which I am aware)
    to get a result that matches perception.

    I don't even know how to construct a g2 Bezier curve, so gn where n > 1 has not
    been tested.
    """
    if not np.array_equal(curve_a(1), curve_b(0)):
        return
    continuity = Continuity.c0

    max_continuity = min(curve_a.degree, curve_b.degree)
    for derivative in range(1, max_continuity + 1):
        derivative_a = curve_a(1, derivative)
        derivative_b = curve_b(0, derivative)
        if not all(np.count_nonzero(x) for x in (derivative_a, derivative_b)):
            return continuity
        if np.allclose(derivative_a, derivative_b):
            continuity = getattr(Continuity, f"c{derivative}")
        else:
            derivative_a = derivative_a / np.linalg.norm(derivative_a)
            derivative_b = derivative_b / np.linalg.norm(derivative_b)
            if np.allclose(derivative_a, derivative_b):
                return getattr(Continuity, f"g{derivative}")
            else:
                return continuity
    return continuity
