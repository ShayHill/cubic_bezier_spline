#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
""" Create matrices for Bezier operations

:author: Shay Hill
:created: 10/2/2020
"""
from functools import lru_cache
from typing import Any, Tuple

import numpy as np
from nptyping import NDArray


def _get_checkerboard(
    shape: Tuple[int, ...], a_val: Any, b_val: Any
) -> NDArray[Any, Any]:
    """
    An nD array of [[a, b], [b, a]]

    :param shape: dimensions of checkerboard
    :param a_val: value at [0][0], [0][2], ... (in 2D)
    :param b_val: value at [0][1], [0][3], ... (in 2D)
    :return: +1 and -1 "checkered" over dimensions. +1 at [0][0]
    """
    check01 = np.indices(shape).sum(axis=0) % 2
    checkered = np.empty_like(check01)
    checkered[np.where(check01 == 0)] = a_val
    checkered[np.where(check01 == 1)] = b_val
    # noinspection PyTypeChecker
    return checkered


@lru_cache
def binom(n, k):
    """
    n choose k

    :param n: number of candidates
    :param k: number of candidates in selection
    :return: n!/(k!(n-k)!)
    """
    if k > n - k:
        k = n - k
    result = 1
    for i in range(1, k + 1):
        result *= n - (k - i)
        result /= i
    return result


@lru_cache
def get_pascals(num: int) -> NDArray[(Any,), float]:
    """
    One line of Pascal's triangle.

    :param num: number of terms
    :return:
        1 -> 1
        2 -> 1, 1
        3 -> 1, 2, 1
        4 -> 1, 3, 3, 1
        ...
    """
    mid = sum(divmod(num, 2))
    left = [1] + [binom(num - 1, x) for x in range(1, mid)]
    # noinspection PyTypeChecker
    return np.array(left + left[: num - mid][::-1], dtype=float)


@lru_cache
def get_mix_matrix(num: int) -> NDArray[(Any, Any), float]:
    """
    Matrix of binomial coefficients for Bezier calculation.

    :param num: how many points in the Bezier curve
    :return: (num, num) matrix of binomial coefficients
    """
    mix = [np.append(get_pascals(x), [0] * (num - x)) for x in range(1, num + 1)]
    mix = get_pascals(num).reshape([-1, 1]) * mix
    mix *= _get_checkerboard(mix.shape, 1, -1)
    return mix
