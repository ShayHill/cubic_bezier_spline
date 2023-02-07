#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
""" Helper code: Create matrices for Bezier operations

:author: Shay Hill
:created: 10/2/2020
"""
from functools import lru_cache
from typing import Any

import numpy as np  # type: ignore
import numpy.typing as npt # type: ignore

FArray = npt.NDArray[np.float_]


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
def get_pascals(num: int) -> FArray:
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
def get_mix_matrix(num: int) -> Farray:
    """
    Matrix of binomial coefficients for Bezier calculation.

    :param num: how many points in the Bezier curve
    :return: (num, num) matrix of binomial coefficients
    """
    mix = [np.append(get_pascals(x), [0] * (num - x)) for x in range(1, num + 1)]
    mix = get_pascals(num).reshape([-1, 1]) * mix
    check = np.sum(np.indices(mix.shape), axis=0) % 2 == 1
    np.negative(mix, out=mix, where=check)
    return mix
