"""Helper code: Create matrices for Bezier operations.

:author: Shay Hill
:created: 10/2/2020
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@lru_cache(maxsize=128)
def binom(n: int, k: int) -> int:
    """Return n choose k.

    :param n: number of candidates
    :param k: number of candidates in selection
    :return: n!/(k!(n-k)!)
    """
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result *= n - (k - i)
        result //= i
    return result


@lru_cache
def get_pascals(num: int) -> npt.NDArray[np.floating[Any]]:
    """One line of Pascal's triangle.

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
    return np.array(left + left[: num - mid][::-1], dtype=float)


def _get_boolean_checkerboard(shape: tuple[int, int]) -> npt.NDArray[np.bool_]:
    """Create a checkerboard of True/False values.

    :param shape: (rows, cols)
    :return: (rows, cols) array of True/False values
    """
    checkerboard = np.sum(np.indices(shape), axis=0) % 2 == 1
    return np.array(checkerboard).astype(np.bool_)


@lru_cache
def get_mix_matrix(num: int) -> npt.NDArray[np.floating[Any]]:
    """Matrix of binomial coefficients for Bezier calculation.

    :param num: how many points in the Bezier curve
    :return: (num, num) matrix of binomial coefficients
    """
    mix = np.zeros((num, num), dtype=float)
    for i in range(1, num + 1):
        mix[i - 1, :i] = get_pascals(i)
    mix = get_pascals(num).reshape([-1, 1]) * mix
    rows, cols = mix.shape
    check = _get_boolean_checkerboard((rows, cols))
    return np.negative(mix, out=mix, where=check)
