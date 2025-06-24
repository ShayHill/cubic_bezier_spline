"""A pairwise function until I drop 3.9 compatibility.

:author: Shay Hill
:created: 2025-06-23
"""

from __future__ import annotations

import itertools as it
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable


_T = TypeVar("_T")


def pairwise(iterable: Iterable[_T]) -> Iterable[tuple[_T, _T]]:
    """Yield pairs of items from an iterable.

    :param iterable: items to pair
    :return: pairs of items from the iterable

    No it.pairwise in Python 3.9.
    """
    items_a, items_b = it.tee(iterable)
    _ = next(items_b, None)
    return zip(items_a, items_b)
