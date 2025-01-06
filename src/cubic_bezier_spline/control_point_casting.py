"""Functions for casting control points to different types.

:author: Shay Hill
:created: 2023-02-08
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Annotated, Union

    import numpy.typing as npt

    Points = Union[Sequence[Sequence[float]], npt.NDArray[np.floating[Any]]]
    APoints = Annotated[npt.NDArray[np.floating[Any]], "(-1, -1)"]
    TPoints = tuple[tuple[float, ...], ...]

_TWO = 2


def as_points_array(points: Points) -> APoints:
    """Convert any 2D nested sequence of floats into an array of floats.

    :param points: a 2d sequence of floats
    :return: True if x is a 2D shape
    :raises ValueError: if x is not a 2D shape
    """
    apoints = np.asarray(points).astype(float)
    if apoints.ndim != _TWO:
        msg = f"Expected 2D array or nested sequence, got {apoints.ndim}"
        raise ValueError(msg)
    return apoints


def as_nested_tuple(points: Points) -> TPoints:
    """Convert any 2D nested sequence of floats into a tuple of tuples.

    :param points: a 2d sequence of floats
    :return: ((x0, y0, ...), (x1, y1, ...), ...)
    """
    apoints = as_points_array(points)
    return tuple(tuple(float(x) for x in y) for y in apoints)


def open_loop(apoints: APoints) -> APoints:
    """Open the loop by removing the last point if it matches the first.

    :param apoints: 2d array of points
    :return: 2d array of points where p[0] != p[-1]
    """
    if len(apoints) < _TWO:
        return apoints
    if np.allclose(apoints[0], apoints[-1]):
        return np.delete(apoints, -1, axis=0)
    return apoints


def close_loop(apoints: APoints) -> APoints:
    """Open the loop by removing the last point if it matches the first.

    :param apoints: 2d array of points
    :return: 2d array of points where p[0] == p[-1]
    """
    if not np.allclose(apoints[0], apoints[-1]):
        return np.append(apoints, [apoints[0]], axis=0)
    return apoints


def as_open_points_array(points: Points) -> APoints:
    """Convert any 2D nested sequence of floats into an array of floats.

    :param points: a 2d sequence of floats
    :return: True if x is a 2D shape
    """
    return open_loop(as_points_array(points))


def as_closed_points_array(points: Points) -> APoints:
    """Convert any 2D nested sequence of floats into an array of floats.

    :param points: a 2d sequence of floats
    :return: True if x is a 2D shape
    """
    return close_loop(as_points_array(points))
