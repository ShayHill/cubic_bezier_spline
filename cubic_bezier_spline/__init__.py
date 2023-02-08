"""Import all the functions from construct_splines to top level.

:author: Shay Hill
:created: 2023-02-08
"""
from .construct_splines import (
    get_closed_approximating_spline,
    get_closed_interpolating_spline,
    get_open_approximating_spline,
    get_open_interpolating_spline,
)

__all__ = [
    "get_open_approximating_spline",
    "get_closed_approximating_spline",
    "get_open_interpolating_spline",
    "get_closed_interpolating_spline",
]
