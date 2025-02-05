"""Display a series of corner-drawing techniques.

:author: Shay Hill
:created: 2025-02-04
"""

import itertools as it
import math
from typing import Any

import numpy as np
import svg_ultralight as su
import vec2_math as v2
from lxml.etree import _Element as EtreeElement  # type: ignore
from numpy import typing as npt
from svg_ultralight.string_conversion import format_numbers_in_string

from cubic_bezier_spline import BezierSpline, new_open_approximating_spline

STROKE_WIDTH = 0.005

Vector = npt.ArrayLike

Vec2 = tuple[float, float]

pnt_sets: list[tuple[Vec2, Vec2, Vec2]] = []

for x in range(0, 25, 3):
    # for x in [10]:
    yy = -x / 30 + 0.25
    # yy = 0
    pnt_sets.append(
        (
            (0, yy),
            (1, yy),
            (x / 10, 1),
        )
    )


xs, ys = zip(*it.chain(*pnt_sets))
min_x = min(xs)
max_x = max(xs)
min_y = min(ys)
max_y = max(ys)

root = su.new_svg_root(
    x_=min_x, y_=min_y, width_=max_x - min_x, height_=max_y - min_y, pad_=0.2
)


def new_path(data: str, **kwargs: str | float) -> EtreeElement:
    return su.new_sub_element(
        root,
        "path",
        d=data,
        fill="none",
        stroke_width=STROKE_WIDTH,
        # opacity=0.7,
        **kwargs,
    )


def move_toward(
    pnt_a: npt.ArrayLike, pnt_b: npt.ArrayLike, dist: float
) -> npt.NDArray[np.floating[Any]]:
    """Move from pnt_a toward pnt_b by dist."""
    delta = np.subtract(pnt_b, pnt_a)
    norm = np.linalg.norm(delta)
    return np.add(pnt_a, np.multiply(delta, dist / norm))


def new_line(
    pnt_a: tuple[float, float], pnt_b: tuple[float, float], **kwargs: str | float
) -> EtreeElement:
    x1, y1 = pnt_a
    x2, y2 = pnt_b
    return su.new_sub_element(
        root, "line", x1=x1, y1=y1, x2=x2, y2=y2, stroke_width=STROKE_WIDTH, **kwargs
    )


def double_gap_interpolating(
    pnt_a: tuple[float, float], pnt_b: tuple[float, float], pnt_c: tuple[float, float]
):
    """Draw a corner by gapping the edges twice."""
    angle = abs(v2.get_signed_angle(v2.vsub(pnt_a, pnt_b), v2.vsub(pnt_c, pnt_b)))

    rad = 0.25

    tangent_length = rad / math.tan(angle / 2)
    control_point_scalar = 1 / (2 * math.sin(angle / 2) + 1)

    cpts = [
        move_toward(pnt_b, pnt_a, tangent_length),
        move_toward(pnt_b, pnt_a, tangent_length * control_point_scalar),
        move_toward(pnt_b, pnt_c, tangent_length * control_point_scalar),
        move_toward(pnt_b, pnt_c, tangent_length),
    ]

    cpt_edges = zip(cpts, cpts[1:])
    print([v2.get_norm(v2.vsub(b, a)) for a, b in cpt_edges])

    # print(cpts)
    _ = new_line(pnt_a, cpts[0], stroke="black")
    _ = new_line(cpts[3], pnt_c, stroke="black")

    spline = new_open_approximating_spline(cpts)
    _ = new_path(spline.svg_data, stroke="blue")
    spline = BezierSpline([cpts])
    _ = new_path(spline.svg_data, stroke="orange")
    arc_d = format_numbers_in_string(
        f"M {cpts[0][0]}, {cpts[0][1]} A {rad},{rad} 0 0 1 {cpts[3][0]},{cpts[3][1]}"
    )
    _ = new_path(arc_d, stroke="green")


if __name__ == "__main__":
    for aa, bb, cc in pnt_sets:
        double_gap_interpolating(aa, bb, cc)
    _ = su.write_svg("corner_test.svg", root)
