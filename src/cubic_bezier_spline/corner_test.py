"""Display a series of corner-drawing techniques.

:author: Shay Hill
:created: 2025-02-04
"""

import math
import svg_ultralight as su
from svg_ultralight.string_conversion import format_numbers_in_string
import vec2_math as v2
from offset_poly import offset_polyline
from offset_poly.offset_corner import GapCorner
from cubic_bezier_spline import (
    new_open_interpolating_spline,
    new_open_linear_spline,
    new_open_approximating_spline,
    BezierSpline,
)
from lxml.etree import _Element as EtreeElement  # type: ignore


import itertools as it


STROKE_WIDTH = 0.005

Vec2 = tuple[float, float]

pnt_sets: list[tuple[Vec2, Vec2, Vec2]] = []

for x in range(0, 25, 3):
# for x in [10]:
    yy = -x / 30 + .25
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

    num = math.sin(angle)
    den = math.sin(math.pi - angle / 2)
    # print(f"angle: {angle}, num: {num}, den: {den}, num/den: {num/den}")

    den = math.sin(math.pi - angle / 2)
    dip = 0.25
    # dip = 1

    poly = [pnt_a, pnt_b, pnt_c]
    gapped_out = GapCorner(*poly, dip, dip)
    isos = v2.get_norm(v2.vsub(gapped_out.cpts[0], gapped_out.cpts[1]))
    base = v2.get_norm(v2.vsub(gapped_out.cpts[0], gapped_out.cpts[2]))
    scalar = base / isos + 1
    # print(isos)
    # print(base)

    print(f"isos: {isos}, base: {base}, scalar: {scalar}")
    ang_c = (math.pi - angle) / 2
    test_val = dip * math.sin(ang_c) / math.sin(angle / 2)
    test_val2 = dip * math.sin(ang_c) * 2

    test_val3 = math.sin(angle / 2) + 1
    print(f"test: {test_val}, test: {test_val2}, test: {test_val3}")
    # print(f"test: {test_val2}")

    dip1 = dip / (1 + num / den)
    dip1 = dip / (1 + pow(2, 0.5))
    dip1 = dip / scalar
    dip2 = dip
    # print(dip1, dip2)
    gap_1 = offset_polyline(poly, dip1)[1].cpts
    gap_2 = offset_polyline(poly, dip2)[1].cpts
    # print(f"gap_1: {gap_1}")
    # print(f"gap_2: {gap_2}")
    cpts = [gap_2[0], gap_1[0], gap_1[2], gap_2[2]]
    # print(cpts)

    # print(v2.get_norm(v2.vsub(cpts[0], pnt_b)))
    # print(v2.get_norm(v2.vsub(cpts[0], cpts[3])))
    # print(dip/math.sin(angle))
    # print(angle / math.pi)

    cpt_edges = zip(cpts, cpts[1:])
    print([v2.get_norm(v2.vsub(b, a)) for a, b in cpt_edges])

    # print(cpts)
    _ = new_line(pnt_a, cpts[0], stroke="black")
    _ = new_line(cpts[3], pnt_c, stroke="black")
    spline = new_open_approximating_spline(gap_2)
    _ = new_path(spline.svg_data, stroke="red")
    spline = new_open_approximating_spline(cpts)
    _ = new_path(spline.svg_data, stroke="blue")
    spline = BezierSpline([cpts])
    _ = new_path(spline.svg_data, stroke="orange")
    arc_d = format_numbers_in_string(
        f"M {cpts[0][0]}, {cpts[0][1]} A {dip},{dip} 0 0 1 {cpts[3][0]},{cpts[3][1]}"
    )
    _ = new_path(arc_d, stroke="green")


if __name__ == "__main__":
    for aa, bb, cc in pnt_sets:
        double_gap_interpolating(aa, bb, cc)
    _ = su.write_svg("corner_test.svg", root)
