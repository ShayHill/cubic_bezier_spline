"""An ugly little script to produce the knot test images.

This won't run without installing svg_ultralight.py in the environment.

:author: Shay Hill
:created: 2023-02-08
"""


from svg_ultralight import (
    new_element,
    new_sub_element,
    new_svg_root,
    write_svg,
    write_png_from_svg,
)

from cubic_bezier_spline import (
    new_closed_approximating_spline,
    new_closed_interpolating_spline,
    new_open_approximating_spline,
    new_open_interpolating_spline,
)

from pathlib import Path

_THIS_DIR = Path(__file__).parent
_INKSCAPE = Path(r"C:\Program Files\Inkscape\bin\inkscape")


pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
dbl_pts = sum([[p, p] for p in pts], [])
tri_pts = sum([[p, p, p] for p in pts], [])


cp_stroke = {
    "stroke_width": 0.01,
    "stroke": "blue",
    "fill": "none",
    "stroke_dasharray": "0.01, 0.01",
    "opacity": "0.5",
}


cpt_labels = ["single", "double", "triple"]
spl_labels = ["open approx", "closed approx", "open interp", "closed interp"]


def draw_examples(infix: str):
    width_per = 1
    height_per = 1.4
    head_room = height_per - width_per
    line_height = 0.08
    pad = 0.4
    full_width = width_per * 3 + pad * 2
    full_height = height_per * 4 + pad * 2 + head_room
    svg = new_svg_root(
        0, -head_room, full_width, full_height, print_width_=1000, pad_=pad * 250
    )
    for i, cptp in enumerate((pts, dbl_pts, tri_pts)):
        oa = new_open_approximating_spline(cptp)
        ca = new_closed_approximating_spline(cptp)
        oi = new_open_interpolating_spline(cptp)
        ci = new_closed_interpolating_spline(cptp)
        for j, (spline, spl_label) in enumerate(zip((oa, ca, oi, ci), spl_labels)):
            cptp_str = f"{cptp}"
            if len(cptp_str) > 67:
                cptp_str = f"{cptp_str[:65]}...]"

            heading = f"{cpt_labels[i]} control points, {spl_label} spline"
            label = f"{cptp_str}"

            x_trans = i * (width_per + pad)
            y_trans = j * (height_per + pad)
            group = new_sub_element(
                svg, "g", transform=f"translate({x_trans}, {y_trans})"
            )
            if infix == "light":
                text_fill = "#000000"
            else:
                text_fill = "#ffffff"

            # add text label
            new_sub_element(
                group,
                "text",
                x=0,
                y=-head_room + line_height,
                text=heading,
                font_size=0.05,
                fill=text_fill,
            )
            new_sub_element(
                group,
                "text",
                x=0,
                y=-head_room + line_height * 2,
                text=label,
                font_size=0.03,
                fill=text_fill,
            )

            group.append(
                new_element(
                    "path",
                    d=spline.svg_data,
                    stroke_width=0.01,
                    stroke="#444444",
                    fill="none",
                )
            )

            # show input control points
            for p in cptp:
                group.append(
                    new_element(
                        "circle", cx=p[0], cy=p[1], r=0.03, fill="red", opacity=0.5
                    )
                )

            # show spline control points
            for curve in spline:
                for p in curve.control_points:
                    group.append(
                        new_element(
                            "circle", cx=p[0], cy=p[1], r=0.02, fill="blue", opacity=0.5
                        )
                    )
                a, b, c, d = curve.control_points
                a_to_b = f"M {a[0]} {a[1]} L {b[0]} {b[1]}"
                c_to_d = f"M {c[0]} {c[1]} L {d[0]} {d[1]}"
                new_sub_element(group, "path", d=a_to_b, **cp_stroke)
                new_sub_element(group, "path", d=c_to_d, **cp_stroke)

            svg_filename = _THIS_DIR / f"test_knot_{infix}.svg"
            write_svg(svg_filename, svg)
            write_png_from_svg(_INKSCAPE, svg_filename)


if __name__ == "__main__":
    draw_examples("light")
    draw_examples("dark")
