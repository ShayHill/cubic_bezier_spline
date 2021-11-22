## Non-rational Bezier curves and splines (composite Bezier curves)

This package exists mostly to create C2-continuous, non-rational cubic Bezier splines. In other words, this will approximate or interpolate
a sequence of points into a sequence of non-rational cubic Bezier curves.

Should be relatively fast, but this may not be production ready. Feel free to learn from and fork this project (that's why I made it public), but I will most likely not respond to issues or feature requests. For me, this is a helper tool to create svg files, not an exercise in completism. As such

### this package will

* Evaluate, differentiate, elevate, and split non-rational Bezier curves of any degree
* Construct non-rational cubic Bezier splines (open and closed, approximating and interpolating)
* Evaluate and differentiate non-rational Bezier splines of any degree

### this package will not**

* Work with rational Bezier splines, b-splines, NURBS, or any other generalization of Bezier curves
* Decrease curve degree
* Approximate curve intersections
* Approximate the length of a curve
* "Stroke" (move left or right) a curve<br/>

** much of the above can be found here: https://github.com/dhermes/bezier

### Public classes / functions

    # control_points -> array.shape(j, k) where
    #     j is number of control points
    #     k is any number of dimensions (x, y, z, etc.)

    BezierCurve(control_points: NDArray[(Any, Any), float])

    __call__ (time: float, derivative: int=0)
    .elevated (to_degree: int)
    .derivative (derivative: int)
    .split (at_time: float)

<br/>

    # Spline control points: An array.shape = (i, j, k) where
    #     i is number of curve-control-point instances (e.g., four each for cubic curves)
    #     j is number of control points in each spline
    #     k is any number of dimensions (x, y, z, etc.)

    BezierSpline(control_points: NDArray[(Any, Any, Any), float])

    __call__(time, derivative: int=0)

<br/>

    get_approximating_spline(control_points: NDArray[(Any, Any), float], close: bool)

    get_interpolating_spline(control_points: NDArray[(Any, Any), float], close: bool)

Most of the math can be found in:

* A Primer on Bezier Curves<br/>
https://pomax.github.io/bezierinfo/
* UCLS-Math-149-Mathematics-of-Computer-Graphics-lecture-notes<br/>
https://www.stkent.com/assets/pdfs/UCLA-Math-149-Mathematics-of-Computer-Graphics-lecture-notes.pdf

