Bezier curves and splines (composite Bezier curves). Substantially complete.

I've written a lot of Bezier code. This project is a little different, because I stick to non-rational Bezier. This is
for working with SVG, not--for once--some exercise in completism.

Uses Numpy matrix math, so it should be relatively fast, but this may not be production ready. Feel free to learn from
and fork this project (that's why I made it public), but I will most likely not respond to issues or feature requests.

As of now, this will
    * Evaluate, differentiate, and split Bezier curves and splines
    * Construct and evaluate Bezier splines (open and closed, approximating and interpolating)

See
    * bezier_curve for a BezierCurve object
    * bezier_spline for a BezierSpline object
    * construct_splines for functions to assemble open/closed, approximating/interpolating Bezier splines

Most of the math can be found in:
    * A Primer on Bezier Curves
      https://pomax.github.io/bezierinfo/
    * UCLS-Math-149-Mathematics-of-Computer-Graphics-lecture-notes
      https://www.stkent.com/assets/pdfs/UCLA-Math-149-Mathematics-of-Computer-Graphics-lecture-notes.pdf

Contains Matrix, De Casteljau, and Bezier Basis solvers. These are equivalent, but I'd rather test against other
methods than tediously compare return values to hand-calculated results.

