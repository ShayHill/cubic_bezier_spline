"""Type hints for the cubic_bezier_spline project.

:author: Shay Hill
:created: 2023-02-08
"""

from typing import Annotated, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

FArray: TypeAlias = npt.NDArray[np.float_]
IArray: TypeAlias = npt.NDArray[np.int_]

# acceptable point definition args
Points = Sequence[Sequence[float]] | FArray
Point = Sequence[float] | FArray

# point arrays used internally
APoints = Annotated[FArray, "(-1, -1)"]
APoint = Annotated[FArray, "(-1,)"]

TPoints = tuple[tuple[float, ...], ...]
