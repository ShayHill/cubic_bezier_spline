#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Type hints for project

:author: Shay Hill
:created: 1/13/2020
"""

from typing import Tuple
from nptyping import Array

Point = Array[float, 2]
Curve = Tuple[Point, Point, Point, Point]
