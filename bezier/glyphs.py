#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Create curves and glyphs

:author: Shay Hill
:created: 1/27/2020
"""

import numpy as np
from typing import Callable, List, Union
from nptyping import Array

from .bezier_curve import BezierCurve
from ..ttf_extractor import extract_glyphs

Glyph = List[BezierCurve]
PointTransformation = Callable[[Array[float]], Array[float]]

def _one_glyph(glyphs, kerning, text):
    cursor = 0
    prev_char = None
    text_path = []
    for char in text:
        cursor += kerning.get((prev_char, char), 0)
        if glyphs[char].path is not None:
            glyph = glyphs[char].path
            text_path += map_over_glyph(lambda x: x + (cursor, 0), glyph)
        cursor += glyphs[char].wide
    return text_path

def glyph_from_string(font: str, text: Union[str, List[str]]):
    """
    Combine paths for each character into one path, apply spacing and kerning.

    :param font: path to a ttf font file
    :param text: the string to get paths for
    :return: one path representing entire text
    """
    just_one = False
    if isinstance(text, str):
        just_one = True
        text = [text]

    glyphs, kerning = extract_glyphs(font, ''.join(text))
    string_glyphs = [_one_glyph(glyphs, kerning, x) for x in text]
    if just_one:
        return string_glyphs[0]
    return string_glyphs




def map_over_glyph(transformation: PointTransformation, glyph: Glyph) -> Glyph:
    """
    Apply a transformation to each point in a list of Bezier curves.

    :param transformation: function taking and returning a point (np.ndarray)
    :param glyph: a list of Bezier curves
    :return: glyph with transformation applied to each point
    """
    return [BezierCurve(*map(transformation, curve)) for curve in glyph]


