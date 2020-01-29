#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Test functions in ttf_extractor.font_glyphs.py

:author: Shay Hill
:created: 1/22/2020
"""

import os
from pathlib import Path

from ttf_extractor.font_glyphs import _create_temp_svg_file, extract_glyphs

impact_font = Path(__file__, "../test_resources/impact.ttf").resolve()


class TestCreateFontSvg:
    def test_unlink(self) -> None:
        """Temp file is returned closed and linked"""
        temp_svg = _create_temp_svg_file(impact_font)
        open(temp_svg.name)
        temp_svg.close()
        os.unlink(temp_svg.name)


class TestExtractGlyphs:
    def test_impact(self) -> None:
        """Return glyphs and kerning for impact.ttf

        This test just runs it"""
        glyphs, kerning = extract_glyphs(impact_font, "Just these characters, please.")

