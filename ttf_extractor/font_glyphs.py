#!/usr/bin/env python3 as et
# _*_ coding: utf-8 _*_
"""Extract a fonts_ttf, reformat into something Python can understand

:author: Shay Hill
:created: 1/10/2020
"""

import os
import string
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, IO, List, NamedTuple, Optional, Tuple, Union
from xml.etree import ElementTree as et

from nptyping import Array

from .path_converter import svg_to_bezier, Curve

WHITESPACE = set(string.whitespace) - {" "}

FONTFORGE = r"C:\Program Files (x86)\FontForgeBuilds\bin\fontforge.exe"


@dataclass
class Glyph:
    name: str
    wide: float
    path: Union[List[Curve], None]

    def __init__(self, xml_glyph: et.Element) -> None:
        self.name = xml_glyph.attrib["glyph-name"]
        self.wide = float(xml_glyph.attrib.get("horiz-adv-x", 0))
        path = xml_glyph.attrib.get("d")
        if path is None:
            self.path = None
        else:
            self.path = svg_to_bezier(path)


Glyphs = Dict[str, Glyph]
Kerning = Dict[Tuple[str, str], float]
CBezPts = Tuple[Array[int, 2], Array[int, 2], Array[int, 2], Array[int, 2]]


class FontExtraction(NamedTuple):
    glyphs: Glyphs
    kerning: Kerning


def _create_temp_svg_file(font_name: str, fontforge_exe: str) -> IO:
    """
    Convert ttf to svg. Write to temporary file. DON'T FORGET TO UNLINK THIS FILE!

    :param font_name: name of font with or without extension

    :param fontforge_exe:
    path to fontforge executable. This defaults to the path on my Laptop. MIGHT work
    an another system if that system's path to fontforge.exe is passed. Might
    not. I'm not going to test it.

    :effects: creates font_name.svg in ``ttf_extractor/fonts_svg``
    """
    to_ttf = font_name
    to_svg = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    to_svg.close()
    command = (
        f'"{fontforge_exe}" -c '
        f'"f = fontforge.open(argv[1]); f.generate(argv[2])" '
        f'"{to_ttf}" '
        f'"{to_svg.name}"'
    )
    subprocess.check_call(command, shell=True)
    return to_svg


@lru_cache(maxsize=1)
def _get_root(path_to_xml: str) -> et.Element:
    """
    Convert svg text into an ElementTree.Element root

    :param path_to_xml: path to svg file
    :return: root element
    """
    with open(path_to_xml, "r") as svg_file:
        return et.fromstring(svg_file.read())


@lru_cache(maxsize=1)
def _get_nsmap(path_to_xml: str) -> Dict[str, str]:
    """
    Map namespace names to addresses.

    :param path_to_xml: path to an xml file
    :return: {'': 'http://...', 'name': 'http://...', ...}

    xml can search and set with these, i.e.
    ``element.find("glyph", nsmap)``
    ``element.find("name:glyph", nsmap)``

    This solution will probably only work with xml, not lxml
    """
    return {x: y for _, (x, y) in et.iterparse(path_to_xml, events=["start-ns"])}


def _get_root_and_nsmap(path_to_svg: str) -> Tuple[et.Element, Dict[str, str]]:
    """
    XML root and nsmap from an xml (presumably svg) file.

    :param path_to_svg: name of a font in fonts_svg (with or w/o extension)
    :return: xml root and nsmap

    Look for font_name.svg in fonts_svg. Extract xml and nsmap
    """
    return _get_root(path_to_svg), _get_nsmap(path_to_svg)


@lru_cache(maxsize=1)
def get_glyphs(path_to_svg: str, required: Union[None, str]) -> Glyphs:
    """
    Map unicode characters to Glyph instances.

    :param path_to_svg: filename of fonts_ttf in fonts_svg folder (with or w/o extension)
    :param required: optional (but strongly encouraged) subset of required characters

        If this argument is not given, will convert and return every glyph in the
        font. That could take several seconds.

        Examples:
            * get_glyphs('Helvetica', 'This is the string I am interested in.')
            * get_glyphs('Helvetica', string.ascii_lowercase)
            * get_glyphs('Helvetica', string.printable)

        The last example only works because get_glyphs will ignore[1] whitespace
        characters except 'space'. A font is not expected to have tab,
        line tabulation, form feed, etc., even though these characters are contained
        in string.printable.

    :return: unicode characters mapped to Glyph instances.

    [1] What else this function ignores: everything except unicode characters,
    so no ligatures.
    """
    xml, nsmap = _get_root_and_nsmap(path_to_svg)
    unicode = {x.attrib.get("unicode"): x for x in xml.iterfind(".//glyph", nsmap)}

    if required is not None:
        required = set(required) - WHITESPACE
        missing = required - unicode.keys()
        if missing:
            raise ValueError(f"font has no glyphs for {missing}")
    else:
        required = unicode.keys()

    glyphs = {}
    for char in required:
        glyphs[char] = Glyph(unicode[char])
    return glyphs


def get_kerning(path_to_svg: str, glyphs: Glyphs) -> Kerning:
    """
    Map character pairs to kerning distances.

    :param path_to_svg: filename of fonts_ttf in fonts_svg folder (with or w/o extension)
    :param glyphs: the return value from get_glyphs. This allows the function to trim
    unnecessary kerning information.
    :return: i.e: {('a', 'b'): 35, ('a', 'c'): 14}

    Takes the name of an svg file converted by FontForge.

    Extracts:
        A dictionary of character tuples to kerning distances.
        {(A, Y): -40}
    """
    xml, nsmap = _get_root_and_nsmap(path_to_svg)

    hkerns = {}
    for hkern in xml.iterfind(".//hkern", nsmap):
        u1 = hkern.attrib.get("u1", None)
        u2 = hkern.attrib.get("u2", None)
        if u1 and u2 and u1 in glyphs and u2 in glyphs:
            hkerns[(u1, u2)] = float(hkern.attrib["k"])
    return hkerns


def extract_glyphs(
    path_to_ttf: str, required: Optional[str] = None, fontforge_exe: str = FONTFORGE
) -> FontExtraction:
    """
    #TODO: document.
    :param path_to_ttf:
    :param required:
    :param fontforge_exe:
    :return:
    """
    temp_svg = _create_temp_svg_file(path_to_ttf, fontforge_exe)
    glyphs = get_glyphs(temp_svg.name, required)
    kerning = get_kerning(temp_svg.name, glyphs)
    os.unlink(temp_svg.name)
    return FontExtraction(glyphs, kerning)
