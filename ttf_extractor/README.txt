200110 09:56:22 

A utility to extract and reformat TTF files to get the text data. This is
written so I can create outlined letters in POV-Ray with sphere sweeps, but I
may find other uses later.

Only one officially public function: extract_glyphs
This function will not work unless FontForge is installed.

Takes a ttf file and optional but recommended subset of required glyphs.
Returns a NamedTuple (
    glyphs: Glyph objects with path and width information,
    kerning: Unicode character pairs mapped to kerning distances
)

One might want to import path_converter.svg_to_bezier for testing or creative
purposes, because svg d strings really are a nice, concise way to define a
curve. Function svg_to_bezier will take 'M3 4 6 7' and return a list of
Bezier control-point tuples. In this case:
[(np.array([3, 4]), np.array([6, 7]))]

See full documentation in ttf_extractor.font_glyphs


