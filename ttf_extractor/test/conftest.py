#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Allow tests to run from command line (python -m pytest ...) when ttf_extractor is
cloned into another project.

:author: Shay Hill
:created: 1/23/2020
"""

import sys
from pathlib import Path

project_root = str(Path(__file__, "../..").resolve())
if project_root not in sys.path:
    sys.path.append(project_root)
