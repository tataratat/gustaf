"""gustav/test/common.py

Common imports/routines needed for testing.
"""

import unittest

import numpy as np

import gustav as gus

TRI_MUSTERMANN = gus.Mesh(
    vertices=None,
    faces=None,
)
TET_MUSTERMANN = gus.Mesh(
    vertices=None,
    elements=None,
)
QUAD_MUSTERMANN = gus.Mesh(
    vertices=None,
    faces=None,
)
HEXA_MUSTERMANN = gus.Mesh(
    vertices=None,
    elements=None,
)
B1P2D_MUSTERMANN = gus.BSpline(
    degrees=None,
    knot_vectors=None,
    control_points=None,
)
