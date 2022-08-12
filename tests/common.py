"""gustaf/test/common.py

Common imports/routines needed for testing.
"""

import unittest
from collections import namedtuple

import numpy as np

import gustaf as gus

# gus.utils.log.configure(debug=True)

# Mesh Stuff
V = np.array(
    [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.],
        [0., 0., 1.],
        [1., 0., 1.],
        [0., 1., 1.],
        [1., 1., 1.],
    ],
    dtype=np.float64,
)

E = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7]
    ],
    dtype=np.int32,
)

TF = np.array(
    [
        [1, 0, 2],
        [0, 1, 5],
        [1, 3, 7],
        [3, 2, 6],
        [2, 0, 4],
        [4, 5, 7],
        [2, 3, 1],
        [5, 4, 0],
        [7, 5, 1],
        [6, 7, 3],
        [4, 6, 2],
        [7, 6, 4]
    ],
    dtype=np.int32,
)

QF = np.array(
    [
        [1, 0, 2, 3],
        [0, 1, 5, 4],
        [1, 3, 7, 5],
        [3, 2, 6, 7],
        [2, 0, 4, 6],
        [4, 5, 7, 6]
    ],
    dtype=np.int32,
)

TV = np.array(
    [
        [0, 2, 7, 3],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [5, 0, 4, 7],
        [5, 0, 7, 1],
        [7, 0, 3, 1]
    ],
    dtype=np.int32,
)

HV = np.array(
    [[0, 1, 3, 2, 4, 5, 7, 6]],
    dtype=np.int32
)


# class obj.
VERT_MUSTER = gus.Vertices(
    vertices=V
)

EDGE_MUSTER = gus.Edges(
    vertices=V,
    edges=E,
)

TRI_MUSTER = gus.Faces(
    vertices=V,
    faces=TF,
)
TET_MUSTER = gus.Volumes(
    vertices=V,
    volumes=TV,
)
QUAD_MUSTER = gus.Faces(
    vertices=V,
    faces=QF,
)
HEXA_MUSTER = gus.Volumes(
    vertices=V,
    volumes=HV,
)


# Spline Stuff
B1P2D_MUSTER = gus.BSpline(
    degrees=None,
    knot_vectors=None,
    control_points=None,
)

CTPS_2D = np.array(
    [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [0, 1],
        [1, 2],
        [2, 1],
        [3, 2],
        [4, 1]
    ],
    dtype=np.float64,
)

KV_2D = [[0, 0, 0, .3, .7, 1, 1, 1], [0, 0, 1, 1]]

DEGREES_2D_NU = [2, 1]
DEGREES_2D_U = [4, 1]

WEIGHTS_2D = np.array(
    [
        [1.0],
        [0.8],
        [1.0],
        [0.8],
        [1.0],
        [1.0],
        [0.8],
        [1.0],
        [0.8],
        [1.0]
    ],
    dtype=np.float64,
)


# alias
logconfig = gus.utils.log.configure

COMMON = namedtuple(
    "COMMON",
    [
        "ut",
        "np",
        "gus",
        "v",
        "e",
        "tf",
        "qf",
        "tv",
        "hv",
        "vert",
        "edge",
        "tri",
        "tet",
        "quad",
        "hexa",
        "ctps2d",
        "kv2d",
        "degrees2dnu",
        "degrees2du",
        "weights2s"
    ]
)

C = COMMON(
    ut=unittest,
    np=np,
    gus=gus,
    v=V,
    e=E,
    tf=TF,
    qf=QF,
    tv=TV,
    hv=HV,
    vert=VERT_MUSTER,
    edge=EDGE_MUSTER,
    tri=TRI_MUSTER,
    tet=TET_MUSTER,
    quad=QUAD_MUSTER,
    hexa=HEXA_MUSTER,
    ctps2d=CTPS_2D,
    kv2d=KV_2D,
    degrees2dnu=DEGREES_2D_NU,
    degrees2du=DEGREES_2D_U,
    weights2s=WEIGHTS_2D
)
