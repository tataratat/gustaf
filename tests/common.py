"""gustaf/test/common.py

Common imports/routines needed for testing.
"""

import unittest
from collections import namedtuple

import numpy as np

import gustaf as gus

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
                [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7],
                [4, 5], [4, 6], [5, 7], [6, 7]
        ],
        dtype=np.int32,
)

TF = np.array(
        [
                [1, 0, 2], [0, 1, 5], [1, 3, 7], [3, 2, 6], [2, 0, 4],
                [4, 5, 7], [2, 3, 1], [5, 4, 0], [7, 5, 1], [6, 7, 3],
                [4, 6, 2], [7, 6, 4]
        ],
        dtype=np.int32,
)

QF = np.array(
        [
                [1, 0, 2, 3], [0, 1, 5, 4], [1, 3, 7, 5], [3, 2, 6, 7],
                [2, 0, 4, 6], [4, 5, 7, 6]
        ],
        dtype=np.int32,
)

TV = np.array(
        [
                [0, 2, 7, 3], [0, 2, 6, 7], [0, 6, 4, 7], [5, 0, 4, 7],
                [5, 0, 7, 1], [7, 0, 3, 1]
        ],
        dtype=np.int32,
)

HV = np.array([[0, 1, 3, 2, 4, 5, 7, 6]], dtype=np.int32)

CTPS_2D = np.array(
        [
                [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 2], [2, 1],
                [3, 2], [4, 1]
        ],
        dtype=np.float64,
)

KV_2D = [[0, 0, 0, .3, .7, 1, 1, 1], [0, 0, 1, 1]]

DEGREES_2D_NU = [2, 1]
DEGREES_2D_U = [4, 1]

WEIGHTS_2D = np.array(
        [[1.0], [0.8], [1.0], [0.8], [1.0], [1.0], [0.8], [1.0], [0.8], [1.0]],
        dtype=np.float64,
)
