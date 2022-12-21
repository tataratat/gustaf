"""gustaf/test/common.py

Common imports/routines needed for testing.
"""

import pytest

import numpy as np

import gustaf as gus

# # try to satisfy flake8 F401
# __all__ = [
#         "unittest",
#         "np",
#         "V",
#         "E",
#         "TF",
#         "QF",
#         "TV",
#         "HV",
#         "CPS_2D",
#         "KVS_2D",
#         "DEGREES_2D_NU",
#         "DEGREES_2D_U",
#         "WEIGHTS_2D",
# ]

# Mesh Stuff


@pytest.fixture
def vertices_3d():
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
    return V


@pytest.fixture
def edges():
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
                    [6, 7],
            ],
            dtype=np.int32,
    )
    return E


@pytest.fixture
def faces_tetra():
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
                    [7, 6, 4],
            ],
            dtype=np.int32,
    )
    return TF


@pytest.fixture
def faces_quad():
    QF = np.array(
            [
                    [1, 0, 2, 3],
                    [0, 1, 5, 4],
                    [1, 3, 7, 5],
                    [3, 2, 6, 7],
                    [2, 0, 4, 6],
                    [4, 5, 7, 6],
            ],
            dtype=np.int32,
    )
    return QF


@pytest.fixture
def volume_tetra():
    TV = np.array(
            [
                    [0, 2, 7, 3],
                    [0, 2, 6, 7],
                    [0, 6, 4, 7],
                    [5, 0, 4, 7],
                    [5, 0, 7, 1],
                    [7, 0, 3, 1],
            ],
            dtype=np.int32,
    )
    return TV


@pytest.fixture
def volume_hex():
    HV = np.array([[0, 1, 3, 2, 4, 5, 7, 6]], dtype=np.int32)
    return HV


@pytest.fixture
def control_points_2d():
    CPS_2D = np.array(
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
                    [4, 1],
            ],
            dtype=np.float64,
    )
    return CPS_2D


@pytest.fixture
def knot_vector_2d():
    KVS_2D = [[0, 0, 0, .3, .7, 1, 1, 1], [0, 0, 1, 1]]
    return KVS_2D


@pytest.fixture
def degrees_2d_nu():
    DEGREES_2D_NU = [2, 1]
    return DEGREES_2D_NU


@pytest.fixture
def degrees_2d_u():
    DEGREES_2D_U = [4, 1]
    return DEGREES_2D_U


@pytest.fixture
def weights_2d():
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
                    [1.0],
            ],
            dtype=np.float64,
    )
    return WEIGHTS_2D


@pytest.fixture
def bspline_2d(control_points_2d, degrees_2d_nu, knot_vector_2d):
    return gus.BSpline(
            control_points=control_points_2d,
            degrees=degrees_2d_nu,
            knot_vectors=knot_vector_2d
    )
