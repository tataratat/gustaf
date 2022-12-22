"""gustaf/test/common.py

Common imports/routines needed for testing.
"""

import pytest

import numpy as np

import gustaf as gus


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
def vertices(vertices_3d):
    return gus.Vertices(vertices_3d)


@pytest.fixture
def edge_connec():
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
def edges(vertices_3d, edge_connec):
    return gus.Edges(vertices_3d, edge_connec)


@pytest.fixture
def tri_connec():
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
def faces_tri(vertices_3d, tri_connec):
    return gus.Faces(vertices_3d, tri_connec)


@pytest.fixture
def quad_connec():
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
def faces_quad(vertices_3d, quad_connec):
    return gus.Faces(vertices_3d, quad_connec)


@pytest.fixture
def tet_connec():
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
def volumes_tet(vertices_3d, tet_connec):
    return gus.Volumes(vertices_3d, tet_connec)


@pytest.fixture
def hexa_connec():
    HV = np.array([[0, 1, 3, 2, 4, 5, 7, 6]], dtype=np.int32)
    return HV


@pytest.fixture
def volumes_hexa(vertices_3d, hexa_connec):
    return gus.Volumes(vertices_3d, hexa_connec)


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


@pytest.fixture
def nurbs_2d(control_points_2d, degrees_2d_nu, knot_vector_2d, weights_2d):
    return gus.NURBS(
            control_points=control_points_2d,
            degrees=degrees_2d_nu,
            knot_vectors=knot_vector_2d,
            weights=weights_2d
    )


@pytest.fixture
def bezier_2d(control_points_2d, degrees_2d_u):
    return gus.Bezier(control_points=control_points_2d, degrees=degrees_2d_u)


@pytest.fixture
def rationalbezier_2d(control_points_2d, degrees_2d_u, weights_2d):
    return gus.RationalBezier(
            control_points=control_points_2d,
            degrees=degrees_2d_u,
            weights=weights_2d
    )


@pytest.fixture
def provide_data_to_unittest(
        request, vertices_3d, edge_connec, tri_connec, quad_connec, tet_connec,
        hexa_connec
):
    request.cls.V = vertices_3d
    request.cls.E = edge_connec
    request.cls.TF = tri_connec
    request.cls.QF = quad_connec
    request.cls.TV = tet_connec
    request.cls.HV = hexa_connec


@pytest.fixture
def sample_tet():
    return np.array([[1, 2, 3, 3], [4, 5, 6, 5], [7, 8, 9, 5]])


@pytest.fixture
def sample_hex_array():
    return np.array(
            [
                    [1, 2, 3, 3, 3, 3, 4, 5], [1, 2, 3, 3, 3, 3, 4, 5],
                    [1, 2, 3, 3, 3, 3, 4, 5], [1, 2, 3, 3, 3, 3, 4, 5]
            ]
    )


@pytest.fixture
def sample_faces_tri(sample_tet):
    return sample_tet


@pytest.fixture
def sample_faces_tet():
    return np.array([[1, 2, 3, 3], [4, 5, 6, 5], [7, 8, 9, 5], [7, 8, 9, 5]])


@pytest.fixture
def sample_hex_error():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])


@pytest.fixture
def sample_tri_error():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])


@pytest.fixture
def sample_hex():
    return [3, 4, 5]


@pytest.fixture
def sample_quad_faces():
    return [2, 4]


@pytest.fixture
def sample_hex_fail():
    return [0, 6, 7]


@pytest.fixture
def sample_quad_faces_fail():
    return [0, 55]


@pytest.fixture
def expected_tet_result():
    return [
            [1, 3, 2], [2, 3, 1], [3, 3, 2], [3, 3, 1], [4, 6, 5], [5, 5, 4],
            [6, 5, 5], [5, 6, 4], [7, 9, 8], [8, 5, 7], [9, 5, 8], [5, 9, 7]
    ]


@pytest.fixture
def expected_quad_result():
    return [
            [2, 1, 3, 3], [1, 2, 3, 3], [2, 3, 4, 3], [3, 3, 5,
                                                       4], [3, 1, 3, 5],
            [3, 3, 4, 5], [2, 1, 3, 3], [1, 2, 3, 3], [2, 3, 4,
                                                       3], [3, 3, 5, 4],
            [3, 1, 3, 5], [3, 3, 4, 5], [2, 1, 3, 3], [1, 2, 3,
                                                       3], [2, 3, 4, 3],
            [3, 3, 5, 4], [3, 1, 3, 5], [3, 3, 4, 5], [2, 1, 3,
                                                       3], [1, 2, 3, 3],
            [2, 3, 4, 3], [3, 3, 5, 4], [3, 1, 3, 5], [3, 3, 4, 5]
    ]


@pytest.fixture
def expected_edges_tri():
    return [
            [1, 2], [2, 3], [3, 3], [3, 1], [4, 5], [5, 6], [6, 5], [5, 4],
            [7, 8], [8, 9], [9, 5], [5, 7]
    ]


@pytest.fixture
def expected_edges_tet():
    return [
            [1, 2], [2, 3], [3, 3], [3, 1], [4, 5], [5, 6], [6, 5], [5, 4],
            [7, 8], [8, 9], [9, 5], [5, 7], [7, 8], [8, 9], [9, 5], [5, 7]
    ]


@pytest.fixture
def expected_hexa_volumes():
    return [
            [0, 1, 4, 3, 12, 13, 16, 15], [1, 2, 5, 4, 13, 14, 17, 16],
            [3, 4, 7, 6, 15, 16, 19, 18], [4, 5, 8, 7, 16, 17, 20, 19],
            [6, 7, 10, 9, 18, 19, 22, 21], [7, 8, 11, 10, 19, 20, 23, 22],
            [12, 13, 16, 15, 24, 25, 28, 27], [13, 14, 17, 16, 25, 26, 29, 28],
            [15, 16, 19, 18, 27, 28, 31, 30], [16, 17, 20, 19, 28, 29, 32, 31],
            [18, 19, 22, 21, 30, 31, 34, 33], [19, 20, 23, 22, 31, 32, 35, 34],
            [24, 25, 28, 27, 36, 37, 40, 39], [25, 26, 29, 28, 37, 38, 41, 40],
            [27, 28, 31, 30, 39, 40, 43, 42], [28, 29, 32, 31, 40, 41, 44, 43],
            [30, 31, 34, 33, 42, 43, 46, 45], [31, 32, 35, 34, 43, 44, 47, 46],
            [36, 37, 40, 39, 48, 49, 52, 51], [37, 38, 41, 40, 49, 50, 53, 52],
            [39, 40, 43, 42, 51, 52, 55, 54], [40, 41, 44, 43, 52, 53, 56, 55],
            [42, 43, 46, 45, 54, 55, 58, 57], [43, 44, 47, 46, 55, 56, 59, 58]
    ]


@pytest.fixture
def expected_quad_faces():
    return [[0, 1, 3, 2], [2, 3, 5, 4], [4, 5, 7, 6]]


@pytest.fixture
def sample_1d_array():
    return np.array([0, 5, 4, 4, 6, 8, 8, 9])
