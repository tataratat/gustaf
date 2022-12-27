import pytest
import numpy as np


@pytest.fixture
def sample_tet():
    return np.array([[1, 2, 3, 3], [4, 5, 6, 5], [7, 8, 9, 5]])


@pytest.fixture
def sample_hex_array():
    return np.array(
        [
            [1, 2, 3, 3, 3, 3, 4, 5],
            [1, 2, 3, 3, 3, 3, 4, 5],
            [1, 2, 3, 3, 3, 3, 4, 5],
            [1, 2, 3, 3, 3, 3, 4, 5],
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
        [1, 3, 2],
        [2, 3, 1],
        [3, 3, 2],
        [3, 3, 1],
        [4, 6, 5],
        [5, 5, 4],
        [6, 5, 5],
        [5, 6, 4],
        [7, 9, 8],
        [8, 5, 7],
        [9, 5, 8],
        [5, 9, 7],
    ]


@pytest.fixture
def expected_quad_result():
    return [
        [2, 1, 3, 3],
        [1, 2, 3, 3],
        [2, 3, 4, 3],
        [3, 3, 5, 4],
        [3, 1, 3, 5],
        [3, 3, 4, 5],
        [2, 1, 3, 3],
        [1, 2, 3, 3],
        [2, 3, 4, 3],
        [3, 3, 5, 4],
        [3, 1, 3, 5],
        [3, 3, 4, 5],
        [2, 1, 3, 3],
        [1, 2, 3, 3],
        [2, 3, 4, 3],
        [3, 3, 5, 4],
        [3, 1, 3, 5],
        [3, 3, 4, 5],
        [2, 1, 3, 3],
        [1, 2, 3, 3],
        [2, 3, 4, 3],
        [3, 3, 5, 4],
        [3, 1, 3, 5],
        [3, 3, 4, 5],
    ]


@pytest.fixture
def expected_edges_tri():
    return [
        [1, 2],
        [2, 3],
        [3, 3],
        [3, 1],
        [4, 5],
        [5, 6],
        [6, 5],
        [5, 4],
        [7, 8],
        [8, 9],
        [9, 5],
        [5, 7],
    ]


@pytest.fixture
def expected_edges_tet():
    return [
        [1, 2],
        [2, 3],
        [3, 3],
        [3, 1],
        [4, 5],
        [5, 6],
        [6, 5],
        [5, 4],
        [7, 8],
        [8, 9],
        [9, 5],
        [5, 7],
        [7, 8],
        [8, 9],
        [9, 5],
        [5, 7],
    ]


@pytest.fixture
def expected_hexa_volumes():
    return [
        [0, 1, 4, 3, 12, 13, 16, 15],
        [1, 2, 5, 4, 13, 14, 17, 16],
        [3, 4, 7, 6, 15, 16, 19, 18],
        [4, 5, 8, 7, 16, 17, 20, 19],
        [6, 7, 10, 9, 18, 19, 22, 21],
        [7, 8, 11, 10, 19, 20, 23, 22],
        [12, 13, 16, 15, 24, 25, 28, 27],
        [13, 14, 17, 16, 25, 26, 29, 28],
        [15, 16, 19, 18, 27, 28, 31, 30],
        [16, 17, 20, 19, 28, 29, 32, 31],
        [18, 19, 22, 21, 30, 31, 34, 33],
        [19, 20, 23, 22, 31, 32, 35, 34],
        [24, 25, 28, 27, 36, 37, 40, 39],
        [25, 26, 29, 28, 37, 38, 41, 40],
        [27, 28, 31, 30, 39, 40, 43, 42],
        [28, 29, 32, 31, 40, 41, 44, 43],
        [30, 31, 34, 33, 42, 43, 46, 45],
        [31, 32, 35, 34, 43, 44, 47, 46],
        [36, 37, 40, 39, 48, 49, 52, 51],
        [37, 38, 41, 40, 49, 50, 53, 52],
        [39, 40, 43, 42, 51, 52, 55, 54],
        [40, 41, 44, 43, 52, 53, 56, 55],
        [42, 43, 46, 45, 54, 55, 58, 57],
        [43, 44, 47, 46, 55, 56, 59, 58],
    ]


@pytest.fixture
def expected_quad_faces():
    return [[0, 1, 3, 2], [2, 3, 5, 4], [4, 5, 7, 6]]


@pytest.fixture
def sample_1d_array():
    return np.array([0, 5, 4, 4, 6, 8, 8, 9])


@pytest.fixture
def sample_c_contiguous_data_array():
    return np.array([0, 5, 4, 4, 6, 8, 8, 9])


@pytest.fixture
def sample_c_contiguous_nd_array():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def sample_c_contiguous_2d_array():
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def expected_c_contiguous_con(sample_c_contiguous_data_array):
    return np.ascontiguousarray(sample_c_contiguous_data_array)


@pytest.fixture
def expected_c_contiguous_con_nd(sample_c_contiguous_nd_array):
    return np.ascontiguousarray(sample_c_contiguous_nd_array)


@pytest.fixture
def sample_c_contiguous_non_con(sample_c_contiguous_data_array):
    return sample_c_contiguous_data_array.reshape(2, 4).transpose()
