import unittest
import pytest
import numpy as np

from gustaf.utils.connec import tet_to_tri, hexa_to_quad


class TestTetToTriFunction(unittest.TestCase):

    def test_tet_to_tri_throwException(self):
        volumes = np.array([0, 5, 4, 4, 6, 8, 8, 9])
        tet_to_tri(volumes)

    def test_tet_to_tri_expect_raiseValueError(self):
        volumes = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
        self.assertRaises(ValueError, tet_to_tri(volumes))

    def test_tet_to_tri_corr_volumes(self):
        volumes = np.array([[1, 2, 3, 3], [4, 5, 6, 5], [7, 8, 9, 5]])
        expect_data = [[1, 3, 2],
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
                       [5, 9, 7]]
        self.assertTrue(np.equal(expect_data, tet_to_tri(volumes)).all())


class TestHexaToQuadFunction(unittest.TestCase):

    def test_hexa_to_quad_throwException(self):
        volumes = np.array([0, 5, 4, 4, 6, 8, 8, 9])
        hexa_to_quad(volumes)

    def test_hexa_to_quad_expect_throwException_raiseValueError(self):
        data_nd_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
        self.assertRaises(ValueError, hexa_to_quad(data_nd_array))

    def test_hexa_to_quad_corr_volume(self):
        volumes = np.array([[1, 2, 3, 3, 3, 3, 4, 5],
                         [1, 2, 3, 3, 3, 3, 4, 5],
                         [1, 2, 3, 3, 3, 3, 4, 5],
                         [1, 2, 3, 3, 3, 3, 4, 5]])

        expect_data = [[2, 1, 3, 3],
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
                       [3, 3, 4, 5]]

        self.assertTrue(np.equal(expect_data, hexa_to_quad(volumes)).all())


class TestFacesToEdgesFunction(unittest.TestCase):
    def test_faces_to_edges_throwException(self):
        pass


class TestMakeQuadFacesFunction(unittest.TestCase):

    def test_make_quad_faces_throwException(self):
        pass


class TestMakeHexaVolumesFunction(unittest.TestCase):

    def test_make_hexa_volumns_throwException(self):
        pass
