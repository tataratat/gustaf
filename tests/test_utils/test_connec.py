import numpy as np

from gustaf.utils.connec import tet_to_tri, hexa_to_quad, faces_to_edges, make_quad_faces, make_hexa_volumes
import pytest


def test_tet_to_tri_throwException(sample_1d_array):
    # TODO this is a bug!!
    with pytest.raises(IndexError):
        tet_to_tri(sample_1d_array)


def test_tet_to_tri_expect_raiseValueError(sample_tri_error):
    with pytest.raises(ValueError):
        tet_to_tri(sample_tri_error)


def test_tet_to_tri_corr_volumes(sample_tet, expected_tet_result):
    assert np.equal(expected_tet_result, tet_to_tri(sample_tet)).all()


def test_hexa_to_quad_throwException(sample_1d_array):
    # TODO this is a bug!
    with pytest.raises(IndexError):
        hexa_to_quad(sample_1d_array)


def test_hexa_to_quad_expect_throwException_raiseValueError(sample_hex_error):
    with pytest.raises(ValueError):
        hexa_to_quad(sample_hex_error)


def test_hexa_to_quad_corr_volume(sample_hex_array, expected_quad_result):
    assert np.equal(expected_quad_result, hexa_to_quad(sample_hex_array)).all()


def test_faces_to_edges_throwException(sample_1d_array):
    # TODO this is a bug!
    with pytest.raises(IndexError):
        faces_to_edges(sample_1d_array)


def test_faces_to_edges_faces_tet(sample_faces_tri, expected_edges_tri):
    assert np.equal(expected_edges_tri, faces_to_edges(sample_faces_tri)).all()


def test_faces_to_edges_feces_tri(sample_faces_tet, expected_edges_tet):
    assert np.equal(expected_edges_tet, faces_to_edges(sample_faces_tet)).all()


def test_make_quad_faces_throwException(sample_hex):
    # TODO this is a bug!
    with pytest.raises(ValueError):
        make_quad_faces(sample_hex)


def test_make_quad_faces(sample_quad_faces, expected_quad_faces):
    assert np.equal(expected_quad_faces, make_quad_faces(sample_quad_faces)).all()


def test_make_quad_faces_wrong_input(sample_quad_faces_fail):
    # TODO this is a bug
    with pytest.raises(ValueError):
        make_quad_faces(sample_quad_faces_fail)


def test_make_hexa_volumes_throwException(sample_hex_fail):
    # TODO this is a bug!
    with pytest.raises(ValueError):
        make_hexa_volumes(sample_hex_fail)


def test_make_hexa_volumes_suc(sample_hex, expected_hexa_volumes):
    assert np.equal(expected_hexa_volumes, make_hexa_volumes(sample_hex)).all()
