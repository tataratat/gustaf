import numpy as np
import pytest
from typing import List

from gustaf.spline.ffd import FFD


@pytest.mark.parametrize(
        "init_values, throw_error",
        [
                ([], False),  # test empty input
                (["jallo"], True),
                ([1], True),
                (["fixture_bspline_2d"], True),
                (["fixture_faces_tri"], False),
                (["fixture_faces_quad"], False),
                (["fixture_faces_tri", "fixture_bspline_3d"], False),
                (["fixture_faces_tri", "fixture_bspline_3d_deformed"], False),
                (["fixture_faces_tri", 1], True),
                (["fixture_faces_tri", "testing"], True),
        ]
)
def test_init_error(init_values: List, throw_error, request):
    for idx, value in enumerate(init_values):
        if isinstance(value, str):
            if "fixture_" in value:
                init_values[idx] = request.getfixturevalue(value[8:])
    if throw_error:
        with pytest.raises(ValueError):
            FFD(*init_values)
    else:
        FFD(*init_values)


def test_mesh_with_empty_init(faces_quad):
    a = FFD()

    a.mesh = faces_quad

    assert np.allclose(a.mesh.vertices, faces_quad.vertices)

    assert np.allclose(a.spline.dim, faces_quad.vertices.shape[1])

    assert np.allclose(a.spline.dim, a.spline.para_dim)

    assert a._spline._data.get("gustaf_ffd_computed", False)

    a.mesh


def test_spline_with_empty_init(bspline_2d):
    a = FFD()

    a.spline = bspline_2d

    with pytest.raises(RuntimeError):
        a.mesh

    assert np.allclose(a.control_points, bspline_2d.control_points)


def test_control_point_setter_with_empty_init(bspline_2d):
    a = FFD()

    # can not set control points if spline is not set
    with pytest.raises(ValueError):
        a.control_points = [1, 2, 3, 4]

    a.spline = bspline_2d

    # can not set control points with other shape than previous
    with pytest.raises(ValueError):
        a.control_points = [[1, 2, 3, 2, 3], [1, 2, 3, 2, 3], [1, 2, 3, 2, 3]]

    a.control_points = bspline_2d.control_points * 2

    assert np.allclose(a.control_points, bspline_2d.control_points)


def test_check_dimensions(faces_quad, bspline_para_1_dim_2):
    a = FFD()

    # spline dimensions do not match (will be checked at mesh setter)
    a.spline = bspline_para_1_dim_2

    # spline and mesh have different dimensions
    with pytest.raises(RuntimeError) as err:
        a.mesh = faces_quad

    assert "geometric" in str(err)  # spline para to dim miss match
    assert "mesh" in str(err)  # mesh dimension to spline miss match


def test_mesh(bspline_2d):
    a = FFD()

    a.spline = bspline_2d

    with pytest.raises(RuntimeError):
        a.mesh


@pytest.mark.parametrize(
        "spline_str, value_error, notimplemented_error", (
                (None, True, False), ("bspline_para_1_dim_2", False, False),
                ("bezier_2d", False, True)
        )
)
def test_elevate_degree(
        spline_str, value_error, notimplemented_error, request
):
    a = FFD()

    if spline_str:
        spline = request.getfixturevalue(spline_str)
        a.spline = spline.copy()
    if value_error:
        with pytest.raises(ValueError):
            a.elevate_degree()
    elif notimplemented_error:
        with pytest.raises(NotImplementedError):
            a.elevate_degree()
    else:
        spline.elevate_degree(0)
        a.elevate_degree(0)
        assert np.allclose(spline.degrees, a.spline.degrees)
        assert np.allclose(spline.knot_vectors, a.spline.knot_vectors)
        assert np.allclose(spline.control_points, a.spline.control_points)


@pytest.mark.parametrize(
    "spline_str, value_error, notimplemented_error",
    (   # only one error type can be checked
        (None, True, False),
        ("bspline_para_1_dim_2", False, False),
        ("bezier_2d", False, True)

    )
)
def test_insert_knots(spline_str, value_error, notimplemented_error, request):
    a = FFD()

    if spline_str:
        spline = request.getfixturevalue(spline_str)
        a.spline = spline.copy()
    if value_error:
        with pytest.raises(ValueError):
            a.insert_knots(0, [.5])
    elif notimplemented_error:
        with pytest.raises(NotImplementedError):
            a.insert_knots(0, [.5])
    else:
        for current in [spline, a]:
            current.insert_knots(0, [.5])
        assert np.allclose(spline.degrees, a.spline.degrees)
        assert np.allclose(spline.knot_vectors, a.spline.knot_vectors)
        assert np.allclose(spline.control_points, a.spline.control_points)


@pytest.mark.parametrize(
    "spline_str, value_error, notimplemented_error",
    (   # only one error type can be checked
        (None, True, False),
        ("bspline_para_1_dim_2", False, False),
        ("bezier_2d", False, True)

    )
)
def test_remove_knots(spline_str, value_error, notimplemented_error, request):
    a = FFD()

    if spline_str:
        spline = request.getfixturevalue(spline_str)
        a.spline = spline.copy()
    if value_error:
        with pytest.raises(ValueError):
            a.remove_knots(0, [.5])
    elif notimplemented_error:
        with pytest.raises(NotImplementedError):
            a.remove_knots(0, [.5])
    else:
        for current in [spline, a]:
            current.insert_knots(0, [.5])
        for current in [spline, a]:
            current.remove_knots(0, [.5])
        assert np.allclose(spline.degrees, a.spline.degrees)
        assert np.allclose(spline.knot_vectors, a.spline.knot_vectors)
        assert np.allclose(spline.control_points, a.spline.control_points)


@pytest.mark.parametrize(
    "spline_str, value_error, notimplemented_error",
    (   # only one error type can be checked
        (None, True, False),
        ("bspline_para_1_dim_2", False, False),
        ("bezier_2d", False, True)
    )
)
def test_rreduce_degree(
        spline_str, value_error, notimplemented_error, request
):
    a = FFD()

    if spline_str:
        spline = request.getfixturevalue(spline_str)
        a.spline = spline.copy()
    if value_error:
        with pytest.raises(ValueError):
            a.reduce_degree(0)
    elif notimplemented_error:
        with pytest.raises(NotImplementedError):
            a.reduce_degree(0)
    else:
        for current in [spline, a]:
            current.reduce_degree(0)
        assert np.allclose(spline.degrees, a.spline.degrees)
        assert np.allclose(spline.knot_vectors, a.spline.knot_vectors)
        assert np.allclose(spline.control_points, a.spline.control_points)
