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

    assert np.allclose(
        a.mesh.vertices,
        faces_quad.vertices
    )

    assert np.allclose(
        a.spline.dim,
        faces_quad.vertices.shape[1]
    )

    assert np.allclose(
        a.spline.dim,
        a.spline.para_dim
    )
    
    assert a._spline._data.get("gustaf_ffd_computed", False)

    a.mesh


def test_spline_with_empty_init(bspline_2d):
    a = FFD()

    a.spline = bspline_2d

    with pytest.raises(RuntimeError):
        a.mesh
    
    assert np.allclose(
        a.control_points,
        bspline_2d.control_points
    )


def test_control_point_setter_with_empty_init(bspline_2d):
    a = FFD()

    # can not set control points if spline is not set
    with pytest.raises(ValueError):
        a.control_points = [1,2,3,4]
    

    a.spline = bspline_2d

    # can not set control points with other shape than previous 
    with pytest.raises(ValueError):
        a.control_points = [[1,2,3,2,3],[1,2,3,2,3],[1,2,3,2,3]]
    
    a.control_points = bspline_2d.control_points * 2

    assert np.allclose(
        a.control_points,
        bspline_2d.control_points
    )




def test_spline_with_empty_init(bspline_2d):
    a = FFD()

    a.spline = bspline_2d

    with pytest.raises(RuntimeError):
        a.mesh
