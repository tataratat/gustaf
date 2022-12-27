import numpy as np
import pytest

import gustaf as gus


@pytest.mark.parametrize("values", [0, 1, {"hallo": 1}, [1, 2, 3], "test"])
def test_extrude_error_on_no_spline_given(values):
    with pytest.raises(NotImplementedError):
        gus.spline.create.extruded(values)


@pytest.mark.parametrize(
    "spline_name", ["bspline_2d", "nurbs_2d", "bezier_2d", "rationalbezier_2d"]
)
@pytest.mark.parametrize(
    "axis,error",
    [
        (None, True),
        (1, True),
        ([1], True),
        ({"hallo": 1}, True),
        ("hallo", True),
        (np.random.rand(3), False),
    ],
)
def test_extrude(spline_name: str, axis, error, request):
    # get the correct spline from the provided fixtures
    spline: gus.spline.base.GustafSpline = request.getfixturevalue(spline_name)
    if error:
        with pytest.raises(ValueError):
            spline.create.extruded(extrusion_vector=axis)
    else:
        extrudate = spline.create.extruded(extrusion_vector=axis)
        x, y, z = np.random.rand(3).tolist()
        assert np.allclose(
            extrudate.evaluate([[x, y, z]]),
            np.hstack((spline.evaluate([[x, y]]), np.zeros((1, 1))))
            + z * axis,
        )


@pytest.mark.parametrize("values", [0, 1, {"hallo": 1}, [1, 2, 3], "test"])
def test_revolved_error_on_no_spline_given(values):
    with pytest.raises(NotImplementedError):
        gus.spline.create.revolved(values)


@pytest.mark.parametrize(
    "spline_name", ["bspline_2d", "nurbs_2d", "bezier_2d", "rationalbezier_2d"]
)
@pytest.mark.parametrize(
    "axis,center,angle,n_knot_span,degree,error",
    [
        # (None, None, None, None, True, True),
        tuple([1]) + tuple([None] * 3) + tuple([True, True]),
        tuple([[1]]) + tuple([None] * 3) + tuple([True, True]),
        tuple([[0, 0, 1e-18]]) + tuple([None] * 3) + tuple([True, True]),
        tuple([{"hallo": 1}]) + tuple([None] * 3) + tuple([True, True]),
        tuple(["hallo"]) + tuple([None] * 3) + tuple([True, True]),
        # (np.random.rand(3))+tuple([None]*4)+tuple((False))
    ],
)
def test_revolved_3d(
    spline_name: str, axis, center, angle, n_knot_span, degree, error, request
):
    # get the correct spline from the provided fixtures
    spline: gus.spline.base.GustafSpline = request.getfixturevalue(spline_name)
    if error:
        with pytest.raises(ValueError):
            spline.create.revolved(axis, center, angle, n_knot_span, degree)
    else:
        if angle is None:
            angle = 360
        cc, ss = np.cos(angle), np.sin(angle)
        r = np.array([[cc, -ss, 0], [ss, cc, 0], [0, 0, 1]])
        revolved = spline.create.revolved(
            axis, center, angle, n_knot_span, degree
        )

        dim_bumped_cps = np.zeros((spline.control_points.shape[0], 1))

        ref_sol = np.matmul(
            np.hstack((spline.control_points, dim_bumped_cps)), r.T
        )

        assert np.allclose(
            revolved.control_points[-10:, :],
            ref_sol,
        ), f"{spline.whatami} failed revolution"

    # Test Revolution Routine


def test_create_revolution():
    """
    Test revolution routines for different input types and arguments
    """
    # Make some lines
    bezier_line = gus.Bezier(control_points=[[1, 0], [2, 1]], degrees=[1])
    nurbs_line = bezier_line.nurbs

    # Revolve always around z-axis
    # init rotation matrix
    r_angle = np.random.rand()
    r_center = np.array([1, 0])
    cc, ss = np.cos(r_angle), np.sin(r_angle)
    R2 = np.array([[cc, -ss], [ss, cc]])

    # Test 2D Revolutions of lines
    for spline_g in (bezier_line, nurbs_line):
        assert np.allclose(
            spline_g.create.revolved(
                angle=r_angle, degree=False
            ).control_points[-2:, :],
            np.matmul(spline_g.control_points, R2.T),
        )

    # Test 2D Revolutions of lines around center
    for spline_g in (bezier_line, nurbs_line):
        assert np.allclose(
            spline_g.create.revolved(
                angle=r_angle, center=r_center, degree=False
            ).control_points[-2:, :],
            np.matmul(spline_g.control_points - r_center, R2.T) + r_center,
        ), f"{spline_g.whatami} failed revolution around center"
