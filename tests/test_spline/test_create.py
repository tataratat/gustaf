import gustaf as gus

import pytest


@pytest.mark.parametrize("values", [0, 1, {"hallo": 1}, [1, 2, 3], "test"])
def test_extrude_error_on_not_spline_given(values):
    with pytest.raises(NotImplementedError):
        gus.spline.create.extruded(values)


@pytest.mark.parametrize(
        "axis,error", [
                (None, True),
                (1, True),
                ([1], True),
        ]
)
def test_extrude_BSpline_error_on_no_axis(
        bspline_2d: gus.BSpline, axis, error
):
    # create BSpline which is used
    if error:
        with pytest.raises(ValueError):
            bspline_2d.create.extruded(extrusion_vector=axis)
    else:
        bspline_2d.create.extruded(extrusion_vector=axis)


# def test_extrude_BSpline_error_on_axis_format_wrong(
#         bspline_2d: gus.BSpline):
#     #create BSpline which is used
#     with pytest.raises(ValueError):
#         bspline_2d.create.extruded()

#     # Test Extrusion routines
#     def test_create_extrude(self):
#         """
#         Test extrusion for different input types and arguments
#         """
#         if not gus.has_spline:
#             print("gustaf cannot load spline ext. skipping test.")
#             return None

#         # make a couple of 2D splines
#         bspline = gus.BSpline(
#                 control_points=c.CPS_2D,
#                 degrees=c.DEGREES_2D_NU,
#                 knot_vectors=c.KVS_2D
#         )
#         nurbs = gus.NURBS(
#                 control_points=c.CPS_2D,
#                 degrees=c.DEGREES_2D_NU,
#                 knot_vectors=c.KVS_2D,
#                 weights=c.WEIGHTS_2D
#         )
#         bezier = gus.Bezier(
#                 control_points=c.CPS_2D,
#                 degrees=c.DEGREES_2D_U,
#         )
#         rationalbezier = gus.RationalBezier(
#                 control_points=c.CPS_2D,
#                 degrees=c.DEGREES_2D_U,
#                 weights=c.WEIGHTS_2D
#         )

#         # Expect Failure - not a spline
#         with self.assertRaises(NotImplementedError):
#             gus.spline.create.extruded([4])
#         # Expect Failure - no axis given
#         with self.assertRaises(ValueError):
#             bspline.create.extruded()
#         # Expect Failure - axis wrong format
#         with self.assertRaises(ValueError):
#             bspline.create.extruded(extrusion_vector=[1])

#         # Create a random axis
#         axis = np.random.rand(3)
#         x, y, z = np.random.rand(3).tolist()

#         # Test results
#         for spline_g in (bspline, nurbs, rationalbezier, bezier):
#             self.assertTrue(
#                     np.allclose(
#                             bspline.create.extruded(extrusion_vector=axis
#                                                     ).evaluate([[x, y, z]]),
#                             np.hstack(
#                                     (
#                                             bspline.evaluate([[x, y]]),
#                                             np.zeros((1, 1))
#                                     )
#                             ) + z * axis
#                     )
#             )

#         # Create a random axis
#         axis = np.random.rand(3)
#         x, y, z = np.random.rand(3).tolist()

#         # Test results
#         for spline_g in (bspline, nurbs, rationalbezier, bezier):
#             self.assertTrue(
#                     np.allclose(
#                             spline_g.create.extruded(extrusion_vector=axis
#                                                      ).evaluate([[x, y, z]]),
#                             np.hstack(
#                                     (
#                                             spline_g.evaluate([[x, y]]),
#                                             np.zeros((1, 1))
#                                     )
#                             ) + z * axis
#                     )
#             )

#     # Test Revolution Routine
#     def test_create_revolution(self):
#         """
#         Test revolution routines for different input types and arguments
#         """
#         if not gus.has_spline:
#             print("gustaf cannot load spline ext. skipping test.")
#             return None

#         # make a couple of 2D splines
#         bspline = gus.BSpline(
#                 control_points=c.CPS_2D,
#                 degrees=c.DEGREES_2D_NU,
#                 knot_vectors=c.KVS_2D
#         )
#         nurbs = gus.NURBS(
#                 control_points=c.CPS_2D,
#                 degrees=c.DEGREES_2D_NU,
#                 knot_vectors=c.KVS_2D,
#                 weights=c.WEIGHTS_2D
#         )
#         bezier = gus.Bezier(
#                 control_points=c.CPS_2D,
#                 degrees=c.DEGREES_2D_U,
#         )
#         rationalbezier = gus.RationalBezier(
#                 control_points=c.CPS_2D,
#                 degrees=c.DEGREES_2D_U,
#                 weights=c.WEIGHTS_2D
#         )

#         # Make some lines
#        bezier_line = gus.Bezier(control_points=[[1, 0], [2, 1]], degrees=[1])
#         nurbs_line = bezier_line.nurbs

#         # Make a cuboid
#         cuboid = gus.Bezier(
#                 control_points=[
#                        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1],
#                         [1, 0, 1], [0, 1, 1], [1, 1, 1]
#                 ],
#                 degrees=[1, 1, 1]
#         )

#         # Expect Failure - not a spline
#         with self.assertRaises(NotImplementedError):
#             gus.spline.create.revolved([4])
#         # Expect Failure - No rotation axis
#         with self.assertRaises(ValueError):
#             cuboid.create.revolved()
#         # Expect Failure - axis wrong format
#         with self.assertRaises(ValueError):
#             bspline.create.revolved(axis=[1])
#         # Expect Failure - axis too small
#         with self.assertRaises(ValueError):
#             bspline.create.revolved(axis=[0, 0, 1e-18])

#         # Revolve always around z-axis
#         # init rotation matrix
#         r_angle = np.random.rand()
#         r_center = -np.random.rand(2)
#         r_center = np.array([1, 0])
#         cc, ss = np.cos(r_angle), np.sin(r_angle)
#         R = np.array([[cc, -ss, 0], [ss, cc, 0], [0, 0, 1]])
#         R2 = np.array([[cc, -ss], [ss, cc]])

#         # Test 3D revolutions for bodies
#         for spline_g in (bspline, nurbs, rationalbezier, bezier):
#             dim_bumped_cps = np.zeros((spline_g.control_points.shape[0], 1))

#             ref_sol = np.matmul(
#                     np.hstack((spline_g.control_points, dim_bumped_cps)), R.T
#             )

#             self.assertTrue(
#                     np.allclose(
#                             spline_g.create.revolved(
#                                     axis=[0, 0, 1],
#                                     center=[0, 0, 0],
#                                     angle=r_angle,
#                                     degree=False
#                             ).control_points[-10:, :],
#                             ref_sol,
#                     ), f"{spline_g.whatami} failed revolution"
#             )

#         # Test 2D Revolutions of lines
#         for spline_g in (bezier_line, nurbs_line):
#             self.assertTrue(
#                     np.allclose(
#                             spline_g.create.revolved(
#                                     angle=r_angle, degree=False
#                             ).control_points[-2:, :],
#                             np.matmul(spline_g.control_points, R2.T)
#                     )
#             )

#         # Test 2D Revolutions of lines around center
#         for spline_g in (bezier_line, nurbs_line):
#             self.assertTrue(
#                     np.allclose(
#                             spline_g.create.revolved(
#                                     angle=r_angle,
#                                     center=r_center,
#                                     degree=False
#                             ).control_points[-2:, :],
#                             np.
#                             matmul(spline_g.control_points - r_center, R2.T)
#                             + r_center
#                     ), f"{spline_g.whatami} failed revolution around center"
#             )

# if __name__ == "__main__":
#     c.unittest.main()
