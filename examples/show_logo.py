import numpy as np
import gustaf as gus

# Draw a fanfare
fanfare = []

# Mouthpiece
r_mouth = 0.2
r_shaft = 0.1
r_rounding = 0.05
l_mouth_1 = 0.4
# Cone
l_cone = 0.8
r_cone = 0.5
# Shaft
l_shaft = 2.
curved_radius = 0.25

# Create Splines
mouth_piece = gus.BSpline(
        knot_vectors=[[0, 0, 0, 0.25, 0.5, 0.5, 0.75, 1, 1, 1]],
        degrees=[2],
        control_points=[
                [l_mouth_1, r_mouth - r_rounding],
                [l_mouth_1 + r_rounding, r_mouth - r_rounding],
                [l_mouth_1 + r_rounding, r_mouth],
                [l_mouth_1, r_mouth],
                [.5 * l_mouth_1, r_mouth],
                [.5 * l_mouth_1, r_shaft],
                [0., r_shaft],
        ]
).nurbs.create.revolved(
        axis=[1, 0, 0],
        center=[0, 0, 0],
        angle=360,
        degree=True,
        n_knot_spans=4
)

#
isqrt2 = 2**(-.5)
cross_section = gus.NURBS(
        degrees=[2],
        knot_vectors=[[0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1]],
        control_points=(
                np.array(
                        [
                                [0., 0, 0], [0., 0.5, -0.5], [0., 1, 0],
                                [0., 1 + 0.5, 0.5], [0., 1, 1],
                                [0., 0.5, 1 + 0.5], [0., 0, 1],
                                [0., -0.5, 0.5], [0, 0, 0]
                        ]
                ) + np.array([0, -.5, -.5])
        ) * (r_shaft * 2 * isqrt2),
        weights=[1, isqrt2, 1, isqrt2, 1, isqrt2, 1, isqrt2, 1]
)

shaft = cross_section.create.extruded(extrusion_vector=[-l_shaft, 0, 0])

lower_shaft = shaft.copy()
lower_shaft.control_points += [0, -2 * curved_radius, 0]
upper_shaft = shaft.copy()
upper_shaft.control_points += [
        0, -2 * curved_radius + 2 * isqrt2 * curved_radius,
        -2 * isqrt2 * curved_radius
]

cross_section.control_points += [-l_shaft, 0, 0]

curved_piece = cross_section.nurbs.create.revolved(
        axis=[0., 0, 1], center=[-l_shaft, -curved_radius, 0], angle=180
)
cross_section.control_points += [l_shaft, -2 * curved_radius, 0]
second_curved_piece = cross_section.nurbs.create.revolved(
        axis=[0, 1., 1.],
        center=[
                0, -2 * curved_radius + isqrt2 * curved_radius,
                -isqrt2 * curved_radius
        ],
        angle=180
)

cone_part = gus.BSpline(
        degrees=[2],
        knot_vectors=[[0, 0, 0, 0.5, 1, 1, 1]],
        control_points=[
                [0., r_shaft, 0.],
                [-.25 * l_cone, r_shaft, 0.],
                [-l_cone, 0.5 * (r_cone + r_shaft), 0.],
                [-l_cone, r_cone, 0.],
        ]
).nurbs.create.revolved(
        axis=[1, 0, 0], center=[0, 0, 0], angle=360, degree=True
)
cone_part.control_points += [
        -l_shaft, -2 * curved_radius + 2 * isqrt2 * curved_radius,
        -2 * isqrt2 * curved_radius
]

fanfare.append(mouth_piece)
fanfare.append(curved_piece)
fanfare.append(second_curved_piece)
fanfare.append(upper_shaft)
fanfare.append(cone_part)
fanfare.append(shaft)
fanfare.append(lower_shaft)

_, showables = gus.show.show_vedo(
        fanfare,
        knots=False,
        control_points=False,
        return_show_list=True,
        resolutions=100
)

for m in showables:
    from vedo import colors
    m.c(colors.getColor(rgb=(255, 215, 0)))

gus.show.show_vedo(showables)
