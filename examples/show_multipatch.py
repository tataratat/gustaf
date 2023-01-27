import random

import gustaf as gus

random.seed(220894)


def split_plane(x):
    return x[:, 0] + x[:, 1] > 1


# Fasten your seat belts
gus.settings.NTHREADS = 4

# Create simple 2D multipatch geometry based of a biquadratic BSpline
b1 = gus.Bezier(
    degrees=[2, 2],
    control_points=[
        [-1, -1],
        [0, -0.8],
        [1, -1.1],
        [-1.3, 0],
        [0, 0],
        [1.3, 0.1],
        [-1, 0.9],
        [0, 0.8],
        [1, 1.1],
    ],
).bspline
b1.insert_knots(0, [i * 0.1 for i in range(1, 10)])
b1.insert_knots(1, [i * 0.1 for i in range(1, 10)])
spline_list = b1.extract.beziers()

# shuffle list to make system more exciting
random.shuffle(spline_list)

multipatch = gus.Multipatch(splines=spline_list)
multipatch.boundaries_from_continuity()
multipatch.show(
    boundary_ids=True, knots=False, control_points=False, resolutions=4
)
multipatch.boundary_from_function(split_plane, from_boundaries=[1, 2])
multipatch.show(
    boundary_ids=True,
    knots=False,
    contours=True,
    control_points=True,
    control_mesh=False,
    resolutions=4,
)

# 3D structure simple
b1 = gus.Bezier(
    degrees=[1, 1, 1],
    control_points=[
        [-2, 0, 3],
        [0, 0, 3],
        [-2, 1, 3],
        [0, 1, 3],
        [-2, 0, 0],
        [0, 0, 0],
        [-2, 1, 0],
        [0, 1, 0],
    ],
)
b2 = gus.Bezier(
    degrees=[1, 1, 1],
    control_points=[
        [0, -1, 0],
        [-2, -1, 0],
        [0, 0, 0],
        [-2, 0, 0],
        [0, -1, 3],
        [-2, -1, 3],
        [0, 0, 3],
        [-2, 0, 3],
    ],
)
b3 = gus.Bezier(
    degrees=[1, 1, 1],
    control_points=[
        [1, 1, 3],
        [0, 1, 3],
        [1, 0, 3],
        [0, 0, 3],
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ],
)
b4 = gus.Bezier(
    degrees=[1, 1, 1],
    control_points=[
        [0, 0, 0],
        [0, -1, 0],
        [1, 0, 0],
        [1, -1, 0],
        [0, 0, 3],
        [0, -1, 3],
        [1, 0, 3],
        [1, -1, 3],
    ],
)

# Multipatch
multipatch = gus.Multipatch([b1, b2, b3, b4])
multipatch.boundaries_from_continuity()
multipatch.show(
    boundary_ids=True, resolutions=5, knots=True, control_points=False
)


# Test 2
def foo(x):
    """
    Parametrization Function (determines thickness)
    """
    return tuple([(x[:, 0]) * 0.05 + (x[:, 1]) * 0.05 + (x[:, 2]) * 0.1 + 0.1])


generator = gus.spline.microstructure.Microstructure()
generator.deformation_function = gus.Bezier(
    degrees=[1, 1, 2],
    control_points=[
        [0, -1, 0],
        [1, -1, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, -2, 1],
        [1, -2, 1],
        [0, -0.2, 1],
        [1, -0.2, 1],
        [0, -1, 2],
        [1, -1, 2],
        [0, 0, 2],
        [1, 0, 2],
    ],
)
generator.microtile = gus.spline.microstructure.tiles.CrossTile3D()
generator.tiling = [3, 3, 5]
generator.parametrization_function = foo

microstructure = generator.create(
    closing_face="z", seperator_distance=0.4, center_expansion=1.3
)


# Multipatch
multipatch = gus.Multipatch(microstructure)
multipatch.boundaries_from_continuity()
multipatch.show(
    boundary_ids=True, resolutions=3, knots=True, control_points=False
)
