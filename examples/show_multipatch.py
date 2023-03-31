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
multipatch.show_options["boundary_ids"] = True
multipatch.show_options["knots"] = False
multipatch.show_options["control_points"] = False
multipatch.show_options["resolutions"] = 4
multipatch.show_options["overwrite_spline_options"] = True
multipatch.show()

multipatch.boundary_from_function(split_plane, mask=[1, 2])
multipatch.show_options["control_points"] = True
multipatch.show_options["control_mesh"] = False
multipatch.show()

gus.spline.io.gismo.export(
    "rectangle_test_mesh.xml", multipatch=multipatch, indent=True
)

# 3D structure simple
b1 = gus.Bezier(
    degrees=[1, 1, 1],
    control_points=[
        [-2 + 2, 0 + 2, 3 + 2],
        [0 + 2, 0 + 2, 3 + 2],
        [-2 + 2, 1 + 2, 3 + 2],
        [0 + 2, 1 + 2, 3 + 2],
        [-2 + 2, 0 + 2, 0 + 2],
        [0 + 2, 0 + 2, 0 + 2],
        [-2 + 2, 1 + 2, 0 + 2],
        [0 + 2, 1 + 2, 0 + 2],
    ],
)
b2 = gus.Bezier(
    degrees=[1, 1, 1],
    control_points=[
        [0 + 2, -1 + 2, 0 + 2],
        [-2 + 2, -1 + 2, 0 + 2],
        [0 + 2, 0 + 2, 0 + 2],
        [-2 + 2, 0 + 2, 0 + 2],
        [0 + 2, -1 + 2, 3 + 2],
        [-2 + 2, -1 + 2, 3 + 2],
        [0 + 2, 0 + 2, 3 + 2],
        [-2 + 2, 0 + 2, 3 + 2],
    ],
)
b3 = gus.Bezier(
    degrees=[1, 1, 1],
    control_points=[
        [1 + 2, 1 + 2, 3 + 2],
        [0 + 2, 1 + 2, 3 + 2],
        [1 + 2, 0 + 2, 3 + 2],
        [0 + 2, 0 + 2, 3 + 2],
        [1 + 2, 1 + 2, 0 + 2],
        [0 + 2, 1 + 2, 0 + 2],
        [1 + 2, 0 + 2, 0 + 2],
        [0 + 2, 0 + 2, 0 + 2],
    ],
)
b4 = gus.Bezier(
    degrees=[1, 1, 1],
    control_points=[
        [0 + 2, 0 + 2, 0 + 2],
        [0 + 2, -1 + 2, 0 + 2],
        [1 + 2, 0 + 2, 0 + 2],
        [1 + 2, -1 + 2, 0 + 2],
        [0 + 2, 0 + 2, 3 + 2],
        [0 + 2, -1 + 2, 3 + 2],
        [1 + 2, 0 + 2, 3 + 2],
        [1 + 2, -1 + 2, 3 + 2],
    ],
)

# Multipatch - custom function for all fields


def plot_func(data, resolutions=None, on=None):
    """
    callback to evaluate derivatives
    """
    if resolutions is not None:
        q = gus.create.vertices.raster([[0, 0], [1, 1]], resolutions).vertices
        return data.derivative(q, [0, 1])
    elif on is not None:
        return data.derivative(on, [0, 1])


multipatch = gus.Multipatch([b1, b2, b3, b4])
multipatch.boundaries_from_continuity()
multipatch.show_options["resolutions"] = 5
multipatch.show_options["knots"] = True
multipatch.show_options["control_points"] = False
multipatch.show_options["overwrite_spline_options"] = True
multipatch.show_options["field_function"] = "me"
multipatch.show_options["scalarbar"] = True
multipatch.show()


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
    closing_face="z", separator_distance=0.4, center_expansion=1.3
)


# Multipatch
multipatch = gus.Multipatch(microstructure)
multipatch.boundaries_from_continuity()
multipatch.show_options["boundary_ids"] = True
multipatch.show_options["resolutions"] = 3
multipatch.show_options["knots"] = True
multipatch.show_options["control_points"] = False
multipatch.show_options["overwrite_spline_options"] = True

multipatch.show(
    # boundary_ids=True, resolutions=3, knots=True, control_points=False
)
