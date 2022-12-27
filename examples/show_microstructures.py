import gustaf as gus
import numpy as np
from vedo import colors, Mesh

# First Test
generator = gus.spline.microstructure.Microstructure()
generator.deformation_function = gus.Bezier(
    degrees=[2, 1],
    control_points=[[0, 0], [1, 0], [2, -1], [-1, 1], [1, 1], [3, 2]],
)
generator.microtile = [
    gus.Bezier(
        degrees=[3], control_points=[[0, 0.5], [0.5, 1], [0.5, 0], [1, 0.5]]
    ),
    gus.Bezier(
        degrees=[4],
        control_points=[
            [0.5, 0],
            [0.75, 0.5],
            [0.8, 0.8],
            [0.25, 0.5],
            [0.5, 1],
        ],
    ),
]
generator.tiling = [8, 8]
generator.show(
    knots=False, control_points=False, title="2D Lattice Microstructure"
)

# Second test


def parametrization_function(x):
    return tuple(
        [0.3 - 0.4 * np.maximum(abs(0.5 - x[:, 0]), abs(0.5 - x[:, 1]))]
    )


generator = gus.spline.microstructure.Microstructure()
generator.deformation_function = gus.Bezier(
    degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
)
generator.microtile = gus.spline.microstructure.tiles.CrossTile2D()
generator.tiling = [5, 5]
generator.parametrization_function = parametrization_function
ms = generator.create(closing_face="x", center_expansion=1.3)
generator.show(
    use_saved=True,
    knots=True,
    control_points=False,
    title="2D Crosstile Parametrized Microstructure",
)

# Third test
generator = gus.spline.microstructure.Microstructure()
generator.deformation_function = gus.Bezier(
    degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
).create.extruded(extrusion_vector=[0, 0, 1])
generator.microtile = [
    gus.Bezier(
        degrees=[3],
        control_points=[
            [0, 0.5, 0.5],
            [0.5, 1, 0.5],
            [0.5, 0, 0.5],
            [1, 0.5, 0.5],
        ],
    ),
    gus.Bezier(
        degrees=[3],
        control_points=[
            [0.5, 0.5, 0.0],
            [0.5, 0, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ],
    ),
    gus.Bezier(
        degrees=[3],
        control_points=[
            [0.5, 0, 0.5],
            [1, 0.5, 0.5],
            [0, 0.5, 0.5],
            [0.5, 1, 0.5],
        ],
    ),
]
generator.tiling = [1, 2, 3]
generator.show(
    knots=False, control_points=False, title="3D Lattice Microstructure"
)

# Fourth test
generator = gus.spline.microstructure.microstructure.Microstructure()
generator.deformation_function = gus.Bezier(
    degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
).create.extruded(extrusion_vector=[0, 0, 1])
generator.microtile = gus.spline.microstructure.tiles.CrossTile3D()
generator.tiling = [2, 2, 3]
generator.show(
    control_points=False, resolutions=2, title="3D Crosstile Microstructure"
)

# Fifth test
# Non-uniform tiling
generator = gus.spline.microstructure.microstructure.Microstructure()
generator.deformation_function = gus.BSpline(
    degrees=[2, 1],
    control_points=[
        [0, 0],
        [0.1, 0],
        [0.2, 0],
        [1, 0],
        [0, 1],
        [0.1, 1],
        [0.2, 2],
        [1, 3],
    ],
    knot_vectors=[[0, 0, 0, 0.15, 1, 1, 1], [0, 0, 1, 1]],
)
generator.microtile = [
    gus.Bezier(
        degrees=[3], control_points=[[0, 0.5], [0.5, 1], [0.5, 0], [1, 0.5]]
    ),
    gus.Bezier(
        degrees=[3], control_points=[[0.5, 0], [1, 0.5], [0, 0.5], [0.5, 1]]
    ),
]
generator.tiling = [5, 1]
generator.show(
    knot_span_wise=False,
    control_points=False,
    resolutions=20,
    title="2D Lattice with global tiling",
)

# Sixth test
# A Parametrized microstructure and its respective inverse structure


def foo(x):
    """
    Parametrization Function (determines thickness)
    """
    return tuple([(x[:, 0]) * 0.05 + (x[:, 1]) * 0.05 + (x[:, 2]) * 0.1 + 0.1])


generator = gus.spline.microstructure.Microstructure()
generator.deformation_function = gus.Bezier(
    degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
).create.extruded(extrusion_vector=[0, 0, 1])
generator.microtile = gus.spline.microstructure.tiles.InverseCrossTile3D()
generator.tiling = [3, 3, 5]
generator.parametrization_function = foo

inverse_microstructure = generator.create(
    closing_face="z", seperator_distance=0.4, center_expansion=1.3
)

# Plot the results
_, showables_inverse = generator.show(
    closing_face="z",
    seperator_distance=0.4,
    center_expansion=1.3,
    title="Parametrized Inverse Microstructure",
    control_points=False,
    knots=True,
    return_showable_list=True,
    resolutions=5,
)

# Corresponding Structure
generator.microtile = gus.spline.microstructure.tiles.CrossTile3D()
microstructure = generator.create(
    closing_face="z", seperator_distance=0.4, center_expansion=1.3
)
_, showables = generator.show(
    closing_face="z",
    center_expansion=1.3,
    title="Parametrized Microstructure",
    control_points=False,
    knots=True,
    return_showable_list=True,
    resolutions=5,
)

# Change the color of the inverse mesh to some blue shade
composed_structure = []
for item in showables:
    if isinstance(item, Mesh):
        composed_structure.append(item)
for item in showables_inverse:
    if isinstance(item, Mesh):
        item.c(colors.getColor(rgb=(0, 102, 153)))
        composed_structure.append(item)

gus.show.show_vedo(
    composed_structure, title="Parametrized Microstructure and its inverse"
)
