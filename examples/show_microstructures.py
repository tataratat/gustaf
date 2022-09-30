import gustaf as gus
import numpy as np

# First Test
generator = gus.spline.microstructure.Generator()
generator.deformation_function = gus.Bezier(
        degrees=[2, 1],
        control_points=[[0, 0], [1, 0], [2, -1], [-1, 1], [1, 1], [3, 2]]
)
generator.microtile = [
        gus.Bezier(
                degrees=[3],
                control_points=[[0, .5], [.5, 1], [.5, 0], [1, .5]]
        ),
        gus.Bezier(
                degrees=[3],
                control_points=[[0.5, 0], [1, .5], [0, 0.5], [.5, 1]]
        )
]
generator.tiling = [8, 8]
gus.show.show_vedo(
        [*generator.create(), generator.deformation_function],
        surface_alpha=0.3,
        knots=False,
        control_points=False,
        title="2D Lattice Microstructure"
)

# Second test


def parametrization_function(x):
    return tuple([.3 - 0.4 * np.maximum(abs(.5 - x[:, 0]), abs(.5 - x[:, 1]))])


generator = gus.spline.microstructure.Generator()
generator.deformation_function = gus.Bezier(
        degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
)
generator.microtile = gus.spline.microstructure.tiles.CrossTile2D()
generator.tiling = [5, 5]
generator.parametrization_function = parametrization_function
gus.show.show_vedo(
        generator.create(closing_face="x", center_expansion=1.3),
        knots=True,
        control_points=False,
        title="2D Crosstile Parametrized Microstructure"
)

# Third test
generator = gus.spline.microstructure.Generator()
generator.deformation_function = gus.Bezier(
        degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
).create.extruded(extrusion_vector=[0, 0, 1])
generator.microtile = [
        gus.Bezier(
                degrees=[3],
                control_points=[
                        [0, .5, .5], [.5, 1, .5], [.5, 0, .5], [1, .5, .5]
                ]
        ),
        gus.Bezier(
                degrees=[3],
                control_points=[
                        [.5, .5, 0.], [.5, 0, .5], [.5, 1., .5], [.5, .5, 1.]
                ]
        ),
        gus.Bezier(
                degrees=[3],
                control_points=[
                        [0.5, 0, .5], [1, .5, .5], [0, 0.5, .5], [.5, 1, .5]
                ]
        )
]
generator.tiling = [1, 2, 3]
gus.show.show_vedo(
        [*generator.create(), generator.deformation_function],
        surface_alpha=0.3,
        knots=False,
        control_points=False,
        title="3D Lattice Microstructure"
)

# Fourth test
generator = gus.spline.microstructure.generator.Generator()
generator.deformation_function = gus.Bezier(
        degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
).create.extruded(extrusion_vector=[0, 0, 1])
generator.microtile = gus.spline.microstructure.tiles.CrossTile3D()
generator.tiling = [2, 2, 3]
gus.show.show_vedo(
        [*generator.create(), generator.deformation_function],
        surface_alpha=0.3,
        control_points=False,
        resolutions=2
)

# Fifth test
# Non-uniform tiling
generator = gus.spline.microstructure.generator.Generator()
generator.deformation_function = gus.BSpline(
        degrees=[2, 1],
        control_points=[
                [0, 0], [.1, 0], [.2, 0], [1, 0], [0, 1], [.1, 1], [.2, 2],
                [1, 3]
        ],
        knot_vectors=[[0, 0, 0, 0.15, 1, 1, 1], [0, 0, 1, 1]]
)
generator.microtile = [
        gus.Bezier(
                degrees=[3],
                control_points=[[0, .5], [.5, 1], [.5, 0], [1, .5]]
        ),
        gus.Bezier(
                degrees=[3],
                control_points=[[0.5, 0], [1, .5], [0, 0.5], [.5, 1]]
        )
]
generator.tiling = [5, 1]
gus.show.show_vedo(
        [
                *generator.create(knot_span_wise=False),
                generator.deformation_function
        ],
        surface_alpha=0.3,
        control_points=False
)

# Fifth test
# A Parametrized microstructure and its respective inverse structure


def foo(x):
    """
    Parametrization Function (determines thickness)
    """
    return tuple([(x[:, 0]) * .05 + (x[:, 1]) * .05 + (x[:, 2]) * .1 + .1])


generator = gus.spline.microstructure.Generator()
generator.deformation_function = gus.Bezier(
        degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
).create.extruded(extrusion_vector=[0, 0, 1])
generator.microtile = gus.spline.microstructure.tiles.InverseCrossTile3D()
generator.tiling = [2, 2, 3]
generator.parametrization_function = foo

inverse_microstructure = generator.create(
        closing_face="z", seperator_distance=0.4, center_expansion=1.3
)

# Corresponding Structure
generator.microtile = gus.spline.microstructure.tiles.CrossTile3D()
microstructure = generator.create(
        closing_face="z", seperator_distance=0.4, center_expansion=1.3
)

# Plot the results
_, showables_inverse = gus.show.show_vedo(
        inverse_microstructure,
        title="Parametrized Inverse Microstructure",
        control_points=False,
        knots=True,
        return_show_list=True,
        resolutions=5
)
_, showables = gus.show.show_vedo(
        microstructure,
        title="Parametrized Microstructure",
        control_points=False,
        knots=True,
        return_show_list=True,
        resolutions=5
)
# Change the color of the inverse mesh to some blue shade
for mesh in showables_inverse:
    from vedo import colors, Mesh
    if isinstance(mesh, Mesh):
        mesh.c(colors.getColor(rgb=(0, 102, 153)))

gus.show.show_vedo(
        sum(showables[1:] + showables_inverse, showables[0]),
        title="Parametrized Microstructure and its inverse"
)
