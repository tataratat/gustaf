import gustaf as gus

generator = gus.spline.microstructure.generator.Generator()
generator.deformation_function = gus.Bezier(
    degrees=[2, 1],
    control_points=[
        [0, 0], [1, 0], [2, -1], [-1, 1], [1, 1], [3, 2]
    ])
generator.microtile = [
    gus.Bezier(
        degrees=[3],
        control_points=[
            [0, .5], [.5, 1], [.5, 0], [1, .5]
        ]),
    gus.Bezier(
        degrees=[3],
        control_points=[
            [0.5, 0], [1, .5], [0, 0.5], [.5, 1]]
    )]
generator.tiling = [8, 8]
gus.show.show_vedo(
    [*generator.create(), generator.deformation_function],
    surface_alpha=0.3,
    knots=False,
    control_points=False,
    title="2D Lattice Microstructure"
)

# # second test
# generator = gus.spline.microstructure.generator.Generator()
# generator.deformation_function = gus.Bezier(degrees=[1, 1], control_points=[
#                                             [0, 0], [1, 0], [0, 1], [1, 1]]).create.extruded(extrusion_vector=[0, 0, 1])
# generator.microtile = [gus.Bezier(degrees=[3], control_points=[
#     [0, .5, .5], [.5, 1, .5], [.5, 0, .5], [1, .5, .5]]), gus.Bezier(degrees=[3], control_points=[
#         [.5, .5,  0.], [.5, 0, .5], [.5, 1., .5], [.5, .5, 1.]]),
#     gus.Bezier(degrees=[3], control_points=[
#         [0.5, 0, .5], [1, .5, .5], [0, 0.5, .5], [.5, 1, .5]])]
# generator.tiling = [1, 2, 3]
# gus.show.show_vedo(
#     [*generator.create(), generator.deformation_function], surface_alpha=0.3)

# # Third test
# generator = gus.spline.microstructure.generator.Generator()
# generator.deformation_function = gus.Bezier(degrees=[1, 1], control_points=[
#                                             [0, 0], [1, 0], [0, 1], [1, 1]]).create.extruded(extrusion_vector=[0, 0, 1])
# generator.microtile = gus.spline.microstructure.cross_tile.CrossTile()
# generator.tiling = [2, 2, 3]
# gus.show.show_vedo(
#     [*generator.create(), generator.deformation_function], surface_alpha=0.3, control_points=False)

# Fourth test


def foo(x):
    return tuple([x[:, 0] * 0.1 + x[:, 1] * 0.1 + (1-+ x[:, 2]) * .1])


generator = gus.spline.microstructure.generator.Generator()
generator.deformation_function = gus.Bezier(degrees=[1, 1],
                                            control_points=[
                                                [0, 0], [1, 0], [0, 1], [1, 1]]
                                            ).create.extruded(
                                                extrusion_vector=[0, 0, 1])
generator.microtile = gus.spline.microstructure.cross_tile.InverseCrossTile()
generator.tiling = [2, 2, 3]
generator.parametrization_function = foo

gus.show.show_vedo(
    [*generator.create(closing_faces=2, seperator_distance=0.4)], title="Parametrized Microstructure", control_points=False, knots=False)
