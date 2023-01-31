import gustaf as gus

gus.settings.NTHREADS = 1

# Second test
para_s = gus.Bezier(
    degrees=[2, 2],
    control_points=[
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.7],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
    ],
)

deformation_function = gus.Bezier(
    degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
)


def boundary_function(
    ID, accepted_distance, tolerance_projection=gus.settings.TOLERANCE
):
    def_fun_bound = deformation_function.extract_boundaries(ID)[0]

    def is_close_enough(x):
        x_dist = def_fun_bound.proximities(
            x, 10, tolerance_projection, nthreads=gus.settings.NTHREADS
        )[3]
        return x_dist.flatten() < accepted_distance

    return is_close_enough


def parametrization_function(x):
    return tuple([para_s.evaluate(x).flatten()])


generator = gus.spline.microstructure.Microstructure()
generator.deformation_function = deformation_function
generator.microtile = gus.spline.microstructure.tiles.CrossTile2D()
generator.tiling = [10, 10]
generator.parametrization_function = parametrization_function
ms = generator.create(closing_face="y", center_expansion=1.3)

ms_mp = gus.Multipatch(ms)

ms_mp.boundary_from_function(boundary_function(0, 10 * gus.settings.TOLERANCE))
ms_mp.boundary_from_function(boundary_function(1, 10 * gus.settings.TOLERANCE))
ms_mp.boundary_from_function(boundary_function(2, 10 * gus.settings.TOLERANCE))
ms_mp.boundary_from_function(boundary_function(3, 10 * gus.settings.TOLERANCE))

ms_mp.show(boundary_ids=True, resolutions=4, knots=True, control_points=False)

# gus.spline.io.gismo.export("microstructure_2d_gismo.xml", multipatch=ms_mp)
