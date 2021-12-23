
from gustav.create.vertices import raster
from gustav.spline import BSpline


def with_bounds(
        parametric_bounds,
        physical_bounds,
        degrees,
        unique_knots,
):
    """
    """

def with_dimension(
        parametric_dim,
        physical_dim,
        nurbs=False,
):
    """
    Creates zero to one bounded (both physical and parametric space) spline
    based on given dimension.
    """
    kvs = [[0, 0, 1, 1] for _ in range(parametric_dim)]
    physical_bounds = [
        [0 for _ in range(physical_dim)],
        [1 for _ in range(physical_dim)],
    ]
    resolutions = [2 for _ in range(parametric_dim)]
    degrees = [1 for _ in range(parametric_dim)]

    # Prepare cps
    cps = raster(physical_bounds, resolutions)