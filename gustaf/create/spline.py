"""gustaf/create/spline.py.

Frequently used spline shapes generation.
"""

import numpy as np

from gustaf.create.vertices import raster
from gustaf.spline.base import BSpline


def with_bounds(
        parametric_bounds,
        physical_bounds,
        degrees=None,
        num_unique_knots=None,
        nurbs=False,
):
    """Creates spline with given parametric bounds, physical bounds, degrees,
    num_unique_knots. Physical bounds can have less or equal number of
    dimension as parametric bounds. (Greater is not supported)

    Parameters
    -----------
    parametric_bounds: (2, n) array-like
    physical_bounds: (2, n) array-like
    degrees: (n,) array-like
      (Optional) Default is 1 for each dimension.
    num_unique_knots: (n,) array-like
      (Optional) Default is 2 for each dimension.
    nurbs: bool
      (Optional) Default is False.

    Returns
    --------
    spline: BSpline or NURBS
      If `spline` is not availabe, will return dict of corresponding
    """
    # First, prepare for degree 1 spline.
    # KV
    l_bound, u_bound = parametric_bounds
    assert len(l_bound) == len(u_bound),\
        "Length of lower and upper parametric_bounds aren't identical"
    kvs = [[lb, lb, u, u] for lb, u in zip(l_bound, u_bound)]

    # CP
    pl_bound, pu_bound = physical_bounds
    assert len(pl_bound) == len(pu_bound),\
        "Length of upper and lower physical_bounds aren't identical"
    dim_diff = len(l_bound) - len(pl_bound)
    if dim_diff < 0:
        raise ValueError(
                "Sorry, we don't support spline generation with phys_dim > "
                "para_dim."
        )

    cps = raster(physical_bounds, [2] * len(pl_bound)).vertices
    cps = np.repeat(cps, dim_diff + 1, axis=0)

    # Now, make spline
    spl = BSpline(
            knot_vectors=kvs,
            control_points=cps,
            degrees=[1] * len(l_bound),
    )

    # Return early if there's nothing left to do
    if degrees is None and num_unique_knots is None:
        return spl

    # Manipulate to satisfy degrees and num_unique_knots
    # Degrees
    for i, d in enumerate(degrees):
        diff = int(d - 1)
        # continue if desired degree is 1
        if diff == 0:
            continue
        # Elevate degree
        for _ in range(diff):
            spl.elevate_degree(i)

    # Return is there's nothing to do
    if num_unique_knots is None:
        return spl if not nurbs else spl.nurbs

    # Knots
    assert len(num_unique_knots) == len(l_bound),\
        "Length of num_unique_knots and parametric_bounds[0] does not match."
    for i, (nuk, lb, ub) in enumerate(zip(num_unique_knots, l_bound, u_bound)):
        if nuk < .5:
            continue
        new_knots = np.linspace(lb, ub, nuk)[1:-1]
        spl.insert_knots(i, new_knots)

    return spl if not nurbs else spl.nurbs


def with_parametric_bounds(
        parametric_bounds,
        degrees=None,
        num_unique_knots=None,
        nurbs=False,
):
    """Similar to with_bounds. Creates spline that has same parametric and
    physical bounds.

    Parameters
    -----------
    parametric_bounds: (2, n) array-like
    degrees: (n,) array-like
      (Optional) Default is 1 for each dimension.
    num_unique_knots: (n,) array-like
      (Optional) Default is 2 for each dimension.
    nurbs: bool
      (Optional) Default is False.

    Returns
    --------
    spline: BSpline or NURBS
      If `spline` is not availabe, will return dict of corresponding
    """
    return with_bounds(
            parametric_bounds=parametric_bounds,
            physical_bounds=parametric_bounds,
            degrees=degrees,
            num_unique_knots=num_unique_knots,
            nurbs=nurbs,
    )


def with_physical_bounds(
        physical_bounds,
        degrees=None,
        num_unique_knots=None,
        nurbs=None,
):
    """Similar to with_bounds. Creates spline that has same [0, 1]^3 parametric
    bounds.

    Parameters
    -----------
    physical_bounds: (2, n) array-like
    degrees: (n,) array-like
      (Optional) Default is 1 for each dimension.
    num_unique_knots: (n,) array-like
      (Optional) Default is 2 for each dimension.
    nurbs: bool
      (Optional) Default is False.

    Returns
    --------
    spline: BSpline or NURBS
      If `spline` is not availabe, will return dict of corresponding
    """
    dim = len(physical_bounds[0])

    return with_bounds(
            parametric_bounds=[
                    [0. for _ in range(dim)],
                    [1. for _ in range(dim)],
            ],
            physical_bounds=physical_bounds,
            num_unique_knots=num_unique_knots,
            nurbs=nurbs,
    )


def parametric_view(spline):
    """Create parametric view of given spline. Previously called
    `naive_spline()`. Degrees are always 1 and knot multiplicity is not
    preserved. Returns BSpline, as BSpline and NURBS should look the same.

    Parameters
    -----------
    spline: BSpline or NURBS

    Returns
    --------
    para_spline: BSpline
    """
    para_spline = with_parametric_bounds(
            parametric_bounds=spline.knot_vector_bounds,
            degrees=[1] * spline.para_dim,
            num_unique_knots=None,
            nurbs=False,
    )

    # loop through knot vectors and insert missing knots
    for i, kv in enumerate(spline.unique_knots):
        if len(kv) > 2:
            para_spline.insert_knots(i, kv[1:-1])

    return para_spline
