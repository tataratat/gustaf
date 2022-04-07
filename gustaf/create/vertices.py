"""gustaf/create/vertices.py

Routines to create vertices.
"""

import numpy as np

from gustaf.vertices import Vertices


def raster(
        bounds,
        resolutions,
):
    """
    Simple wraper of np.mgrid to extract raster points of desired bounds and
    resolutions.

    Parameters
    -----------
    bounds: (2, d) array-like
      float
    resolutions: (d,) array-like
      int. It will be casted to int.

    Returns
    --------
    raster_vertices: Vertices
    """
    if len(resolutions) != len(bounds[0]) == len(bounds[1]):
        raise ValueError("Length of resolutions and bounds should match.")

    points = "np.mgrid["
    for i, (b0, b1) in enumerate(zip(bounds[0], bounds[1])):
        points += f"{b0}:{b1}:{int(resolutions[i])}j,"
    points += "]"

    # Organize it nicely: 2D np.ndarray with shape (prod(resolutions), dim)
    points = eval(points).T.reshape(-1, len(resolutions))

    return Vertices(points)
