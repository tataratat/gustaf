"""gustaf/create/vertices.py.

Routines to create vertices.
"""

import numpy as np

from gustaf.utils.arr import enforce_len
from gustaf.vertices import Vertices


def raster(
    bounds,
    resolutions,
):
    """Simple wrapper of np.mgrid to extract raster points of desired bounds
    and resolutions.

    Parameters
    -----------
    bounds: (2, d) array-like
      float
    resolutions: (d,) array-like or int
      int. It will be casted to int.
      In case int is given, it will be repeated to match the length of
      each bounds

    Returns
    --------
    raster_vertices: Vertices
    """
    if isinstance(resolutions, (int, float)):
        resolutions = enforce_len(resolutions, len(bounds[0]))

    if len(resolutions) != len(bounds[0]) == len(bounds[1]):
        raise ValueError("Length of resolutions and bounds should match.")

    slices = []
    for b0, b1, r in zip(bounds[0], bounds[1], resolutions):
        slices.append(slice(b0, b1, int(r) * 1j))

    # Organize it nicely: 2D np.ndarray with shape (prod(resolutions), dim)
    points = np.mgrid[slices].T.reshape(-1, len(resolutions))

    return Vertices(points)
