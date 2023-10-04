"""gustaf/create/volumes.py
Routines to create volumes.
"""

import numpy as np

from gustaf import create, utils
from gustaf.volumes import Volumes


def box(bounds=None, resolutions=None):
    """Create structured hexahedron block mesh.

    Parameters
    -----------
    bounds: (2, 3) array
        Minimum and maximum coordinates.
    resolutions: (3) array
        Vertex count in each dimension.

    Returns
    --------
    volume_mesh: Volumes
    """

    if resolutions is None:
        resolutions = [2, 2, 2]
    if bounds is None:
        bounds = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    if np.array(bounds).shape != (2, 3):
        raise ValueError("Bounds must have a dimension of (2, 3).")
    if len(resolutions) != 3:
        raise ValueError("Resolutions must have three entries.")
    if not np.greater(resolutions, 1).all():
        raise ValueError("All resolution values must be at least 2.")

    vertex_mesh = create.vertices.raster(bounds, resolutions)
    connectivity = utils.connec.make_hexa_volumes(resolutions)
    volume_mesh = Volumes(vertex_mesh.vertices, connectivity)

    return volume_mesh
