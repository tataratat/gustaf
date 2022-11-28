"""gustaf/create/volumes.py
Routines to create volumes.
"""

import numpy as np
import random

from gustaf.volumes import Volumes
from gustaf import utils
from gustaf import create

def hexa_block_mesh(
        bounds = [[0., 0., 0.], [1., 1., 1.]],
        resolutions = [2, 2, 2]
        ):
    """
    Create structured hexahedron block mesh.
    Parameters
    -----------
    bounds: (2, 3) array
        Minimum and maximum coordinates.
    resolutions: (3) array
        Vertex count in each dimension.
    create_vertex_groups: bool
    create_face_groups: bool
    Returns
    --------
    volume_mesh: Volumes
    """
    assert np.array(bounds).shape == (2, 3), \
            "bounds array must have 2x3 entries."
    assert len(resolutions) == 3, \
            "resolutions array must have three entries."
    assert np.greater(resolutions, 1).all(), \
            "All resolutions must be at least 2."

    vertex_mesh = create.vertices.raster(bounds, resolutions)

    connectivity = utils.connec.make_hexa_volumes(resolutions)
    volume_mesh = Volumes(vertex_mesh.vertices, connectivity)

    return volume_mesh