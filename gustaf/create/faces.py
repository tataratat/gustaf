"""gustaf/create/faces.py
Routines to create faces.
"""

import numpy as np

from gustaf.faces import Faces
from gustaf import utils, create
from gustaf import settings

def quad_block_mesh(
        bounds = [[0, 0], [1, 1]],
        resolutions = [2, 2]
        ):
    """
    Create structured quadrangle block mesh.
    Parameters
    -----------
    bounds: (2, 2) array
        Minimum and maximum coordinates.
    resolutions: (2) array
        Vertex count in each dimension.
    Returns
    --------
    face_mesh: Volumes
    """
    assert np.array(bounds).shape == (2, 2), \
            "bounds array must have 2x2 entries."
    assert len(resolutions) == 2, \
            "resolutions array must have two entries."
    assert np.greater(resolutions, 1).all(), \
            "All resolutions must be at least 2."

    vertex_mesh = create.vertices.raster(bounds, resolutions)
    connectivity = utils.connec.make_quad_faces(resolutions)
    face_mesh = Faces(vertex_mesh.vertices, connectivity)

    return face_mesh