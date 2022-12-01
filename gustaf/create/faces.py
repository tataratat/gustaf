"""gustaf/create/faces.py
Routines to create faces.
"""

import numpy as np

from gustaf.faces import Faces
from gustaf import utils, create


def box(bounds=[[0, 0], [1, 1]], resolutions=[2, 2]):
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
    if np.array(bounds).shape != (2, 2):
        raise ValueError("Bounds must have a dimension of (2, 2).")
    if len(resolutions) != 2:
        raise ValueError("Resolutions must have two entries.")
    if not np.greater(resolutions, 1).all():
        raise ValueError("All resolution values must be at least 2.")

    vertex_mesh = create.vertices.raster(bounds, resolutions)
    connectivity = utils.connec.make_quad_faces(resolutions)
    face_mesh = Faces(vertex_mesh.vertices, connectivity)

    return face_mesh
