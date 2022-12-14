"""gustaf/create/faces.py
Routines to create faces.
"""

import numpy as np

from gustaf.faces import Faces
from gustaf import utils
from gustaf import create
from gustaf import settings


def box(
        bounds=[[0, 0], [1, 1]],
        resolutions=[2, 2],
        simplex=False,
        backslash=False
):
    """Create structured quadrangle or triangle block mesh.

    Parameters
    -----------
    bounds: (2, 2) array
        Minimum and maximum coordinates.
    resolutions: (2) array
        Vertex count in each dimension.
    simplex: boolean
        If true, Mesh will be triangular (simplex).

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

    if simplex:
        # create quad mesh as basis
        quad_mesh = box(bounds=bounds, resolutions=resolutions)

        # turn into triangles
        face_mesh = tosimplex(quad_mesh, backslash)

    else:
        vertex_mesh = create.vertices.raster(bounds, resolutions)
        connectivity = utils.connec.make_quad_faces(resolutions)
        face_mesh = Faces(vertex_mesh.vertices, connectivity)

    return face_mesh


def tosimplex(quad, backslash=False):
    """Given quad faces, diagonalize them to turn them into triangles.

    If quad is counterclockwiese (CCW), triangle will also be CCW and
    vice versa. Will return a tri-mesh, if input is triangular.
    Default diagonalization looks like this:

    .. code-block::

        (3) *---* (2)
            |  /|
            | / |
            |/  |
        (0) *---* (1)

    resembling 'slash'.

    .. code-block::

        (3) *---* (2)
            |\\  |
            | \\ |
            |  \\|
        (0) *---* (1)

    resembling 'backslash'.

    If you want to diagonalize the other way, set `backslash=True`.

    Parameters
    ----------
    quad: Faces
        Faces representation which is to be converted from a cubic mesh into a
        simplex mesh.
    backslash: bool

    Returns
    --------
    tri: Faces
        Simplexifyed mesh.
    """

    if not isinstance(quad, Faces):
        raise ValueError(
                "Input to tosimplex needs to be of type Faces, but it's "
                + type(quad)
        )

    if quad.whatami.startswith("quad"):

        # split variants
        split_slash = [[0, 1, 2], [2, 3, 0]]
        split_backslash = [[0, 1, 3], [3, 1, 2]]

        quad_faces = quad.faces
        tf_half = int(quad_faces.shape[0])
        tri_faces = np.full((tf_half * 2, 3), -1, dtype=settings.INT_DTYPE)

        split = split_backslash if backslash else split_slash

        tri_faces[:tf_half] = quad_faces[:, split[0]]
        tri_faces[tf_half:] = quad_faces[:, split[1]]

        tri = Faces(
                vertices=quad.vertices.copy(),
                faces=tri_faces,
        )
    else:
        tri = quad
        utils.log.debug(
                "Non quadrilateral mesh provided, return original"
                " mesh."
        )

    return tri
