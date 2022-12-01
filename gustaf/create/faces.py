"""gustaf/create/faces.py
Routines to create faces.
"""

import numpy as np

from gustaf.faces import Faces
from gustaf import utils, create
from gustaf import settings


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

def simplexify(quad, backslash=False, alternate=True):
    """
    Given quad faces, diagonalize them to turn them into triangles.
    If quad is counterclockwiese (CCW), triangle will also be CCW and 
    vice versa.
    Default diagonalization looks like this:
    (3) *---* (2)
        |  /|
        | / |
        |/  |
    (0) *---* (1)
    resembling 'slash'.
    If you want to diagonalize the other way, set `backslash=True`.
    The algorithm will try to alternate the diagonal directions of neighboring
    elements, if `alternate=True`. This will only work well if the elements are
    ordered in a suitable manner.
    Parameters
    ----------
    quad: Faces
    backslash: bool
    alternate: bool
    Returns
    --------
    tri: Faces
    """
    whatami = quad.whatami
    if not whatami.startswith("quad"):
        raise ValueError(
            "Input to simplexify needs to be a quad mesh, but it's "
            + whatami
        )

    if alternate == True:
        utils.log.warning("Be aware that even though `alternate=True` was "
                "set, the diagonals might not alternate direction as "
                "expected, depending on mesh structure and element order.")

    # split variants
    split_slash = [[0, 1, 2], [2, 3, 0]]
    split_backslash = [[0, 1, 3], [3, 1, 2]]

    quad_faces = quad.faces
    tf_half = int(quad_faces.shape[0])
    tri_faces = np.full((tf_half * 2, 3), -1, dtype=settings.INT_DTYPE)

    if not alternate:
        split = split_backslash if backslash else split_slash

        tri_faces[:tf_half] = quad_faces[:, split[0]]
        tri_faces[tf_half:] = quad_faces[:, split[1]]
    else:
        split_fav = split_backslash if backslash else split_slash
        split_alt = split_slash if backslash else split_backslash

        split_fav_intersections = np.intersect1d(split_fav[0], split_fav[1],
                assume_unique=True)

        intersection_vertices = np.full(quad.vertices.shape[0], False)
        for quad_index, quad_face in enumerate(quad_faces):
            element_intersection_vertices = quad_face[
                    intersection_vertices[quad_face]]
            if not len(element_intersection_vertices):
                split = split_fav
            else:
                split = split_fav if np.isin(
                        element_intersection_vertices,
                        quad_face[split_fav_intersections],
                        assume_unique=True).any() else split_alt

            # would be more efficient here to work with a pre-calculated
            # intersection of split[0] and split[1]
            new_intersection_vertices = quad_face[
                    np.intersect1d(split[0], split[1],
                    assume_unique=True)]
            intersection_vertices[new_intersection_vertices] = True

            tri_faces[2 * quad_index] = quad_face[split[0]]
            tri_faces[2 * quad_index + 1] = quad_face[split[1]]

    tri = Faces(
        vertices=quad.vertices.copy(),
        faces=tri_faces,
    )
    return tri

def triangle(
        bounds = [[0, 0], [1, 1]],
        resolutions = [2, 2],
        alternate_diagonals = True
        ):
    """
    Create structured triangle block mesh.
    This combines `box` and `simplexify`.
    Parameters
    -----------
    bounds: (2, 2) array
        Minimum and maximum coordinates.
    resolutions: (2) array
        Vertex count in each dimension.
    alternate_diagonals : bool
        (Default: True) Alternate diagonals during simplexification.
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

    # create quad mesh as basis
    quad_mesh = box(
            bounds=bounds,
            resolutions=resolutions
            )

    # turn into triangles
    tri_mesh = simplexify(quad_mesh,
            alternate=alternate_diagonals)

    return tri_mesh

