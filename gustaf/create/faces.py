"""gustaf/create/faces.py

Routines to create faces.
"""

import numpy as np

from gustaf.faces import Faces
from gustaf.utils import log
from gustaf import settings


def simplexify(quad, backslash=False, alternate=True):
    """
    Given quad faces, diagonalize them to turn them into triangles.
    If quad is CCW, triangle will also be CCW and vice versa.
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
    if quad.get_whatami() != "quad":
        raise ValueError(
            "Input to simplexify needs to be a quad mesh, but it's "
            + quad.get_whatami()
        )

    if alternate == True:
        log.warning("Be aware that even though `alternate=True` was set, "
                "the diagonals might not alternate direction as expected, "
                "depending on mesh structure and element order.")

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

    # since the vertices are identical, copy vertex groups
    for group_name, group_vertex_ids in quad.vertex_groups.items():
        tri.vertex_groups[group_name] = group_vertex_ids

    # create matching face groups
    tri_face_ids = np.arange(tri_faces.shape[0]).reshape(-1, 2)
    for group_name, group_face_ids in quad.face_groups.items():
        tri.face_groups[group_name] = tri_face_ids[group_face_ids].flatten()

    return tri
