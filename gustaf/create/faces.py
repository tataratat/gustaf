"""gustaf/create/faces.py

Routines to create faces.
"""

import numpy as np

from gustaf.faces import Faces


def quad_to_tri(quad, backslash=False, alternate=True):
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
            "Input to quad_to_tri needs to be a quad mesh, but it's "
            + quad.get_whatami()
        )

    # split variants
    split_slash = [[0, 1, 2], [2, 3, 0]]
    split_backslash = [[0, 1, 3], [3, 1, 2]]

    quad_faces = quad.faces
    tf_half = int(quad_faces.shape[0])
    tri_faces = np.ones((tf_half * 2, 3), dtype=np.int32) * -1

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
            element_intersection_vertices =\
                    quad_face[intersection_vertices[quad_face]]
            if not len(element_intersection_vertices):
                split = split_fav
            else:
                split = split_fav if np.isin(
                        element_intersection_vertices,
                        quad_face[split_fav_intersections],
                        assume_unique=True).any() \
                else split_alt

            # would be more efficient here to work with a pre-calculated
            # intersection of split[0] and split[1]
            new_intersection_vertices =\
                    quad_face[np.intersect1d(split[0], split[1],
                    assume_unique=True)]
            intersection_vertices[new_intersection_vertices] = True

            tri_faces[quad_index] = quad_face[split[0]]
            tri_faces[quad_index + tf_half] = quad_face[split[1]]

    tri = Faces(
        vertices=quad.vertices.copy(),
        faces=tri_faces,
    )

    return tri
