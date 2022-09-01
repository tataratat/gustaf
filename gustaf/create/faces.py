"""gustaf/create/faces.py

Routines to create faces.
"""

import numpy as np

from gustaf.faces import Faces
from gustaf import utils, create
from gustaf import settings

def quad_block_mesh(
        bounds = [[0, 0], [1, 1]],
        resolutions = [2, 2],
        create_vertex_groups = True,
        create_edge_groups = True
        ):
    """
    Create structured quadrangle block mesh.

    Parameters
    -----------
    bounds: (2, 2) array
        Minimum and maximum coordinates.
    resolutions: (2) array
        Vertex count in each dimension.
    create_vertex_groups: bool
    create_face_groups: bool

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

    if not create_vertex_groups and not create_face_groups:
        connectivity = utils.connec.make_quad_faces(
                resolutions, create_vertex_groups=False)
        face_mesh = Faces(vertex_mesh.vertices, connectivity)
    else:
        connectivity, vertex_groups = utils.connec.make_quad_faces(
                resolutions, create_vertex_groups=True)
        face_mesh = Faces(vertex_mesh.vertices, connectivity)

        if create_vertex_groups:
            for group_name, vertex_ids in vertex_groups.items():
                face_mesh.vertex_groups[group_name] = vertex_ids
        if create_edge_groups:
            edge_connectivity = face_mesh.get_edges()
            for group_name, vertex_ids in vertex_groups.items():
                face_mesh.edge_groups[group_name] = (
                utils.groups.vertex_to_element_group(edge_connectivity,
                        vertex_ids))

    return face_mesh

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

    # since the vertices are identical, copy vertex groups
    for group_name, group_vertex_ids in quad.vertex_groups.items():
        tri.vertex_groups[group_name] = group_vertex_ids

    # create matching face groups
    tri_face_ids = np.arange(tri_faces.shape[0]).reshape(-1, 2)
    for group_name, group_face_ids in quad.face_groups.items():
        tri.face_groups[group_name] = tri_face_ids[group_face_ids].flatten()

    # preserve edge groups
    number_of_quad_edges = 4
    number_of_tri_edges = 3
    number_of_edge_nodes = 2
    # we need the quad face/edge connectivity in a way that allows us to find it
    # by global quad edge ID
    quad_edges = quad.get_edges_sorted()
    # we need the tri face/edge connectivity in a way that allows us to find it
    # by tri face ID
    tri_edges = tri.get_edges_sorted().reshape(-1,
            number_of_tri_edges, number_of_edge_nodes)
    for group_name, group_edge_ids in quad.edge_groups.items():
        group_quad_face_ids = group_edge_ids // number_of_quad_edges
        group_tri_face_ids = tri_face_ids[group_quad_face_ids].flatten()
        # tri face/edge connectivity
        group_tri_edges = tri_edges[group_tri_face_ids].reshape(-1,
                2 * number_of_tri_edges, number_of_edge_nodes)
        # group edge connectivity
        # (the shape needs to sort of match that of the tri connectivity)
        group_edges = quad_edges[group_edge_ids].reshape(-1, 1,
                number_of_edge_nodes)
        # find tri edges that match the quad edges
        # (matches[:,0] are tri indices, matches[:,1] are local edge IDs)
        matches = np.argwhere(np.equal(group_tri_edges, group_edges).all(
            axis=2).reshape(-1, number_of_tri_edges))
        # get tri edge IDs
        tri.edge_groups[group_name] = group_tri_face_ids[
                matches[:, 0]] * number_of_tri_edges + matches[:, 1]

    return tri

def tri_block_mesh(
        bounds = [[0, 0], [1, 1]],
        resolutions = [2, 2],
        create_vertex_groups = True,
        create_edge_groups = True,
        alternate_diagonals = True
        ):
    """
    Create structured triangle block mesh.

    This combines `quad_block_mesh` and `simplexify`.

    Parameters
    -----------
    bounds: (2, 2) array
        Minimum and maximum coordinates.
    resolutions: (2) array
        Vertex count in each dimension.
    create_vertex_groups: bool
    create_face_groups: bool
    alternate_diagonals : bool
        (Default: True) Alternate diagonals during simplexification.

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

    # create quad mesh as basis
    quad_mesh = quad_block_mesh(
            bounds=bounds,
            resolutions=resolutions,
            create_vertex_groups=create_vertex_groups,
            create_edge_groups=create_edge_groups
            )

    # turn into triangles
    tri_mesh = simplexify(quad_mesh,
            alternate=alternate_diagonals)

    return tri_mesh

