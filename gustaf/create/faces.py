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

    return 