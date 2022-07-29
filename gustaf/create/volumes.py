"""gustaf/create/volumes.py

Routines to create volumes.
"""

import numpy as np
import random

from gustaf.volumes import Volumes


def extrude_tri_to_tet(tri, thickness = 1., layers = 1, randomize = False):
    """
    Given triangular faces, create three-dimensional tetrahedral volumes.

    The direct extrusion of a triangle is a 6-vertex wedge. This is divided into
    three tetrahedra.

    The method is deterministic by default. It walks over the flat vertices in
    given order and lifts them up by create volumes on the surrounding faces.
    The `randomize` argument will walk through the vertices in random order to
    create a less structured result. The randomization will be repeated for
    every layer.

    Parameters
    ----------
    tri: Faces
    thickness: float
    layers: int
    randomize: bool

    Returns
    --------
    tet: Volumes
    """
    if layers < 1:
        raise ValueError("The number of layers must be >1.")

    if tri.get_whatami() != "tri":
        raise ValueError(
            "Input to extrude_tri_to_tet needs to be a tri mesh, but it's "
            + tri.get_whatami()
        )

    # nodes
    number_of_2d_nodes = tri.vertices.shape[0]
    number_of_3d_nodes = (layers + 1) * number_of_2d_nodes
    # node coordinates
    vertices_3d = np.zeros([number_of_3d_nodes, 3])
    vertices_3d[:,:2] = np.tile(tri.vertices, [layers + 1, 1])
    vertices_3d[:,2] = np.repeat(np.arange(layers + 1) *
            thickness, number_of_2d_nodes)
    # elements
    number_of_2d_faces = tri.faces.shape[0]
    number_of_3d_volumes = layers * 3 * number_of_2d_faces
    volumes_3d = np.zeros([number_of_3d_volumes, 4])
    # this is an array that maps from a 2d vertex index to the current upper
    # vertex index. we start with identity.
    top_nodes = np.arange(number_of_2d_nodes)

    # we need a mapping from nodes to elements
    node_elements = dict()
    for vertex_id in range(number_of_2d_nodes):
        node_elements[vertex_id] = set()
    for face_index, face in enumerate(tri.faces):
        for vertex_id in face:
            node_elements[vertex_id] |= {face_index}

    volume_iterator = 0
    # loop over 2D (flat) nodes
    vertex_range = list(range(number_of_2d_nodes))
    for layer in range(layers):
        if randomize:
            random.shuffle(vertex_range)
        #for flat_vertex_id, face_ids in node_elements.items():
        for flat_vertex_id in vertex_range:
            face_ids = node_elements[flat_vertex_id]
            anchors = {top_nodes[flat_vertex_id], number_of_2d_nodes +
                    top_nodes[flat_vertex_id]}
            # loop over node's 2D elements
            for face_id in face_ids:
                # remove anchor
                remaining_nodes = set(tri.faces[face_id]).difference({flat_vertex_id})
                # add top nodes
                new_element_nodes = anchors | set(top_nodes[list(remaining_nodes)])
                # create element
                volumes_3d[volume_iterator] = list(new_element_nodes)
                volume_iterator += 1

            # lift last node
            top_nodes[flat_vertex_id] += number_of_2d_nodes

    tet = Volumes(vertices=vertices_3d, volumes=volumes_3d)

    return tet
