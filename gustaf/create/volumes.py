"""gustaf/create/volumes.py

Routines to create volumes.
"""

import numpy as np
import random

from gustaf.volumes import Volumes
from gustaf import utils
from gustaf import create

def hexa_block_mesh(
        bounds = [[0, 0, 0], [1, 1, 1]],
        resolutions = [2, 2, 2],
        create_vertex_groups = True,
        create_face_groups = True
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

    if not create_vertex_groups and not create_face_groups:
        connectivity = utils.connec.make_hexa_volumes(
                resolutions, create_vertex_groups=False)
        volume_mesh = Volumes(vertex_mesh.vertices, connectivity)
    else:
        connectivity, vertex_groups = utils.connec.make_hexa_volumes(
                resolutions, create_vertex_groups=True)
        volume_mesh = Volumes(vertex_mesh.vertices, connectivity)

        if create_vertex_groups:
            for group_name, vertex_ids in vertex_groups.items():
                volume_mesh.vertex_groups[group_name] = vertex_ids
        if create_face_groups:
            face_connectivity = volume_mesh.get_faces()
            for group_name, vertex_ids in vertex_groups.items():
                volume_mesh.face_groups[group_name] = (
                utils.groups.vertex_to_element_group(face_connectivity,
                        vertex_ids))

    return volume_mesh

def extrude_to_tet(
        source,
        thickness = 1.,
        layers = 1,
        randomize = False,
        bottom_group = "bottom",
        top_group = "top"
        ):
    """
    Given triangular or quadrangular faces, create three-dimensional tetrahedral
    volumes.

    The direct extrusion of a triangle is a 6-vertex wedge. This is divided into
    three tetrahedra. If quadrangles are provided, they will be converted to
    triangles.

    The method is deterministic by default. It walks over the flat vertices in
    given order and lifts them up by create volumes on the surrounding faces.
    The `randomize` argument will walk through the vertices in random order to
    create a less structured result. The randomization will be repeated for
    every layer.

    Parameters
    ----------
    source: Faces
    thickness: float
    layers: int
    randomize: bool
    bottom_group: str
        Name of face and vertex group created at the bottom
    top_group: str
        Name of face and vertex group created at the top

    Returns
    --------
    tet: Volumes
    """
    if layers < 1:
        raise ValueError("The number of layers must be >=1.")

    if not source.get_whatami() in ["tri", "quad"]:
        raise ValueError(
            "Input to extrude_to_tet needs to be a tri or quad mesh, but it's "
            + source.get_whatami()
        )

    if source.get_whatami() == "quad":
        utils.log.info("Quadrangle mesh provided to `extrude_to_tet`. "
                "Creating triangles to continue.")
        source = create.faces.simplexify(source)

    # nodes
    number_of_2d_nodes = source.vertices.shape[0]
    number_of_3d_nodes = (layers + 1) * number_of_2d_nodes
    # node coordinates
    vertices_3d = np.zeros([number_of_3d_nodes, 3])
    vertices_3d[:,:2] = np.tile(source.vertices, [layers + 1, 1])
    vertices_3d[:,2] = np.repeat(np.arange(layers + 1) *
            thickness, number_of_2d_nodes)
    # elements
    number_of_2d_faces = source.faces.shape[0]
    number_of_3d_volumes = layers * 3 * number_of_2d_faces
    volumes_3d = np.zeros([number_of_3d_volumes, 4])
    # this is an array that maps from a 2d vertex index to the current upper
    # vertex index. we start with identity.
    top_nodes = np.arange(number_of_2d_nodes)

    # we need a mapping from nodes to elements
    node_elements = dict()
    for vertex_id in range(number_of_2d_nodes):
        node_elements[vertex_id] = list()
    for face_index, face in enumerate(source.faces):
        for vertex_id in face:
            node_elements[vertex_id].append(face_index)

    volume_iterator = 0
    # loop over 2D (flat) nodes
    vertex_range = np.arange(number_of_2d_nodes)
    for layer in range(layers):
        if randomize:
            np.random.shuffle(vertex_range)
        #for flat_vertex_id, face_ids in node_elements.items():
        for flat_vertex_id in vertex_range:
            face_ids = node_elements[flat_vertex_id]
            anchors = np.array([top_nodes[flat_vertex_id], number_of_2d_nodes +
                    top_nodes[flat_vertex_id]])
            # loop over node's 2D elements
            for face_id in face_ids:
                # create element
                volumes_3d[volume_iterator] = np.union1d(anchors,
                        top_nodes[source.faces[face_id]])
                volume_iterator += 1

            # lift last node
            top_nodes[flat_vertex_id] += number_of_2d_nodes

    tet = Volumes(vertices=vertices_3d, volumes=volumes_3d)

    # vertex groups are also extruded
    for vertex_group, vertex_ids in source.vertex_groups.items():
        tet.vertex_groups[vertex_group] = (vertex_ids.repeat(layers + 1) +
                np.tile(np.arange(layers + 1) * number_of_2d_nodes,
                        vertex_ids.shape[0]))
    # two additional vertex groups are added at the bottom and top
    tet.vertex_groups[bottom_group] = np.arange(number_of_2d_nodes)
    tet.vertex_groups[top_group] = (np.arange(number_of_2d_nodes) + 
            layers * number_of_2d_nodes)

    # face groups are kept on the boundaries
    # two additional face groups are added at the bottom and top
    tet.face_groups.import_vertex_group(bottom_group)
    tet.face_groups.import_vertex_group(top_group)

    return tet

