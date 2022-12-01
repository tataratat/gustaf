"""gustaf/create/volumes.py
Routines to create volumes.
"""

import numpy as np

from gustaf.volumes import Volumes
from gustaf import utils
from gustaf import create


def box(bounds=[[0., 0., 0.], [1., 1., 1.]], resolutions=[2, 2, 2]):
    """
    Create structured hexahedron block mesh.
    Parameters
    -----------
    bounds: (2, 3) array
        Minimum and maximum coordinates.
    resolutions: (3) array
        Vertex count in each dimension.
    Returns
    --------
    volume_mesh: Volumes
    """
    if np.array(bounds).shape != (2, 3):
        raise ValueError("Bounds must have a dimension of (2, 3).")
    if len(resolutions) != 3:
        raise ValueError("Resolutions must have three entries.")
    if not np.greater(resolutions, 1).all():
        raise ValueError("All resolution values must be at least 2.")

    vertex_mesh = create.vertices.raster(bounds, resolutions)

    connectivity = utils.connec.make_hexa_volumes(resolutions)
    volume_mesh = Volumes(vertex_mesh.vertices, connectivity)

    return volume_mesh

def extrude_to_tet(
        source,
        thickness = 1.,
        layers = 1,
        randomize = False
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

    Returns
    --------
    tet: Volumes
    """
    if layers < 1:
        raise ValueError("The number of layers must be >=1.")

    if not source.whatami in ["tri", "quad"]:
        raise ValueError(
            "Input to extrude_to_tet needs to be a tri or quad mesh, but it's "
            + source.whatami
        )

    if source.whatami == "quad":
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

    return tet

def tet_block_mesh(
        bounds = [[0., 0., 0.], [1., 1., 1.]],
        resolutions = [2, 2, 2],
        alternate_diagonals = True,
        randomize_extrusion = False
        ):
    """
    Create structured tetrahedron block mesh.
    This creates a quad mesh on the plane spanned by the first two axes and then
    runs the simplexification and extrusion on it.
    Parameters
    -----------
    bounds: (2, 3) array
        Minimum and maximum coordinates.
    resolutions: (3) array
        Vertex count in each dimension.
    alternate_diagonals : bool
        (Default: True) Try to alternate diagonal direction during
        simplexification.
    randomize_extrusion : bool
        (Default: False) Use randomization during extrusion.
    Returns
    --------
    volume_mesh: Volumes
    """

    if np.array(bounds).shape != (2, 3):
        raise ValueError("Bounds must have a dimension of (2, 3).")
    if len(resolutions) != 3:
        raise ValueError("Resolutions must have three entries.")
    if not np.greater(resolutions, 1).all():
        raise ValueError("All resolution values must be at least 2.")

    # we need to make sure bounds is a numpy array for indexing purposes
    if not isinstance(bounds, np.ndarray):
        bounds = np.array(bounds)

    # create a 2D quad mesh along the first two axes
    quad_mesh = create.faces.box(
            bounds = bounds[:,:2],
            resolutions = resolutions[:2]
            )

    # simplexify
    tri_mesh = create.faces.simplexify(quad_mesh,
            alternate=alternate_diagonals)

    # extrude along third axis
    number_of_layers = resolutions[2] - 1
    tet_mesh = create.volumes.extrude_to_tet(tri_mesh,
            thickness = (bounds[1, 2] - bounds[0, 2])/number_of_layers,
            layers = number_of_layers,
            randomize = randomize_extrusion)

    # move all vertices to correct position along third axis
    tet_mesh.vertices[:,2] += bounds[0, 2]

    return tet_mesh
