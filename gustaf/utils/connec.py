"""gustaf/gustaf/utils/connec.py.

Useful functions for connectivity operation. Ranging from edges to
volumes. Named connec because connectivity is too long. Would have been
even cooler, if it was palindrome.
"""

import numpy as np

from gustaf import helpers, settings
from gustaf.utils import arr


def tet_to_tri(volumes):
    """Computes tri faces based on following index scheme.

    ``Tetrahedron``

    .. code-block::

        Ref: (node_ind), face_ind

                      (0)
                     _/|
                   _/ 1|
            (1)  _/____| (3)
               _/|    /|
             _/ 0| 2/ 3|
            /____|/____|
          (0)   (2)    (0)

        face_ind | node_ind
        ---------|----------
        0        | 0 2 1
        1        | 1 3 0
        2        | 2 3 1
        3        | 3 2 0

    Parameters
    -----------
    volumes: (n, 4) np.ndarray

    Returns
    --------
    faces: (n * 4, 3) np.ndarray
    """
    volumes = arr.make_c_contiguous(volumes, settings.INT_DTYPE)

    if volumes.ndim != 2 or volumes.shape[1] != 4:
        raise ValueError("Given volumes are not `tet` volumes")

    fpe = 4  # faces per element
    faces = (
        np.ones(((volumes.shape[0] * fpe), 3), dtype=np.int32) * -1
    )  # -1 for safety check

    faces[:, 0] = volumes.ravel()
    faces[::fpe, [1, 2]] = volumes[:, [2, 1]]
    faces[1::fpe, [1, 2]] = volumes[:, [3, 0]]
    faces[2::fpe, [1, 2]] = volumes[:, [3, 1]]
    faces[3::fpe, [1, 2]] = volumes[:, [2, 0]]

    if (faces == -1).any():
        raise ValueError("Something went wrong while computing faces")

    return faces


def hexa_to_quad(volumes):
    """Computes quad faces based on following index scheme.

    ``Hexahedron``

    .. code-block::


                (6)    (7)
                *------*
                |      |
         (6) (2)| 3    |(3)   (7)    (6)
         *------*------*------*------*
         |      |      |      |      |
         | 2    | 0    | 4    | 5    |
         *------*------*------*------*
         (5) (1)|      |(0)   (4)    (5)
                | 1    |
                *------*
                (5)    (4)

        face_ind | node_ind
        ---------|----------
        0        | 1 0 3 2
        1        | 0 1 5 4
        2        | 1 2 6 5
        3        | 2 3 7 6
        4        | 3 0 4 7
        5        | 4 5 6 7

    Parameters
    -----------
    volumes: (n, 8) np.ndarray

    Returns
    --------
    faces: (n * 8, 4) np.ndarray
    """
    volumes = arr.make_c_contiguous(volumes, settings.INT_DTYPE)

    if volumes.ndim != 2 or volumes.shape[1] != 8:
        raise ValueError("Given volumes are not `hexa` volumes")

    fpe = 6  # faces per element
    faces = (
        np.ones(((volumes.shape[0] * fpe), 4), dtype=np.int32) * -1
    )  # -1 for safety check

    faces[::fpe] = volumes[:, [1, 0, 3, 2]]
    faces[1::fpe] = volumes[:, [0, 1, 5, 4]]
    faces[2::fpe] = volumes[:, [1, 2, 6, 5]]
    faces[3::fpe] = volumes[:, [2, 3, 7, 6]]
    faces[4::fpe] = volumes[:, [3, 0, 4, 7]]
    faces[5::fpe] = volumes[:, [4, 5, 6, 7]]

    if (faces == -1).any():
        raise ValueError("Something went wrong while computing faces")

    return faces


def volumes_to_faces(volumes):
    """Guidance function for `tet_to_tri` and `hexa_to_quad`.

    Parameters
    -----------
    volumes: (n, 4) or (m, 8) np.ndarray

    Returns
    --------
    faces: (n*4, 3) or (m*6, 4) np.ndarray
    """
    volumes = arr.make_c_contiguous(volumes, settings.INT_DTYPE)
    if volumes.shape[1] == 4:
        return tet_to_tri(volumes)

    elif volumes.shape[1] == 8:
        return hexa_to_quad(volumes)


def faces_to_edges(faces):
    """Compute edges based on following edge scheme.

    .. code-block::

        Ref: (node_ind), edge_ind

            (0)
             /\\
          0 /  \\2
           /____\\
        (1)  1   (2)

              2
        (3)*-----*(2)
           |     |
         3 |     | 1
        (0)*-----*(1)
              0

    Note: if `edges` index matter for tets, reorder it!

    Parameters
    -----------
    faces: (n, 3) or (n, 4) np.ndarray

    Returns
    --------
    edges: (n * 3, 2) or (n * 4, 2) np.ndarray
    """
    if faces.ndim != 2:
        raise ValueError(
            "Given input has wrong dimension. "
            "The input array for a faces has to be dim of 2"
        )

    num_faces = faces.shape[0]
    vertices_per_face = faces.shape[1]

    num_edges = int(num_faces * vertices_per_face)
    edges = np.ones((num_edges, 2), dtype=np.int32) * -1  # -1 for safety
    edges[:, 0] = faces.ravel()

    for i in range(vertices_per_face):
        # v_ind : corresponding vertex index for i value
        if i == int(vertices_per_face - 1):
            v_ind = 0
        else:
            v_ind = i + 1

        edges[i::vertices_per_face, 1] = faces[:, v_ind]

    # Quick sanity check - No entries are left untouched.
    if (edges == -1).any():
        raise ValueError("There was an error while computing edges.")

    return edges


def range_to_edges(range_, closed=False, continuous=True):
    """Given range, for example (a, b), returns an edge sequence that
    sequentially connects indices. If int is given as range, it is considered
    as (0, value). Used to be called "closed/open_loop_index_train".

    Parameters
    -----------
    range_: list, tuple, or int
    closed: bool
    continuous: bool

    Returns
    --------
    edges: (n, 2) np.ndarray
    """
    if isinstance(range_, int):
        indices = np.arange(range_)
    elif isinstance(range_, (list, tuple)):
        if len(range_) > 2:
            raise ValueError("Input range is too long")

        # keep the arange syntax. If this isn't wanted,
        # change to (ran[0], ran[1]+1)
        indices = np.arange(*range_)

    # closed is ignored
    if not continuous:
        if indices.size % 2 == 1:
            raise ValueError("Ranges should result in even number of indices.")
        return indices.reshape(-1, 2)

    # continuous edges
    indices = np.repeat(indices, 2)
    if closed:
        indices = np.append(indices, indices[0])[1:]
    else:
        indices = indices[1:-1]

    return indices.reshape(-1, 2)


def sequence_to_edges(seq, closed=False):
    """Given a sequence of int, "connect" to turn them into edges.

    Parameters
    -----------
    seq: (n,) array-like
    closed: bool

    Returns
    --------
    edges: (m, 2) np.ndarray
    """
    edges = np.repeat(seq, 2)[1:-1]
    edges = edges.reshape(-1, 2)
    if closed:
        edges = np.vstack((edges, [edges[-1, -1], edges[0, 0]]))

    return edges


def make_quad_faces(resolutions):
    """Given number of nodes per each dimension, returns connectivity
    information of a structured mesh. Counter clock wise connectivity.

    .. code-block::

        (3)*------*(2)
           |      |
           |      |
        (0)*------*(1)

    Parameters
    ----------
    resolutions: list

    Returns
    -------
    faces: (n, 4) np.ndarray
    """
    nnpd = np.asarray(resolutions)  # number of nodes per dimension
    if any(nnpd < 1):
        raise ValueError(f"The number of nodes per dimension is wrong: {nnpd}")

    total_nodes = np.prod(nnpd)
    total_faces = (nnpd[0] - 1) * (nnpd[1] - 1)
    try:
        node_indices = np.arange(total_nodes).reshape(nnpd[1], nnpd[0])
    except ValueError as e:
        raise ValueError(f"Problem with generating node indices. {e}")

    faces = np.ones((total_faces, 4)) * -1

    faces[:, 0] = node_indices[: (nnpd[1] - 1), : (nnpd[0] - 1)].ravel()
    faces[:, 1] = node_indices[: (nnpd[1] - 1), 1 : nnpd[0]].ravel()
    faces[:, 2] = node_indices[1 : nnpd[1], 1 : nnpd[0]].ravel()
    faces[:, 3] = node_indices[1 : nnpd[1], : (nnpd[0] - 1)].ravel()

    if faces.all() == -1:
        raise ValueError("Something went wrong during `make_quad_faces`.")

    return faces.astype(np.int32)


def make_hexa_volumes(resolutions):
    """Given number of nodes per each dimension, returns connectivity
    information of structured hexahedron elements. Counter clock wise
    connectivity.

    .. code-block::

         (7)*-------*(6)
           /|      /|
          / | (5) / |
      (4)*-------*  |
         |  *----|--*(2)
         | /(3)  | /
         |/      |/
      (0)*-------*(1)

    Parameters
    -----------
    resolutions: list

    Returns
    --------
    elements: (n, 8) np.ndarray
    """
    nnpd = np.asarray(resolutions)  # number of nodes per dimension
    if any(nnpd < 1):
        raise ValueError(f"The number of nodes per dimension is wrong: {nnpd}")

    total_nodes = np.prod(nnpd)
    total_volumes = np.prod(nnpd - 1)
    node_indices = np.arange(total_nodes, dtype=np.int32).reshape(nnpd[::-1])

    volumes = np.ones((total_volumes, 8), dtype=np.int32) * int(-1)

    volumes[:, 0] = node_indices[
        : (nnpd[2] - 1), : (nnpd[1] - 1), : (nnpd[0] - 1)
    ].ravel()
    volumes[:, 1] = node_indices[
        : (nnpd[2] - 1), : (nnpd[1] - 1), 1 : nnpd[0]
    ].ravel()
    volumes[:, 2] = node_indices[
        : (nnpd[2] - 1), 1 : nnpd[1], 1 : nnpd[0]
    ].ravel()
    volumes[:, 3] = node_indices[
        : (nnpd[2] - 1), 1 : nnpd[1], : (nnpd[0] - 1)
    ].ravel()
    volumes[:, 4] = node_indices[
        1 : nnpd[2], : (nnpd[1] - 1), : (nnpd[0] - 1)
    ].ravel()
    volumes[:, 5] = node_indices[
        1 : nnpd[2], : (nnpd[1] - 1), 1 : nnpd[0]
    ].ravel()
    volumes[:, 6] = node_indices[1 : nnpd[2], 1 : nnpd[1], 1 : nnpd[0]].ravel()
    volumes[:, 7] = node_indices[
        1 : nnpd[2], 1 : nnpd[1], : (nnpd[0] - 1)
    ].ravel()

    if (volumes == -1).any():
        raise ValueError("Something went wrong during `make_hexa_volumes`.")

    return volumes.astype(settings.INT_DTYPE)


def subdivide_edges(edges):
    """Subdivide edges. We assume that mid point is newly added points.

    ``Subdivided Edges``

    .. code-block::

        Edges (Lines)

        Ref: (node_ids), edge_ids

        (0)      (2)      (1)
         *--------*--------*

        edge_ids | node_ids
        ---------|----------
        0        | 0 2
        1        | 2 1

    Parameters
    -----------
    edges: (n, 2) np.ndarray

    Returns
    --------
    subdivided_edges: (n * 2, 2) np.ndarray
    """
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Invalid edges shape!")

    raise NotImplementedError


def subdivide_tri(
    mesh,
    return_dict=False,
):
    """Subdivide triangles. Each triangle is divided into 4 meshes.

    ``Subdivided Faces``

    .. code-block::

        Triangles

        Ref: (node_ind), face_ind

                 (0)
                     _/|
                   _/ 0|
            (3)  _/____|(5)
               _/|    /|
             _/ 1| 3/ 2|
            /____|/____|
          (1)   (4)    (2)

        face_ind | node_ind
        ---------|----------
        0        | 0 3 5
        1        | 1 4 3
        2        | 2 5 4
        3        | 3 4 5

    Parameters
    -----------
    mesh: Mesh
    return_dict: bool

    Returns
    --------
    new_vertices: (n, d) np.ndarray
    subdivided_faces: (m, 3) np.ndarray
    mesh_dict: dict
      iff `return_dict=True`,
      returns dict(vertices=new_vertices, faces=subdivided_faces).
    """
    # This will only pass if the mesh is triangle mesh.
    if mesh.faces.shape[1] != 3:
        raise ValueError("Invalid faces shape!")

    # Form new vertices
    edge_mid_v = mesh.vertices[mesh.unique_edges().values].mean(axis=1)
    new_vertices = np.vstack((mesh.vertices, edge_mid_v))

    subdivided_faces = np.empty(
        (mesh.faces.shape[0] * 4, mesh.faces.shape[1]),
        dtype=np.int32,
    )

    mask = np.ones(subdivided_faces.shape[0], dtype=bool)
    mask[3::4] = False

    # 0th column minus (every 4th row, starting from 3rd row)
    subdivided_faces[mask, 0] = mesh.faces.ravel()

    # Form ids for new vertices
    new_vertices_ids = mesh.unique_edges().inverse + int(mesh.faces.max() + 1)
    # 1st & 2nd columns
    subdivided_faces[mask, 1] = new_vertices_ids
    subdivided_faces[mask, 2] = new_vertices_ids.reshape(-1, 3)[
        :, [2, 0, 1]
    ].ravel()

    # Every 4th row, starting from 3rd row
    subdivided_faces[~mask, :] = new_vertices_ids.reshape(-1, 3)

    if return_dict:
        return dict(
            vertices=new_vertices,
            faces=subdivided_faces,
        )

    else:
        return new_vertices, subdivided_faces


def subdivide_quad(
    mesh,
    return_dict=False,
):
    """Subdivide quads.

    Parameters
    -----------
    mesh: Mesh
    return_dict: bool

    Returns
    --------
    new_vertices: (n, d) np.ndarray
    subdivided_faces: (m, 4) np.ndarray
    mesh_dict: dict
      iff `return_dict=True`,
      returns dict(vertices=new_vertices, faces=subdivided_faces).
    """
    # This will only pass if the mesh is quad mesh.
    if mesh.faces.shape[1] != 4:
        raise ValueError("Invalid faces shape!")

    # Form new vertices
    edge_mid_v = mesh.vertices[mesh.unique_edges().values].mean(axis=1)
    face_centers = mesh.centers()
    new_vertices = np.vstack(
        (
            mesh.vertices,
            edge_mid_v,
            face_centers,
        )
    )

    subdivided_faces = np.empty(
        (mesh.faces.shape[0] * 4, mesh.faces.shape[1]),
        dtype=np.int32,
    )

    subdivided_faces[:, 0] = mesh.faces.ravel()
    subdivided_faces[:, 1] = mesh.unique_edges().inverse + len(mesh.vertices)
    subdivided_faces[:, 2] = np.repeat(
        np.arange(len(face_centers)) + (len(mesh.vertices) + len(edge_mid_v)),
        4,
        # dtype=np.int32,
    )
    subdivided_faces[:, 3] = (
        subdivided_faces[:, 1].reshape(-1, 4)[:, [3, 0, 1, 2]].ravel()
    )

    if return_dict:
        return dict(
            vertices=new_vertices,
            faces=subdivided_faces,
        )

    else:
        return new_vertices, subdivided_faces


def sorted_unique(connectivity, sorted_=False):
    """Given connectivity array, finds unique entries, based on its axis=1
    sorted values. Returned value will be sorted.

    Parameters
    -----------
    connectivity: (n, d) np.ndarray
    sorted_: bool

    Returns
    --------
    unique_info: Unique2DIntegers
    """
    s_connec = connectivity if sorted_ else np.sort(connectivity, axis=1)

    unique_stuff = arr.unique_rows(
        s_connec,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        dtype_name=settings.INT_DTYPE,
    )

    return helpers.data.Unique2DIntegers(
        unique_stuff[0],  # values
        unique_stuff[1],  # ids
        unique_stuff[2],  # inverse
        unique_stuff[3],  # counts
    )
