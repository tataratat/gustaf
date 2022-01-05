"""gustav/gustav/utils/connec.py

Useful functions for connectivity operation. Ranging from edges to volumes.
Named connec because connectivity is too long. Would have been even cooler,
if it was palindrome.
"""

import numpy as np

from gustav import settings
from gustav.utils import arr

def tet_to_tri(volumes):
    """
    Computes tri faces based on following index scheme.

    ``Tetrahedron``

    .. code-block::

        Ref: (node_ind), face_ind

                 (0)
                 /\ 
                / 1\ 
            (1)/____\(3)
              /\    /\ 
             / 0\ 2/ 3\ 
            /____\/____\ 
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

    if volumes.shape[1] != 4:
        raise ValueError("Given volumes are not `tet` volumes")

    fpe = 4 # faces per element
    faces = np.ones(
        ((volumes.shape[0] * fpe), 3),
        dtype=np.int32
    ) * -1 # -1 for safety check

    faces[:, 0] = volumes.flatten()
    faces[::fpe, [1, 2]] = volumes[:, [2, 1]]
    faces[1::fpe, [1, 2]] = volumes[:, [3, 0]]
    faces[2::fpe, [1, 2]] = volumes[:, [3, 1]]
    faces[3::fpe, [1, 2]] = volumes[:, [2, 0]]

    if (faces == -1).any():
        raise ValueError("Something went wrong while computing faces")

    return faces


def hexa_to_quad(volumes):
    """
    Computes quad faces based on following index scheme.

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

    if volumes.shape[1] != 8:
        raise ValueError("Given volumes are not `hexa` volumes")

    fpe = 6 # faces per element
    faces = np.ones(
        ((volumes.shape[0] * fpe), 4),
        dtype=np.int32
    ) * -1 # -1 for safety check

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
    """
    Guidance function for `tet_to_tri` and `hexa_to_quad`.

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
    """
    Compute edges based on following edge scheme.

    .. code-block::

        Ref: (node_ind), edge_ind

             (0)
             /\ 
          0 /  \ 2
           /____\ 
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
    num_faces = faces.shape[0]
    vertices_per_face = faces.shape[1]
           
    num_edges = int(num_faces * vertices_per_face)
    edges = np.ones((num_edges, 2), dtype=np.int32) * -1 # -1 for safety
    edges[:,0] = faces.flatten()

    for i in range(vertices_per_face):
        # v_ind : corresponding vertex index for i value
        if i == int(vertices_per_face - 1):
            v_ind = 0
        else:
            v_ind = i + 1

        edges[i::vertices_per_face, 1] = faces[:,v_ind]

    # Quick sanity check - No entries are left untouched.
    if (edges == -1).any():
        raise ValueError("There was an error while computing edges.")

    return edges


def range_to_edges(range_, closed=False):
    """
    Given range, for example (a, b), returns an edge sequence that
    sequentially connects indices. If int is given as range, it is considered
    as (0, value).
    Used to be called "closed/open_loop_index_train".

    Parameters
    -----------
    range_: list, tuple, or int
    closed: bool

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

    indices = np.repeat(indices, 2)
    if closed:
        indices = np.append(indices, indices[0])[1:]
    else:
        indices = indices[1:-1]

    return indices.reshape(-1, 2)


def sequence_to_edges(seq, closed=False):
    """
    Given a sequence of int, "connect" to turn them into edges.

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
        edges = np.vstack(
            (
                edges,
                [edges[-1, -1], edges[0, 0]]
            )
        )

    return edges


def make_quad_faces(resolutions):
    """
    Given number of nodes per each dimension, returns connectivity information 
    of a structured mesh.
    Counter clock wise connectivity.

    (3)*------*(2)
       |      |
       |      |
    (0)*------*(1)

    Parameters
    -----------
    resolutions: list

    Returns
    --------
    faces: (n, 4) np.ndarray
    """
    nnpd = np.asarray(resolutions) # number of nodes per dimension
    total_nodes = np.product(nnpd)
    total_faces = (nnpd[0] - 1) * (nnpd[1] - 1)
    node_indices = np.arange(total_nodes).reshape(nnpd[1], nnpd[0])
    faces = np.ones((total_faces, 4)) * -1

    faces[:,0] = node_indices[:(nnpd[1] - 1), :(nnpd[0] - 1)].flatten()
    faces[:,1] = node_indices[:(nnpd[1] - 1), 1:nnpd[0]].flatten()
    faces[:,2] = node_indices[1:nnpd[1], 1:nnpd[0]].flatten()
    faces[:,3] = node_indices[1:nnpd[1], :(nnpd[0]-1)].flatten()

    if faces.all() == -1:
        raise ValueError("Something went wrong during `make_quad_faces`.")

    return faces.astype(np.int32)


def make_hexa_volumes(resolutions):
    """
    Given number of nodes per each dimension, returns connectivity information 
    of structured hexahedron elements.
    Counter clock wise connectivity.

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
    nnpd = np.asarray(resolutions) # number of nodes per dimension
    total_nodes = np.product(nnpd)
    total_volumes = np.product(nnpd - 1)
    node_indices = np.arange(total_nodes, dtype=np.int32).reshape(nnpd[::-1])
    volumes = np.ones((total_volumes, 8), dtype=np.int32) * int(-1)

    volumes[:, 0] = node_indices[
        :(nnpd[2] - 1),
        :(nnpd[1] - 1),
        :(nnpd[0] - 1)
    ].flatten()
    volumes[:, 1] = node_indices[
        :(nnpd[2] - 1),
        :(nnpd[1] - 1),
        1:nnpd[0]
    ].flatten()
    volumes[:, 2] = node_indices[
        :(nnpd[2] - 1),
        1:nnpd[1],
        1:nnpd[0]
    ].flatten()
    volumes[:, 3] = node_indices[
        :(nnpd[2] - 1),
        1:nnpd[1],
        :(nnpd[0]-1)
    ].flatten()
    volumes[:, 4] = node_indices[
        1:nnpd[2],
        :(nnpd[1] - 1),
        :(nnpd[0] - 1)
    ].flatten()
    volumes[:, 5] = node_indices[
        1:nnpd[2],
        :(nnpd[1] - 1),
        1:nnpd[0]
    ].flatten()
    volumes[:, 6] = node_indices[
        1:nnpd[2],
        1:nnpd[1],
        1:nnpd[0]
    ].flatten()
    volumes[:, 7] = node_indices[
        1:nnpd[2],
        1:nnpd[1],
        :(nnpd[0]-1)
    ].flatten()

    if (volumes == -1).any():
        raise ValueError("Something went wrong during `make_hexa_volumes`.")

    return volumes.astype(settings.INT_DTYPE)
 
