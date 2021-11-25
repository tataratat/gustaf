"""gustav/gustav/utils/connec.py

Useful functions for connectivity operation. Ranging from edges to volumes.
Named connec because connectivity is too long. Would have been even cooler,
if it was palindrome.
"""

import numpy as np

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
    volumes = utils.arr.make_c_contiguous(volumes, "int32")

    if element.shape[1] != 4:
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
    volumes: (n, 6) np.ndarray

    Returns
    --------
    faces: (n * 6, 4) np.ndarray
    """
    volumes = utils.arr.make_c_contiguous(volumes, "int32")

    if element.shape[1] != 6:
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
