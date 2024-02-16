"""gustaf/gustaf/utils/connec.py.

Useful functions for connectivity operation. Ranging from edges to
volumes. Named connec because connectivity is too long. Would have been
even cooler, if it was palindrome.
"""

import collections

import numpy as np

try:
    import napf
except ImportError:
    from gustaf.helpers.raise_if import ModuleImportRaiser

    napf = ModuleImportRaiser("napf")

from gustaf import helpers, settings
from gustaf.utils import arr


def tet_to_tri(volumes):
    """Computes tri faces based on following index scheme.

    ``Tetrahedron``

    .. code-block:: text

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
    volumes = np.asanyarray(volumes, settings.INT_DTYPE)

    if volumes.ndim != 2 or volumes.shape[1] != 4:
        raise ValueError("Given volumes are not `tet` volumes")

    fpe = 4  # faces per element
    faces = (
        np.ones(((volumes.shape[0] * fpe), 3), dtype=settings.INT_DTYPE) * -1
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

    .. code-block:: text


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
    volumes = np.asanyarray(volumes, settings.INT_DTYPE)

    if volumes.ndim != 2 or volumes.shape[1] != 8:
        raise ValueError("Given volumes are not `hexa` volumes")

    fpe = 6  # faces per element
    faces = np.empty(((volumes.shape[0] * fpe), 4), dtype=settings.INT_DTYPE)

    faces[::fpe] = volumes[:, [1, 0, 3, 2]]
    faces[1::fpe] = volumes[:, [0, 1, 5, 4]]
    faces[2::fpe] = volumes[:, [1, 2, 6, 5]]
    faces[3::fpe] = volumes[:, [2, 3, 7, 6]]
    faces[4::fpe] = volumes[:, [3, 0, 4, 7]]
    faces[5::fpe] = volumes[:, [4, 5, 6, 7]]

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
    volumes = np.asanyarray(volumes, settings.INT_DTYPE)
    if volumes.shape[1] == 4:
        return tet_to_tri(volumes)

    elif volumes.shape[1] == 8:
        return hexa_to_quad(volumes)


def faces_to_edges(faces):
    """Compute edges based on following edge scheme.

    .. code-block:: text

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
    edges = np.empty((num_edges, 2), dtype=settings.INT_DTYPE)

    edges[:, 0] = faces.ravel()

    for i in range(vertices_per_face):
        # v_ind : corresponding vertex index for i value
        v_ind = 0 if i == int(vertices_per_face - 1) else i + 1

        edges[i::vertices_per_face, 1] = faces[:, v_ind]

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
        indices = np.arange(range_, dtype=settings.INT_DTYPE)
    elif isinstance(range_, (list, tuple)):
        # pass range_ as is and check for valid output
        indices = np.arange(*range_, dtype=settings.INT_DTYPE)
        if len(indices) < 2:
            raise ValueError(
                f"{range_} is invalid range input. "
                "It must result in minimum of size=2 array."
            )

    # closed is ignored
    if not continuous:
        if indices.size % 2 == 1:
            raise ValueError(
                "Ranges should result in even number of indices for "
                "continuous edges."
            )
        return indices.reshape(-1, 2)

    return sequence_to_edges(indices, closed)


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
    edges = np.repeat(seq, 2)

    if closed:
        first = int(edges[0])  # this is redundant copy to ensure detaching
        # equivalent to np.roll(edges, -1) without copy
        # safe since numpy is single thread
        edges[:-1] = edges[1:]
        edges[-1] = first
    else:
        # reselect only part of the array instead of copy
        # this should only update array interface protocol
        edges = edges[1:-1]

    return edges.reshape(-1, 2)


def make_quad_faces(resolutions):
    """Given number of nodes per each dimension, returns connectivity
    information of a structured mesh. Counter clock wise connectivity.

    .. code-block:: text

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
        node_indices = np.arange(
            total_nodes, dtype=settings.INT_DTYPE
        ).reshape(nnpd[1], nnpd[0])
    except ValueError as e:
        raise ValueError(f"Problem with generating node indices. {e}")

    faces = np.empty((total_faces, 4), dtype=settings.INT_DTYPE)

    faces[:, 0] = node_indices[: (nnpd[1] - 1), : (nnpd[0] - 1)].ravel()
    faces[:, 1] = node_indices[: (nnpd[1] - 1), 1 : nnpd[0]].ravel()
    faces[:, 2] = node_indices[1 : nnpd[1], 1 : nnpd[0]].ravel()
    faces[:, 3] = node_indices[1 : nnpd[1], : (nnpd[0] - 1)].ravel()

    return faces


def make_hexa_volumes(resolutions):
    """Given number of nodes per each dimension, returns connectivity
    information of structured hexahedron elements. Counter clock wise
    connectivity.

    .. code-block:: text

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
    node_indices = np.arange(total_nodes, dtype=settings.INT_DTYPE).reshape(
        nnpd[::-1]
    )

    volumes = np.empty((total_volumes, 8), dtype=settings.INT_DTYPE)

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

    return volumes


def subdivide_edges(edges):
    """Subdivide edges. We assume that mid point is newly added points.

    ``Subdivided Edges``

    .. code-block:: text

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


def subdivide_tri(mesh, return_dict=False):
    """Subdivide triangles. Each triangle is divided into 4 meshes.

    ``Subdivided Faces``

    .. code-block:: text

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
        dtype=settings.INT_DTYPE,
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
        return {
            "vertices": new_vertices,
            "faces": subdivided_faces,
        }

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
        dtype=settings.INT_DTYPE,
    )

    subdivided_faces[:, 0] = mesh.faces.ravel()
    subdivided_faces[:, 1] = mesh.unique_edges().inverse + len(mesh.vertices)
    subdivided_faces[:, 2] = np.repeat(
        np.arange(len(face_centers)) + (len(mesh.vertices) + len(edge_mid_v)),
        4,
        dtype=settings.INT_DTYPE,
    )
    subdivided_faces[:, 3] = (
        subdivided_faces[:, 1].reshape(-1, 4)[:, [3, 0, 1, 2]].ravel()
    )

    if return_dict:
        return {
            "vertices": new_vertices,
            "faces": subdivided_faces,
        }

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


def _sequentialize_directed_edges(edges, start=None, return_edges=False):
    """
    Sequentialize directed edges.
    """
    # we want to have an np array
    edges = np.asanyarray(edges)

    # Build a lookup_array
    lookup_array = np.full(edges.max() + 1, -1, dtype=settings.INT_DTYPE)
    lookup_array[edges[:, 0]] = edges[:, 1]

    # select starting point - lowest index
    starting_point = int(edges.min()) if start is None else int(start)

    # initialize a set to keep track of processes vertices
    next_candidates = set(edges[:, 0])
    # we want to keep track of single occurrences, as they are line start
    line_starts = set(np.where(np.bincount(edges.ravel()) == 1)[0])
    # for this to work, we can't have a line that starts at column 1.
    # so, we remove those.
    ls_col1 = set(np.where(np.bincount(edges[:, 1].ravel()) == 1)[0])
    line_starts.difference_update(ls_col1)

    polygons = []
    is_polygon = []
    for _ in range(len(edges)):
        # Get polygon - start with first two points
        polygon = [starting_point]
        polygon.append(lookup_array[polygon[-1]])

        # then keep looking until we come to the start.
        # if there's only one edges, this will exit right away
        for _ in range(len(next_candidates)):
            # sequence closes -> polygon
            if polygon[0] == polygon[-1]:
                polygon.pop()
                is_polygon.append(True)
                break

            # sequence leads to unspecified connection -> not polygon
            if lookup_array[polygon[-1]] < 0:
                is_polygon.append(False)
                break
            polygon.append(lookup_array[polygon[-1]])

        polygons.append(polygon)

        # check if we counted all the edges
        # if so, exit
        next_candidates.difference_update(polygons[-1])
        line_starts.difference_update(polygons[-1])

        if len(next_candidates) == len(line_starts) == 0:
            break

        # let's try to find the next starting point
        starting_point = None
        if len(line_starts) != 0:
            starting_point = min(line_starts)
        else:
            starting_point = min(next_candidates)

    if not return_edges:
        return polygons, is_polygon

    else:
        polygon_edges = []
        for p, is_p in zip(polygons, is_polygon):
            polygon_edges.append(sequence_to_edges(p, closed=is_p))

        return polygon_edges, is_polygon


def _sequentialize_edges(edges, start=None, return_edges=False):
    """
    sequentialize undirected edges. No overlaps are allowed, for now.
    """
    edges = np.asanyarray(edges)

    # only applicable to closed polygons and open lines
    # not for arbitrarily connected edges
    bc = np.bincount(edges.ravel())
    if not all(bc < 3):
        raise ValueError(
            "This function currently supports individual lines/polygons "
            "search. Given edges include connections with more than 3 edges."
        )

    # we want to keep track of single occurrences, as they are line start
    line_starts = set(np.where(bc == 1)[0])

    # initialize a set to keep track of processes vertices
    next_candidates = set(edges.ravel())

    # create a look up to each edge column
    edge_col = collections.namedtuple("a", "b")
    edge_col.a = edges[:, 0]
    edge_col.b = edges[:, 1]

    # create trees for each edge column
    tree = collections.namedtuple("a", "b")
    tree.a = napf.KDT(edge_col.a.reshape(-1, 1))
    tree.b = napf.KDT(edge_col.b.reshape(-1, 1))

    # radius search size
    r = 0.1

    current_id = np.argmin(edge_col.a) if start is None else start
    start_value = int(edge_col.a[current_id])
    other_col = edge_col.b

    polygons = []
    is_polygon = []
    for _ in range(len(edges)):
        polygon = [start_value, other_col[current_id]]

        #        while polygon[0] != polygon[-1]:
        for _ in range(len(next_candidates)):
            if polygon[0] == polygon[-1]:
                break
            # search for ids
            a_ids = tree.a.radius_search([[polygon[-1]]], r, True)[0][0]
            b_ids = tree.b.radius_search([[polygon[-1]]], r, True)[0][0]

            # in total, there should be 2 otherwise, we can end this search
            # and this is not a polygon
            hits = len(a_ids) + len(b_ids)
            if hits != 2:
                break

            # so, we have 2 hits. we need to get the partner of non-current_id
            # index
            found = False
            for ai in a_ids:
                if ai != current_id:
                    found = True
                    current_id = ai
                    polygon.append(edge_col.b[current_id])
                    break
            if found:
                continue

            for bi in b_ids:
                if bi != current_id:
                    found = True
                    current_id = bi
                    polygon.append(edge_col.a[current_id])
                    break
            if found:
                continue

            raise RuntimeError(
                "Something went wrong. Please report this issue to "
                "github.com/tataratat/gustaf/issues]"
            )

        # if indices closes itself, it is a polygon.
        # Otherwise it is an open line
        if polygon[0] == polygon[-1]:
            polygon.pop()
            is_polygon.append(True)
        else:
            is_polygon.append(False)
        polygons.append(polygon)

        # check if we counted all the edges
        # if so, exit
        # let's try to find the next starting point
        next_candidates.difference_update(polygons[-1])
        line_starts.difference_update(polygons[-1])

        if len(next_candidates) == len(line_starts) == 0:
            break

        # Assign next starting point
        # generally first value of (those are min() in a set) line start or
        # leftover candidates
        start_value = None
        if len(line_starts) != 0:
            start_value = min(line_starts)
        else:
            start_value = min(next_candidates)

        # adjust state values
        a_ids = tree.a.radius_search([[start_value]], r, True)[0][0]
        if len(a_ids) != 0:
            current_id = a_ids[0]
            other_col = edge_col.b
            continue
        b_ids = tree.b.radius_search([[start_value]], r, True)[0][0]
        if len(b_ids) != 0:
            current_id = b_ids[0]
            other_col = edge_col.a
            continue

        raise RuntimeError(
            "Something went wrong. Please report this issue to "
            "github.com/tataratat/gustaf/issues"
        )

    if not return_edges:
        return polygons, is_polygon

    else:
        polygon_edges = []
        for p, is_p in zip(polygons, is_polygon):
            polygon_edges.append(sequence_to_edges(p, closed=is_p))

        return polygon_edges


def sequentialize_edges(edges, start=None, return_edges=False, directed=False):
    """
    Organize edge connectivities to describe polygon or a line.
    This supports edges that describes separated/individual polygons and lines.
    In other words, it doesn't support edges of overlapping vertices.

    Parameters
    -----------
    edges: (n, 2) list-like
    start: int
      (Optional) Specify starting point. It will take minimum index otherwise.
    return_edges: bool
      (Optional) Default is False. If set True, returns sequences as edges.
    directed: bool
      (Optional) Default is False. Set True, if given edges are directed.
      It should return the result faster.

    Returns
    --------
    sequences: list
      list of vertex ids. Or edges iff return_edges is True.
    is_polygon: list
      Tells if the sequence is a polygon or a line.

    Examples
    ---------
    >>> e = gus.Edges(vertices, edges)
    >>> ordered_sequence, is_polygon = sequentialize_edges(e.edges)

    >>> f = gus.Faces(vertices, faces)
    >>> ordered_sequence, is_polygon = sequentialize_edges(
    ...     f.edges()[f.single_edges()]
    ... )
    """
    if directed:
        return _sequentialize_directed_edges(edges, start, return_edges)
    else:
        return _sequentialize_edges(edges, start, return_edges)
