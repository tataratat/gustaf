"""gustaf/gustaf/utils/mesh.py

All the operations that takes mesh as input and returns data to construct a 
desired new mesh. There are always `return_kwargs` option, which allows you to
generate new mesh by `gus.Mesh(**routine_name(*args, return_kwargs=True,)`
"""

import numpy as np

def subdivide_edges(edges):
    """
    Subdivide edges. We assume that mid point is newly added points.

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
    subd_edges: (n * 2, 2) np.ndarray
    """
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Invalid edges shape!")

    subd_edges = np.ones(
        (edges.shape[0] * 2, 2)
    )

    raise NotImplementedError


def subdivide_tri(mesh, return_kwargs=False,):
    """
    Subdivide triangles. Each triangle is divided into 4 meshes.

    ``Subdivided Faces``

    .. code-block::

        Triangles

        Ref: (node_ind), face_ind

                 (0)
                 /\ 
                / 0\ 
            (3)/____\(5)
              /\    /\ 
             / 1\ 3/ 2\ 
            /____\/____\ 
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
    return_kwargs: bool

    Returns
    --------
    new_vertices: (n, d) np.ndarray
    subd_faces: (m, 3) np.ndarray
    mesh_kwargs: dict
      iff `return_kwargs=True`,
      returns dict(vertices=new_vertices, faces=subd_faces).
    """
    # This will only pass if the mesh is triangle mesh.
    if mesh.faces.shape[1] != 3:
        raise ValueError("Invalid faces shape!")

    # Form new vertices
    edge_mid_v = mesh.vertices[mesh.edges_unique].mean(axis=1)
    new_vertices = np.vstack((mesh.vertices, edge_mid_v)

    subd_faces = np.empty(
        (mesh.faces.shape[0] * 4, mesh.faces.shape[1]),
        dtype=np.int32,
    )

    mask = np.ones(mesh.faces.shape[0], dtype=bool)
    mask[3::4] = False

    # 0th column minus (every 4th row, starting from 3rd row)
    subd_faces[mask, 0] = mesh.faces.flatten() # copies

    # Form ids for new vertices
    new_vertices_ids = mesh.edges_unique_inverse + int(mesh.faces.max() + 1)
    # 1st & 2nd columns
    subd_faces[mask, 1] = new_vertices_ids
    subd_faces[mask, 2] = new_vertices_ids.reshape(-1, 3)[:, [2,0,1]].flatten()

    # Every 4th row, starting from 3rd row
    subd_faces[~mask, :] = new_vertices_ids.reshape(-1, 3)

    if return_kwargs:
        return dict(
            vertices=new_vertices,
            faces=subd_faces,
        )

    else:
        return new_vertices, subd_faces
    

def subdivide_quad(mesh, return_kwargs=False,):
    """
    Subdivide quads.

    Parameters
    -----------
    mesh: Mesh
    return_kwargs: bool

    Returns
    --------
    new_vertices: (n, d) np.ndarray
    subd_faces: (m, 4) np.ndarray
    mesh_kwargs: dict
      iff `return_kwargs=True`,
      returns dict(vertices=new_vertices, faces=subd_faces).
    """
    # This will only pass if the mesh is quad mesh.
    if mesh.faces.shape[1] != 4:
        raise ValueError("Invalid faces shape!")

    # Form new vertices
    edge_mid_v = mesh.vertices[mesh.edges_unique].mean(axis=1)
    face_centers = mesh.element_centers
    new_vertices = np.vstack(
        (
            mesh.vertices,
            edge_mid_v,
            face_centers,
        )
    )

    subd_faces = np.empty(
        (mesh.faces.shape[0] * 4, mesh.faces.shape[1]),
        dtype=np.int32,
    )

    subd_faces[:, 0] = mesh.faces.flatten()
    subd_faces[:, 1] = mesh.edges_unique_inverse + len(self.vertices)
    subd_faces[:, 2] = np.repeat(
        np.arange(len(face_centers)) + (len(mesh.vertices) + len(edge_mid_v)),
        4,
        dtype=np.int32,
    )
    subd_faces[:, 3] = subd_faces[:, 1].reshape(-1, 4)[:, [3,0,1,2]].flatten()

    if return_kwargs:
        return dict(
            vertices=new_vertices,
            faces=subd_faces,
        )

    else:
        return new_vertices, subd_faces
