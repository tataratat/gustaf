"""gustaf/create/faces.py
Routines to create faces.
"""

import numpy as np

from gustaf import create, settings, utils
from gustaf.faces import Faces


def box(bounds=None, resolutions=None, simplex=False, backslash=False):
    """Create structured quadrangle or triangle block mesh.

    Parameters
    -----------
    bounds: (2, 2) array
        Minimum and maximum coordinates.
    resolutions: (2) array
        Vertex count in each dimension.
    simplex: boolean
        If true, Mesh will be triangular (simplex).

    Returns
    --------
    face_mesh: Volumes
    """

    if resolutions is None:
        resolutions = [2, 2]
    if bounds is None:
        bounds = [[0, 0], [1, 1]]
    if np.array(bounds).shape != (2, 2):
        raise ValueError("Bounds must have a dimension of (2, 2).")
    if len(resolutions) != 2:
        raise ValueError("Resolutions must have two entries.")
    if not np.greater(resolutions, 1).all():
        raise ValueError("All resolution values must be at least 2.")

    if simplex:
        # create quad mesh as basis
        quad_mesh = box(bounds=bounds, resolutions=resolutions)

        # turn into triangles
        face_mesh = to_simplex(quad_mesh, backslash)

    else:
        vertex_mesh = create.vertices.raster(bounds, resolutions)
        connectivity = utils.connec.make_quad_faces(resolutions)
        face_mesh = Faces(vertex_mesh.vertices, connectivity)

    return face_mesh


def to_simplex(quad, alternate=False):
    """Given quad faces, diagonalize them to turn them into triangles.

    If quad is counterclockwise (CCW), triangle will also be CCW and
    vice versa. Will return a tri-mesh, if input is triangular.
    Default diagonalization looks like this:

    .. code-block:: text

        (3) *---* (2)
            |  /|
            | / |
            |/  |
        (0) *---* (1)

    resembling 'slash'.

    .. code-block:: text

        (3) *---* (2)
            |\\  |
            | \\ |
            |  \\|
        (0) *---* (1)

    resembling 'backslash'.

    If you want to alternate the `slash`-direction, set `alternate`-variable.

    Parameters
    ----------
    quad: Faces
      Faces representation which is to be converted from a cubic mesh into a
      simplex mesh.
    alternate: bool
       Alternate between forward and back-slash to avoid "favored" meshing
       direction (important in some analysis problem).

    Returns
    --------
    tri: Faces
        Simplexifyed mesh.
    """

    if not isinstance(quad, Faces):
        raise ValueError(
            "Input to to_simplex needs to be of type Faces, but it's "
            + type(quad)
        )

    if quad.whatami.startswith("quad"):
        # split variants
        split_slash = [[0, 1, 2], [2, 3, 0]]
        split_backslash = [[0, 1, 3], [3, 1, 2]]

        quad_faces = quad.faces
        tf_half = int(quad_faces.shape[0])
        tri_faces = np.full((tf_half * 2, 3), -1, dtype=settings.INT_DTYPE)

        split = split_backslash if alternate else split_slash

        # If backslash assign every other with backslash else only forward
        tri_faces[0:tf_half:2] = quad_faces[0::2, split_slash[0]]
        tri_faces[1:tf_half:2] = quad_faces[1::2, split[0]]
        tri_faces[tf_half::2] = quad_faces[0::2, split_slash[1]]
        tri_faces[(tf_half + 1) :: 2] = quad_faces[1::2, split[1]]

        tri = Faces(
            vertices=quad.vertices.copy(),
            faces=tri_faces,
        )
    else:
        tri = quad
        utils.log.debug(
            "Non quadrilateral mesh provided, return original" " mesh."
        )

    return tri


def to_quad(tri):
    """
        In case current mesh is triangle surface mesh, it splits triangle faces
        into three quad faces by inserting another vertices in the middle of
        each triangle face. Warning: mesh quality could be bad!

        ``(new) Quad Face``

        .. code-block:: text

            Ref: (node_ind), face_ind

                     (2)
                     / \
                    / 2 \
                (5)/\\   /\\(4)
                  /  \\ /  \
                 / 0  |(6) \
                /_____|___1_\
              (0)   (3)    (1)

            face_ind | node_ind
            ---------|----------
            0        | 0 3 6 5
            1        | 1 4 6 3
            2        | 2 5 6 4

        Parameters
        -----------
        None

        Returns
        --------
        quad_mesh: Mesh
          Only if current mesh is triangle surface mesh. Otherwise, None.
        """
    if tri.elements is None:
        return None

    if not tri.whatami.startswith("tri"):
        return None

    # Form new vertices: 1. mid edge vertices; 2. center vertices.
    edge_mids = tri.to_edges(unique=True).centers()

    # New vertices - current vertices & mid edge vertices & center vertices
    vertices = np.vstack(
        (
            tri.const_vertices,
            edge_mids,
            tri.centers(),
        )
    )
    em_offset = len(tri.vertices)  # edge mid offset
    fc_offset = len(tri.vertices) + len(edge_mids)  # faces center offset

    # New faces - current faces * 3, as each triangle face should produce
    #  3 quad faces
    faces = np.empty((tri.faces.shape[0] * 3, 4), dtype=settings.INT_DTYPE)

    # Assign face ind
    # First col.
    faces[:, 0] = tri.faces.ravel()
    # Second col.
    edge_mid_column = tri.unique_edges().inverse + em_offset
    faces[:, 1] = edge_mid_column
    # Third col.
    faces[:, 2] = np.repeat(np.arange(len(tri.faces)) + fc_offset, 3)
    # Fourth col.
    faces[:, 3] = edge_mid_column.reshape(-1, 3)[:, [2, 0, 1]].ravel()

    return Faces(vertices=vertices, faces=faces)


def edges_as_quad(edges, scaled_normals):
    """
    Expands edges to hexa with given scaled_normals.

    Parameters
    ----------
    edges: Edges
      (n, d) vertices, (m, 2) edges
    scaled_normals: (n, d) np.ndarray
      Values will be used to subtract/add to existing vertices.

    Returns
    -------
    expanded: Faces
    """
    if edges.whatami != "edges":
        raise TypeError("expand currently only supports Edges.")

    size = scaled_normals.shape[0]
    dim = scaled_normals.shape[1]
    if len(edges.vertices) != size:
        raise ValueError("Edges.vertices and scaled_normals size mismatch.")

    if dim != edges.vertices.shape[1]:
        raise ValueError(
            "Dimension mismatch between Edges.Vertices and scaled_normals"
        )

    vertices = np.empty((size * 2, dim), dtype=settings.FLOAT_DTYPE)
    vertices[:size] = edges.vertices - scaled_normals
    vertices[size:] = edges.vertices + scaled_normals

    quad = np.empty((len(edges.edges), 4), dtype=settings.INT_DTYPE)
    quad[:, :2] = edges.edges
    quad[:, 2] = edges.edges[:, 1] + size
    quad[:, 3] = edges.edges[:, 0] + size

    return Faces(vertices, quad)


def vertex_normals(
    faces,
    area_weighting=False,
    angle_weighting=False,
    return_original_ids=False,
):
    """
    Computes vertex normals and saves it in vertex_data.
    This calls inplace remove_unreferenced_vertices, but original IDs can be
    retrieved using the flag `return_original_ids`.

    The normals are computed on the face-centers and their contributions are
    weighted and added to the vertex normals. Per default, all element faces
    that are adjacent to a vertex are added with equal contributions, but it is
    also possible to use weightings by area of the adjacent element
    (`area_weighting`) or by the angle between edges at the corner vertex.

    Parameters
    ----------
    faces: Faces
    area_weighting : bool (false)
      Use the element area as a weighting to its respective normal contribution
    angle_weighting : bool (false)
      Use the angle of between element edges as a weighting to its respective
      normal contribution
    return_original_ids : bool (false)
      return the original ids in the global mesh

    Returns
    -------
    faces: Faces
      faces with vertex_data["normals"] computed.
    """
    if not faces.whatami.startswith("tri"):
        raise ValueError("Vertex normals only supports triangle faces")

    if faces.vertices.shape[1] != 3:
        raise ValueError("Vertex normals only support 3d triangles")

    if return_original_ids:
        original_ids = np.where(faces.referenced_vertices())[0]

    faces.remove_unreferenced_vertices()

    triangles = faces.vertices[faces.faces]

    # compute (1 - 0) and (2 - 0)
    edge_ab = triangles[:, 1] - triangles[:, 0]
    edge_bc = triangles[:, 2] - triangles[:, 1]

    # get normal of each faces and normalize
    crossed = utils.arr.cross3d(edge_ab, edge_bc)
    crossed_length = np.linalg.norm(crossed, axis=1).reshape(-1, 1)

    weights = np.ones_like(crossed, dtype=settings.FLOAT_DTYPE)

    # get area based weights
    if area_weighting:
        weights *= crossed_length
    #     crossed /= crossed_length
    # else:
    crossed /= crossed_length

    # get triangle corner angles (same as faces.faces)
    if angle_weighting:
        angles = np.empty_like(crossed, dtype=settings.FLOAT_DTYPE)
        norm_ab = np.linalg.norm(edge_ab, axis=1)
        norm_bc = np.linalg.norm(edge_bc, axis=1)
        edge_ca = edge_ab + edge_bc
        norm_ca = np.linalg.norm(edge_ca, axis=1)
        np.arcsin(
            crossed_length.ravel() / (norm_ab * norm_bc).ravel(),
            out=angles[:, 0],
        )
        np.arccos(
            np.einsum("ij,ij->i", edge_bc, edge_ca) / (norm_bc * norm_ca),
            out=angles[:, 1],
        )
        angles[:, 2] = np.pi - angles[:, 0] - angles[:, 1]

        weights *= angles

    # initialize
    normals = np.zeros_like(faces.vertices, dtype=settings.FLOAT_DTYPE)

    # sum normals and weights
    np.add.at(
        normals, faces.faces[:, 0], crossed * weights[:, 0].reshape(-1, 1)
    )
    np.add.at(
        normals, faces.faces[:, 1], crossed * weights[:, 1].reshape(-1, 1)
    )
    np.add.at(
        normals, faces.faces[:, 2], crossed * weights[:, 2].reshape(-1, 1)
    )

    # normalize
    normals /= np.linalg.norm(normals, axis=1).reshape(-1, 1)

    faces.vertex_data["normals"] = normals
    if return_original_ids:
        return (faces, original_ids)
    else:
        return faces
