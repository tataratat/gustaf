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
