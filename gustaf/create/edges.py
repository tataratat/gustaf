"""gustaf/create/edges.py
Routines to create edges.
"""

import numpy as np

from gustaf import utils
from gustaf.edges import Edges


def from_vertices(vertices, closed=False, continuous=True):
    """
    Creates Edges with given vertices. If close==True,
    last vertices will be connected to the first one.
    If continuous==False, it assumes that every two vertices form
    an independent line.

    Parameters
    ----------
    vertices: (n, d) np.ndarray or Vertices
    close: bool
    continuous: bool

    Returns
    -------
    edges: Edges
    """
    if hasattr(vertices, "vertices"):  # noqa SIM108
        v = vertices.vertices
    else:
        v = vertices

    edges = Edges(v, utils.connec.range_to_edges(len(v), closed, continuous))

    return edges


def from_data(gus_obj, data, scale=None, data_norm=None):
    """
    Creates edges from gustaf object with vertices.
    Data can be either multi-dim array-like data or a str describing a name
    of vertex_data that belongs to given gustaf object.
    len(gus_obj.vertices) number of edges will be created, where origin and
    end of each edge is created using the following scheme:
    [[vertices[0], vertices[0] + (array_data[0] * scale)], ...].
    By default, scaling value will be
    max([1, (aabb_diagonal_norm * 0.1 / max_data_norm)]).
    If there's dimension mismatch between vertices and the data, will append
    zero paddings!

    Parameters
    ----------
    gus_obj: Vertices
      gus.Vertices or its derived classes
    data: str or (n_vertices, d) array-like
     If str, will be considered as data and search for saved vertex_data.
    scale: float
      Absolute value.
    data_norm: float or array-like
      If float, will be considered as max_norm of the data. Else, searches for
      max value. Doesn't enforce len to match.

    Returns
    -------
    data_arrow: Edges
    """
    if not isinstance(gus_obj, Edges.__boundary_class__):
        raise TypeError(
            "Invalid input. Expecting gus.Vertices or its subclasses"
        )

    # get origin and increment (= array_data * scale)
    origin = gus_obj.const_vertices
    if isinstance(data, str):
        # will raise if data doesn't exist
        increment = gus_obj.vertex_data[data]
    elif isinstance(data, (tuple, list, np.ndarray)):
        increment = np.asanyarray(data)
        if len(increment) != len(origin):
            raise ValueError(
                f"Data length mismatch: expected ({len(origin)}) / "
                "given ({len(increment)})"
            )
    else:
        raise TypeError(f"Couldn't process {type(data)}-data as input.")

    # data dim check
    inc_dim = increment.shape[1]
    origin_dim = origin.shape[1]
    if inc_dim == 1:
        # if data is scalar, it should be addable to any vertices
        utils.log.debug(
            f"gus.create.edges.from_data() - requested data ({data}) is",
            "scalar. Edge orientation will be [1,1,..,1].",
        )
    elif inc_dim != origin_dim:
        # match data and vertex dim.
        utils.log.debug(
            "gus.create.edges.from_data() - dimension mismatch between",
            f"vertices ({origin_dim}) and data ({inc_dim}). Matching",
            "dimension by appending zeros.",
        )
        dim_diff = int(inc_dim - origin_dim)
        zero_pad = np.zeros((len(origin), abs(dim_diff)))
        if dim_diff > 0:
            origin = np.hstack((origin, zero_pad))
        else:
            increment = np.hstack((increment, zero_pad))

    # apply default scale
    if scale is None:
        if isinstance(data, str):
            norm = gus_obj.vertex_data.as_scalar(data, None, True)
        elif data_norm is None:
            norm = np.linalg.norm(increment, axis=1)
        else:
            norm = np.asanyarray(data_norm)

        max_data_norm = norm.max()
        aabb_diagonal_norm = gus_obj.bounds_diagonal_norm()
        scale = min((1.0, aabb_diagonal_norm * (0.1) / max_data_norm))
        utils.log.debug(
            f"creating edges from data with scaling factor ({scale})"
        )

    # by here, this should be good to go
    vs = np.hstack((origin, origin + (increment * scale))).reshape(
        -1, origin.shape[1]
    )

    return Edges(vs, utils.connec.range_to_edges(len(vs), continuous=False))
