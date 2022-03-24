"""gustaf/spline/extract.py

Extract operations. Both discrete and spline extraction.
"""


import itertools

import numpy as np

from gustaf import utils
from gustaf import settings
from gustaf.vertices import Vertices
from gustaf.edges import Edges
from gustaf.faces import Faces
from gustaf.volumes import Volumes

from gustaf.spline._utils import to_res_list


def edges(
        spline,
        resolution=100,
        extract_dim=None,
        extract_knot=None,
        all_knots=False,
):
    """
    Extract edges (lines) from a given spline.
    Only entity you can extract without dimension limit

    Parameters
    -----------
    spline: Spline
    resolution: int 
    extract_dim: int
      Parametric dimension to extract.
    extract_knot: list
      (spline.para_dim - 1,) shaped knot location along extract_dim
    all_knots: bool
      Switch to allow all knot-line extraction.

    Returns
    --------
    edges: Edges
    """
    resolution = int(resolution)

    if spline.para_dim == 1:
        return Edges(
            vertices=spline.sample(resolution),
            edges=utils.connec.range_to_edges(
                (0, resolution),
                closed=False,
            )
        )

    else:
        # This should be possible for spline of any dimension.
        # As long as it satisfies the following condition
        if extract_knot is not None:
            if len(extract_knot) != spline.para_dim - 1:
                raise ValueError(
                    "Must satisfy len(extract_knot) == spline.para_dim -1."
                )

        # This may take awhile.
        if all_knots:
            edgess = [] # edges' is not a valid syntax
            unique_knots = np.array(spline.unique_knots, dtype=object)
            for i in range(spline.para_dim):
                mask = np.ones(spline.para_dim, dtype=bool)
                mask[i] = False
                # gather knots along current knot
                extract_knot_queries = list(
                    itertools.product(*unique_knots[mask])
                )

                for ekq in extract_knot_queries:
                    edgess.append(
                        edges(spline, resolution, i, ekq, False)
                    )

            return Edges.concat(edgess)

        # Get parametric points to extract
        queries = np.empty(
            (resolution, spline.para_dim),
            dtype="float64", # hardcoded for splinelibpy
            order="C", # hardcoded for splinelibpy
        )
        # get ~extract_dim 
        not_ed = np.arange(spline.para_dim).tolist()
        not_ed.pop(extract_dim)
        queries[:, not_ed] = extract_knot
        queries[:, extract_dim] = np.linspace(
            min(spline.knot_vectors[extract_dim]),
            max(spline.knot_vectors[extract_dim]),
            resolution,
        )

        return Edges(
            vertices=spline.evaluate(queries),
            edges=utils.connec.range_to_edges(
                (0, resolution),
                closed=False,
            )
        )


def faces(spline, resolutions,):
    """
    Extract faces from spline.
    Valid iff para_dim is one of the followings: {2, 3}.
    In case of {3}, it will return only surfaces.
    If internal faces are desired, used `spline.extract.volumes().get_faces()`.
    Note that dimension higher than 3 is not showable.

    Parameters
    -----------
    spline: BSpline or NURBS
    resolutions: int or list

    Returns
    --------
    faces: faces
    """
    resolutions = to_res_list(resolutions, spline.para_dim)

    if spline.para_dim == 2:
        return Faces(
            vertices=spline.sample(resolutions),
            faces=utils.connec.make_quad_faces(resolutions),
        )

    elif spline.para_dim == 3:
        # TODO: use spline extraction routine to first extract
        # spline, extract faces, merge vertices.

        # Spline to surfaces
        vertices = []
        faces = []
        offset = 0
        kvs = spline.knot_vectors
        for i in range(spline.para_dim):
            extract = i
            # Get extracting dimension
            extract_along = [0, 1, 2] 
            extract_along.pop(extract)

            # Extract range
            extract_range = [
                [
                    min(kvs[extract_along[0]]),
                    max(kvs[extract_along[0]]),
                ],
                [
                    min(kvs[extract_along[1]]),
                    max(kvs[extract_along[1]]),
                ],
            ]

            extract_list = [
                min(kvs[extract]),
                max(kvs[extract]),
            ]

            # surface point queries (spq)
            spq = np.linspace(
                extract_range[0][0],
                extract_range[0][1],
                resolutions[extract_along[0]],
            ).reshape(-1, 1)

            # expand horizontally and init with 1
            spq = np.hstack((spq, np.ones((len(spq), 1))))
            spq = np.vstack(
                np.linspace(
                    spq * [1, extract_range[1][0]],
                    spq * [1, extract_range[1][1]],
                    resolutions[extract_along[1]],
                )
            )

            # expand horizontally and init with 1
            spq = np.hstack((spq, np.ones((len(spq), 1))))
            spq = np.vstack(
                np.linspace(
                    spq * [1, 1, extract_list[0]],
                    spq * [1, 1, extract_list[1]],
                    2
                )
            )

            surface_point_queries = utils.arr.make_c_contiguous(
                spq,
                dtype="float64", # hardcoded since splinelibpy uses this dtype
            )
            surface_point_queries = surface_point_queries[
                :,
                np.argsort(
                    [extract_along[0], extract_along[1], extract]
                )
            ]

            vertices.append(
                spline.evaluate(
                    surface_point_queries[
                        :int(surface_point_queries.shape[0] / 2)
                    ]
                )
            )

            if len(faces) != 0:
                offset = faces[-1].max() + 1

            tmp_faces = utils.connec.make_quad_faces(
                [
                    resolutions[extract_along[0]],
                    resolutions[extract_along[1]],
                ]
            )

            faces.append(tmp_faces + int(offset))

            vertices.append(
                spline.evaluate(
                    surface_point_queries[
                        int(surface_point_queries.shape[0] / 2):
                    ]
                )
            )

            offset = faces[-1].max() + 1

            faces.append(tmp_faces + int(offset))

        # make faces and merge vertices before returning
        f = Faces(
            vertices=np.vstack(vertices),
            faces=np.vstack(faces)
        )
        f.merge_vertices(inplace=True)

        return f

    else:
        raise ValueError(
            "Invalid spline to make faces."
        )


def volumes(spline, resolutions):
    """
    Extract volumes from spline.
    Valid iff spline.para_dim == 3.

    Parameters
    -----------
    spline: BSpline or NURBS
    resolutions: 

    Returns
    --------
    volumes: Volumes
    """
    if spline.para_dim != 3:
        raise ValueError(
            "Volume extraction from a spline is only valid for para_dim: 3 "
            + "dim: 3 splines."
        )

    return Volumes(
        vertices=spline.sample(resolutions),
        volumes=utils.connec.make_hexa_volumes(resolutions),
    )


def control_points(spline):
    """
    Extracts control points and return as vertices.
    Same can be achieved by doing `gustaf.Vertices(spline.control_points)`

    Parameters
    -----------
    spline: BSpline or NURBS

    Returns
    --------
    cps_as_Vertices: Vertices
    """
    return Vertices(spline.control_points)


def control_edges(spline):
    """
    Extract control edges (mesh).
    Valid iff para_dim is 1.

    Parameters
    -----------
    edges: BSpline or NURBS

    Returns
    --------
    edges: Edges
    """
    if spline.para_dim != 1:
        raise ValueError("Invalid spline type!")

    return Edges(
        vertices=spline.control_points,
        edges=utils.connec.range_to_edges(
            len(spline.control_points),
            closed=False
        )
    )


def control_faces(spline):
    """
    Extract control face (mesh).
    Valid iff para_dim is 2.

    Parameters
    -----------
    spline: BSpline or NURBS

    Returns
    --------
    faces: Faces
    """
    if spline.para_dim != 2:
        raise ValueError("Invalid spline type!")

    return Faces(
        vertices=spline.control_points,
        faces=utils.connec.make_quad_faces(spline.control_mesh_resolutions),
    )


def control_volumes(spline):
    """
    Extract control volumes (mesh).
    Valid iff para_dim is 3.

    Parameters
    -----------
    spline: BSpline or NURBS

    Returns
    --------
    volumes: Volumes
    """
    if spline.para_dim != 3:
        raise ValueError("Invalid spline type!")

    return Volumes(
        vertices=spline.control_points,
        volumes=utils.connec.make_hexa_volumes(spline.control_mesh_resolutions),
    )


def control_mesh(spline,):
    """
    Calls control_edges, control_faces, control_volumes
    based on current spline.

    Parameters
    -----------
    None

    Returns
    --------
    control_mesh: Edges or Faces or Volumes
    """
    if spline.para_dim == 1:
        return control_edges(spline)
    elif spline.para_dim == 2:
        return control_faces(spline)
    elif spline.para_dim == 3:
        return control_volumes(spline)
    else:
        raise ValueError(
            "Invalid para_dim to extract control_mesh. "
            "Supports 1 to 3."
        )


class _Extractor:
    """
    Helper class to allow direct extraction from spline obj (BSpline or NURBS).
    Internal use only.

    Examples
    ---------
    >>> myspline = <your-spline>
    >>> spline_faces = myspline.extract.faces()
    """

    def __init__(self, spl):
        self.spline = spl

    def edges(self, *args, **kwargs):
        return edges(self.spline, *args, **kwargs)

    def faces(self, *args, **kwargs):
        return faces(self.spline, *args, **kwargs)

    def volumes(self, *args, **kwargs):
        return volumes(self.spline, *args, **kwargs)

    def control_points(self):
        return control_points(self.spline)
    
    def control_edges(self):
        return control_edges(self.spline)
    
    def control_faces(self):
        return control_faces(self.spline)
    
    def control_volumes(self):
        return control_volumes(self.spline)
    
    def control_mesh(self):
        return control_mesh(self.spline)

