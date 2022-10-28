"""gustaf/spline/extract.py.

Extract operations. Both discrete and spline extraction.
"""

import itertools

import numpy as np

from gustaf import utils
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
    """Extract edges (lines) from a given spline. Only entity you can extract
    without dimension limit.

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
    if not all_knots:
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
            edgess = []  # edges' is not a valid syntax
            unique_knots = np.array(spline.unique_knots, dtype=object)
            for i in range(spline.para_dim):
                mask = np.ones(spline.para_dim, dtype=bool)
                mask[i] = False
                # gather knots along current knot
                extract_knot_queries = list(
                        itertools.product(*unique_knots[mask])
                )

                for ekq in extract_knot_queries:
                    edgess.append(edges(spline, resolution[i], i, ekq, False))

            return Edges.concat(edgess)

        # Get parametric points to extract
        queries = np.empty(
                (resolution, spline.para_dim),
                dtype="float64",  # hardcoded for splinelibpy
                order="C",  # hardcoded for splinelibpy
        )
        # get ~extract_dim
        not_ed = np.arange(spline.para_dim).tolist()
        not_ed.pop(extract_dim)
        queries[:, not_ed] = extract_knot

        # get knot extrema
        uniq_knots = spline.unique_knots[extract_dim]
        min_knot_position = min(uniq_knots)
        max_knot_position = max(uniq_knots)

        queries[:, extract_dim] = np.linspace(
                min_knot_position,
                max_knot_position,
                resolution,
        )

        return Edges(
                vertices=spline.evaluate(queries),
                edges=utils.connec.range_to_edges(
                        (0, resolution),
                        closed=False,
                )
        )


def faces(
        spline,
        resolutions,
):
    """Extract faces from spline. Valid iff para_dim is one of the followings:
    {2, 3}. In case of {3}, it will return only surfaces. If internal faces are
    desired, used `spline.extract.volumes().faces()`. Note that dimension
    higher than 3 is not showable.

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
        # accomodate bezier Splines
        ukvs = spline.unique_knots

        for i in range(spline.para_dim):
            extract = i
            # Get extracting dimension
            extract_along = [0, 1, 2]
            extract_along.pop(extract)

            # Extract range
            extract_range = [
                    [
                            min(ukvs[extract_along[0]]),
                            max(ukvs[extract_along[0]]),
                    ],
                    [
                            min(ukvs[extract_along[1]]),
                            max(ukvs[extract_along[1]]),
                    ],
            ]

            extract_list = [
                    min(ukvs[extract]),
                    max(ukvs[extract]),
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
                            spq * [1, 1, extract_list[1]], 2
                    )
            )

            surface_point_queries = utils.arr.make_c_contiguous(
                    spq,
                    dtype="float64",
            )
            sorted_ids = np.argsort(
                    [extract_along[0], extract_along[1], extract]
            )
            surface_point_queries = surface_point_queries[:, sorted_ids]

            vertices.append(
                    spline.evaluate(
                            surface_point_queries[:int(
                                    surface_point_queries.shape[0] / 2
                            )]
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
                                    int(surface_point_queries.shape[0] / 2):]
                    )
            )

            offset = faces[-1].max() + 1

            faces.append(tmp_faces + int(offset))

        # make faces and merge vertices before returning
        f = Faces(vertices=np.vstack(vertices), faces=np.vstack(faces))
        f.merge_vertices()

        return f

    else:
        raise ValueError("Invalid spline to make faces.")


def volumes(spline, resolutions):
    """Extract volumes from spline. Valid iff spline.para_dim == 3.

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
                "Volume extraction from a spline is only valid for "
                "para_dim: 3 dim: 3 splines."
        )

    return Volumes(
            vertices=spline.sample(resolutions),
            volumes=utils.connec.make_hexa_volumes(resolutions),
    )


def control_points(spline):
    """Extracts control points and return as vertices. Same can be achieved by
    doing `gustaf.Vertices(spline.control_points)`

    Parameters
    -----------
    spline: BSpline or NURBS

    Returns
    --------
    cps_as_Vertices: Vertices
    """
    return Vertices(spline.control_points)


def control_edges(spline):
    """Extract control edges (mesh). Valid iff para_dim is 1.

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
                    len(spline.control_points), closed=False
            )
    )


def control_faces(spline):
    """Extract control face (mesh). Valid iff para_dim is 2.

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
            faces=utils.connec.make_quad_faces(
                    spline.control_mesh_resolutions
            ),
    )


def control_volumes(spline):
    """Extract control volumes (mesh). Valid iff para_dim is 3.

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
            volumes=utils.connec.make_hexa_volumes(
                    spline.control_mesh_resolutions
            ),
    )


def control_mesh(spline, ):
    """Calls control_edges, control_faces, control_volumes based on current
    spline.

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


def beziers(spline):
    """Extracts Bezier-type objects of any spline-type object.

  Parameters
  ----------
  spline : Gustaf-Spline

  Returns
  -------
  bezier_list : list<bezier-types>
  """
    from gustaf.spline.base import Bezier, RationalBezier

    if "Bezier" in spline.whatami:
        return [spline]
    elif "BSpline" in spline.whatami:
        return [
                Bezier(**s.todict())
                for s in super(type(spline), spline).extract_bezier_patches()
        ]
    elif "NURBS" in spline.whatami:
        return [
                RationalBezier(**s.todict())
                for s in super(type(spline), spline).extract_bezier_patches()
        ]
    else:
        raise TypeError("Unknown Spline-Type.")


def spline(spline, para_dim, split_plane):
    """Extract a subspline from a given representation.

    Parameters
    ----------
    para_dim : int
      parametric dimension to be extrac ted
    split_plane : float / tuple<float, float>
      intervall or value in parametric space to be extracted from the spline
      representation

    Returns
    -------
    spline
    """
    from gustaf.spline.base import GustafSpline

    # Check type
    if not issubclass(type(spline), GustafSpline):
        raise TypeError("Unknown spline representation passed to subspline")

    # Check arguments for sanity
    if para_dim > spline.para_dim:
        raise ValueError(
                "Requested parametric dimension exceeds spline's parametric"
                " dimensionality."
        )
    if isinstance(split_plane, list):
        if not (
                (len(split_plane) == 2)
                and (isinstance(split_plane[0], float))
                and (isinstance(split_plane[0], float))
        ):
            raise ValueError(
                    "Range must be float or tuple of floats with length 2"
            )
    elif not isinstance(split_plane, float):
        raise ValueError(
                "Range must be float or tuple of floats with length 2"
        )
    else:
        # Convert float to tuple to facilitate
        split_plane = list([split_plane])

    # Check if is bezier-type
    is_bezier = "Bezier" in spline.whatami
    is_rational = "weights" in spline.required_properties
    if is_bezier:
        if is_rational:
            spline_copy = spline.nurbs
        else:
            spline_copy = spline.bspline
    else:
        spline_copy = spline.copy()

    for _ in range(spline_copy.degrees[para_dim]):
        # Will do nothing if spline already has sufficient number of knots
        # at given position
        spline_copy.insert_knots(para_dim, split_plane)

    # Start extraction
    cps_res = spline_copy.control_mesh_resolutions
    start_id = spline_copy.knot_vectors[para_dim].index(split_plane[0])
    end_id = spline_copy.knot_vectors[para_dim].index(split_plane[-1])
    para_dim_ids = np.arange(np.prod(cps_res))
    for i_pd in range(para_dim):
        para_dim_ids -= para_dim_ids % cps_res[i_pd]
        para_dim_ids = para_dim_ids // cps_res[i_pd]
    # indices are shifted by one
    para_dim_ids = para_dim_ids % cps_res[para_dim] + 1

    # Return new_spline
    spline_info = {}
    spline_info["control_points"] = spline_copy.cps[(para_dim_ids >= start_id)
                                                    & (para_dim_ids <= end_id)]
    spline_info["degrees"] = spline_copy.degrees.tolist()
    if start_id == end_id:
        spline_info["degrees"].pop(para_dim)
    if not is_bezier:
        spline_info["knot_vectors"] = spline_copy.knot_vectors.copy()
        if start_id == end_id:
            spline_info["knot_vectors"].pop(para_dim)
        else:
            spline_info["knot_vectors"][para_dim] = (
                    [spline_copy.knot_vectors[para_dim][start_id]]
                    + spline_copy.knot_vectors[para_dim]
                    [start_id:(end_id + spline_copy.degrees[para_dim])] + [
                            spline_copy.knot_vectors[para_dim]
                            [(end_id + spline_copy.degrees[para_dim] - 1)]
                    ]
            )
    if is_rational:
        spline_info["weights"] = spline_copy.weights[
                (para_dim_ids >= start_id) & (para_dim_ids <= end_id)]

    return type(spline)(**spline_info)


class Extractor:
    """Helper class to allow direct extraction from spline obj (BSpline or
    NURBS). Internal use only.

    Examples
    ---------
    >>> myspline = <your-spline>
    >>> spline_faces = myspline.extract.faces()
    """

    def __init__(self, spl):
        self._spline = spl

    def edges(self, *args, **kwargs):
        return edges(self._spline, *args, **kwargs)

    def faces(self, *args, **kwargs):
        return faces(self._spline, *args, **kwargs)

    def volumes(self, *args, **kwargs):
        return volumes(self._spline, *args, **kwargs)

    def control_points(self):
        return control_points(self._spline)

    def control_edges(self):
        return control_edges(self._spline)

    def control_faces(self):
        return control_faces(self._spline)

    def control_volumes(self):
        return control_volumes(self._spline)

    def control_mesh(self):
        return control_mesh(self._spline)

    def beziers(self):
        return beziers(self._spline)

    def spline(self, splittin_plane=None, interval=None):
        """Extract a spline from a spline.

        Use a (number of) splitting planes to extract a subsection from the
        parametric domain of it.

        Parameters
        ----------
        splitting_plane : int / dictionary (int : (float))
          if integer : parametric dimension to be extracted
          if dictionary : list of splitting planes and ranges to be passed
        interval : float / tuple<float,float>
          intervall or value in parametric space to be extracted from the
          spline representation
        Returns
        -------
        spline
        """
        if isinstance(splittin_plane, dict):
            if interval is not None:
                raise ValueError("Arguments incompatible expect dictionary")
            splittin_plane = dict(
                    sorted(splittin_plane.items(), key=lambda x: x[0])[::-1]
            )
            spline_copy = self._spline.copy()
            for key, item in splittin_plane.items():
                spline_copy = spline(spline_copy, key, item)
            return spline_copy
        else:
            return spline(self._spline, splittin_plane, interval)
