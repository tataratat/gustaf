"""gustaf/spline/visualize.py.

Spline visualization module. Supports visualization of following spline
(para_dim, dim) pairs: (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)
"""
import numpy as np

from gustaf import Vertices, settings
from gustaf.helpers import options
from gustaf.utils import log
from gustaf.utils.arr import enforce_len


class SplineShowOption(options.ShowOption):
    """
    Show options for splines.
    """

    __slots__ = ()

    # if we start to support more backends, most of this options should become
    # some sort of spline common.
    _valid_options = options.make_valid_options(
        *options.vedo_common_options,
        options.Option(
            "vedo",
            "control_points",
            "Show spline's control points."
            "Options propagates to control mesh, unless it is specified.",
            (bool,),
        ),
        options.Option(
            "vedo",
            "control_mesh",
            "Show spline's control mesh.",
            (bool,),
        ),
        options.Option("vedo", "knots", "Show spline's knots.", (bool,)),
        options.Option(
            "vedo",
            "fitting_queries",
            "Shows fitting queries if they are locally saved in splines.",
            (bool,),
        ),
        options.Option(
            "vedo",
            "control_points_alpha",
            "Transparency of control points in range [0, 1].",
            (float, int),
        ),
        options.Option(
            "vedo",
            "control_point_ids",
            "Show ids of control_points.",
            (bool,),
        ),
        options.Option(
            "vedo",
            "resolutions",
            "Sampling resolution for spline.",
            (int, list, tuple, np.ndarray),
        ),
        options.Option(
            "vedo",
            "arrowdata_on",
            "Specify parametric coordinates to place arrowdata.",
            (list, tuple, np.ndarray),
        ),
    )

    _helps = "GustafSpline"

    def __init__(self, helpee):
        """
        Parameters
        ----------
        helpee: GustafSpline
        """
        self._helpee = helpee
        # checks if helpee inherits from GustafSpline
        if self._helps not in str(type(helpee).__mro__):
            raise TypeError(
                f"This show option if for {self._helps}.",
                f"Given helpee is {type(helpee)}.",
            )
        self._options = dict()
        self._backend = settings.VISUALIZATION_BACKEND
        self._options[self._backend] = dict()


def make_showable(spline):
    return eval(f"_{spline.show_options._backend}_showable(spline)")


def _vedo_showable(spline):
    """
    Goes through common precedures for preparing showable splines.

    Parameters
    ----------
    None

    Returns
    -------
    gus_dict: dict
      dict of sampled spline as gustaf objects
    """
    # get spline and knots
    gus_primitives = eval(f"_vedo_showable_para_dim_{spline.para_dim}(spline)")

    # apply spline color
    sampled_spline = gus_primitives["spline"]
    default_color = "green" if spline.para_dim > 1 else "black"
    sampled_spline.show_options["c"] = spline.show_options.get(
        "c", default_color
    )
    sampled_spline.show_options["alpha"] = spline.show_options.get("alpha", 1)
    sampled_spline.show_options["lighting"] = spline.show_options.get(
        "lighting", "glossy"
    )

    # axis?
    sampled_spline.show_options["axes"] = spline.show_options.get(
        "axes", False
    )

    # with color representable, scalar field data
    res = enforce_len(
        spline.show_options.get("resolutions", 100), spline.para_dim
    )
    dataname = spline.show_options.get("dataname", None)
    sampled_splinedata = spline.splinedata.as_scalar(
        dataname, res, default=None
    )
    if dataname is not None and sampled_splinedata is not None:
        # transfer data
        sampled_spline.vertexdata[dataname] = sampled_splinedata

        # transfer options - maybe vectorized query?
        keys = ("vmin", "vmax", "scalarbar", "cmap", "cmapalpha")
        spline.show_options.copy_valid_options(
            sampled_spline.show_options, keys
        )

        # mark that we want to see this data
        sampled_spline.show_options["dataname"] = dataname

    elif dataname is not None and sampled_splinedata is None:
        log.debug(f"No splinedata named ({dataname}) for {spline}. Skipping")

    # with arrow representable, vector data
    adata_name = spline.show_options.get("arrowdata", None)
    adapted_adata = spline.splinedata.get(adata_name, None)
    if adata_name is not None and adapted_adata is not None:
        # if location is specified, this will be a separate Vertices obj with
        # configured arrowdata
        has_locations = adapted_adata.has_locations
        adata_on = "arrowdata_on" in spline.show_options.keys()
        create_vertices = has_locations or adata_on

        # this case causes conflict of interest. raise
        if has_locations and adata_on:
            raise ValueError(
                f"arrowdata-({adata_name}) has fixed location, "
                "but and `arrowdata_on` is set.",
            )

        if create_vertices:
            # prepare corresponding queries
            if adapted_adata.has_locations:
                queries = adapted_adata.locations
                on = None
            else:
                queries = spline.show_options["arrowdata_on"]
                on = queries

            # bound /  dim check
            bounds = spline.parametric_bounds
            if queries.shape[1] != len(bounds[0]):
                raise ValueError(
                    "Dimension mismatch: arrowdata locations-"
                    f"{queries.shape[1]} / para_dim-{spline.para_dim}."
                )
            # tolerance padding. may still cause issues in splinepy.
            # in that case, we will have to scale queries.
            lb_diff = queries.min(axis=0) - bounds[0] + settings.TOLERANCE
            ub_diff = queries.max(axis=0) - bounds[1] - settings.TOLERANCE
            if any(lb_diff < 0) or any(ub_diff > 0):
                raise ValueError(
                    f"Specified locations of ({adata_name}) are out side the "
                    f"parametric bounds ({bounds}) by [{lb_diff}, {ub_diff}]."
                )

            # get arrow
            adata = spline.splinedata.as_arrow(adata_name, on=on)

            # create vertices that can be shown as arrows
            loc_vertices = Vertices(spline.evaluate(queries), copy=False)
            loc_vertices.vertexdata[adata_name] = adata

            # transfer options
            keys = ("arrowdata_scale", "arrowdata_color", "arrowdata")
            spline.show_options.copy_valid_options(
                loc_vertices.show_options, keys
            )

            # add to primitives
            gus_primitives["arrowdata"] = loc_vertices

        else:  # sample arrows and append to sampled spline.
            sampled_spline.vertexdata[adata_name] = spline.splinedata.as_arrow(
                adata_name, resolutions=res
            )
            # transfer options
            keys = ("arrowdata_scale", "arrowdata_color", "arrowdata")
            spline.show_options.copy_valid_options(
                sampled_spline.show_options, keys
            )

    # double check on same obj ref
    gus_primitives["spline"] = sampled_spline

    # control_points & control_points_alpha
    show_cps = spline.show_options.get("control_points", True)
    if show_cps:
        # control points (vertices)
        cps = spline.extract.control_points()
        cps.show_options["c"] = "red"
        cps.show_options["r"] = 10
        cps.show_options["alpha"] = spline.show_options.get(
            "control_points_alpha", 0.8
        )
        # add
        gus_primitives["control_points"] = cps

        if spline.show_options.get("control_point_ids", True):
            # a bit redundant, but nicely separable
            cp_ids = spline.extract.control_points()
            cp_ids.show_options["labels"] = np.arange(len(cp_ids.vertices))
            cp_ids.show_options["label_options"] = {"font": "VTK"}
            gus_primitives["control_point_ids"] = cp_ids

    if spline.show_options.get("control_mesh", show_cps):
        # pure control mesh
        cmesh = spline.extract.control_mesh()  # either edges or faces
        if spline.para_dim != 1:
            cmesh = cmesh.toedges(unique=True)

        cmesh.show_options["c"] = "red"
        cmesh.show_options["lw"] = 4
        cmesh.show_options["alpha"] = spline.show_options.get(
            "control_points_alpha", 0.8
        )
        # add
        gus_primitives["control_mesh"] = cmesh

    # fitting queries
    if hasattr(spline, "_fitting_queries") and spline.show_options.get(
        "fitting_queries", True
    ):
        fqs = Vertices(spline._fitting_queries)
        fqs.show_options["c"] = "blue"
        fqs.show_options["r"] = 10
        gus_primitives["fitting_queries"] = fqs

    return gus_primitives


def _vedo_showable_para_dim_1(spline):
    """
    Assumes showability check has been already performed

    Parameters
    ----------
    spline: GustafSpline

    Returns
    -------
    gus_primitives: dict
      keys are {spline, knots}
    """
    gus_primitives = dict()
    res = enforce_len(
        spline.show_options.get("resolutions", 100), spline.para_dim
    )
    sp = spline.extract.edges(res[0])

    # specify curve width
    sp.show_options["lw"] = 8
    # add spline
    gus_primitives["spline"] = sp

    # place knots
    if spline.show_options.get("knots", True):
        uks = np.asanyarray(spline.unique_knots[0]).reshape(-1, 1)
        knots = Vertices(spline.evaluate(uks))
        knots.show_options["labels"] = ["x"] * len(uks)
        knots.show_options["label_options"] = {
            "justify": "center",
            "c": "green",
        }
        gus_primitives["knots"] = knots

    return gus_primitives


def _vedo_showable_para_dim_2(spline):
    """
    Assumes showability check has been already performed

    Parameters
    ----------
    spline: GustafSpline

    Returns
    -------
    gus_primitives: dict
      keys are {spline, knots}
    """
    gus_primitives = dict()
    res = enforce_len(
        spline.show_options.get("resolutions", 100), spline.para_dim
    )
    sp = spline.extract.faces(res)
    gus_primitives["spline"] = sp

    # knots
    if spline.show_options.get("knots", True):
        knot_lines = spline.extract.edges(res, all_knots=True)
        knot_lines.show_options["c"] = "black"
        knot_lines.show_options["lw"] = 3
        gus_primitives["knots"] = knot_lines

    return gus_primitives


def _vedo_showable_para_dim_3(spline):
    """
    Currently same as _vedo_showable_para_dim_2
    """
    return _vedo_showable_para_dim_2(spline)
