"""gustaf/spline/visualize.py.

Spline visualization module. Supports visualization of following spline
(para_dim, dim) pairs: ((1, 2), (1, 3), (2, 2), (2, 3), (3, 3)
"""
import numpy as np

from gustaf import Vertices, settings
from gustaf.helpers import options
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
            "Show spline's control points and control mesh.",
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


def _vedo_common(spline):
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
