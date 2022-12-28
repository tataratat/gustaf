"""gustaf/spline/visualize.py.

Spline visualization module. Supports visualization of following spline
(para_dim, dim) pairs: ((1, 2), (1, 3), (2, 2), (2, 3), (3, 3)
"""
import numpy as np

from gustaf.helpers import options


class SplineShowOption(options.ShowOption):
    """
    Show options for splines.
    """

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


def _vedo_show_para_dim_1():
    pass
