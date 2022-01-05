"""gustav/spline/base.py

Base for splines.
Contains show and inherited classes from `spline`.
"""

import splinepy

from gustav import settings
from gustav import show as showmodule
from gustav._base import GustavBase
from gustav.vertices import Vertices
from gustav.spline.extract import _Extractor
from gustav.spline._utils import to_res_list

def show(
        spline,
        resolutions=100,
        control_points=True,
        knots=True,
        show_fitting_queries=True,
        return_showables=False,
        return_vedo_showables=True,
        # From here, | only relevant if "vedo" is backend.
        #            V
        parametric_space=False,
        surface_alpha=1,
        lighting="glossy",
        control_point_ids=True,
        solution_cps=None,
        solution_spline=None,
):
    """
    Shows splines with various options.
    They are excessively listed, so that it can be adjustable.

    Parameters
    -----------
    spline: BSpline or NURBS
    resolutions: int or (spline.para_dim,) array-like
    control_points: bool
    knots: bool
    show_fitting_queries: bool
    return_showables: bool
    return_vedo_showables: bool
      Only relevant iff visualization backend is `vedo`
      and return_showable is True.
    parametric_space: bool
      Only relevant for `vedo` backend.
    surface_alpha: float
      Only relevant for `vedo` backend. Effective range [0, 1].
    lighting: str
      Only relevant for `vedo` backend.
    control_point_ids: bool
    field_name: str

    Returns
    --------
    things_to_show: dict
      list of gustav objects that are showable.
    """
    # Showing is only possible for following splines
    allowed_dim_combo = (
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
        (3, 3),
    )
    if (spline.para_dim, spline.dim) not in allowed_dim_combo:
        raise ValueError("Sorry, can't show given spline.")

    # Prepare correct form of resolution input
    resolutions = to_res_list(resolutions, spline.para_dim)

    # Prepare things to show dict.
    things_to_show = dict()

    # (discretized) spline itself with basic color scheme.
    if spline.para_dim == 1:
        sp = spline.extract.edges(resolutions[0])
        sp.vis_dict.update(c="black", lw=8)

    if spline.para_dim == 2 or spline.para_dim == 3:
        sp = spline.extract.faces(resolutions)
        sp.vis_dict.update(c="green")

    things_to_show.update(spline=sp)

    # control_points = control mesh + control_points
    # control mesh always takes unique edge representation.
    if control_points:
        control_mesh = spline.extract.control_mesh()
        if spline.para_dim != 1:
            control_mesh = control_mesh.toedges(unique=True)

        # Set alpha to < 1, so that they don't "overshadow" spline
        control_mesh.vis_dict.update(c="red", lw=4, alpha=.8)
        things_to_show.update(control_mesh=control_mesh) # mesh itself
        # Add big vertices to emphasize cps.
        cps = control_mesh.tovertices()
        cps.vis_dict.update(c="red", r=10, alpha=.8)
        things_to_show.update(control_points=cps) # only points

    if knots:
        # Knot lines for non-curve splines.
        # Knot for curves are only added for vedo backend.
        if spline.para_dim > 1:
            knot_lines = spline.extract.edges(resolutions[0], all_knots=True)
            knot_lines.vis_dict.update(c="black", lw=6)
            things_to_show.update(knots=knot_lines)

    if show_fitting_queries and hasattr(spline, "_fitting_queries"):
        fitting_queries = Vertices(spline._fitting_queries)
        fitting_queries.vis_dict.update(c="blue", r=10)
        things_to_show.update(fitting_queries=fitting_queries)

    # Return here, if backend is not vedo        
    if not settings.VISUALIZATION_BACKEND.startswith("vedo"):
        if return_showables:
            return things_to_show

        else:
            showmodule.show(list(things_to_show.values()))
            return None

    # iff backend is vedo, we provide fancier visualization
    elif settings.VISUALIZATION_BACKEND.startswith("vedo"):
        # showable, but not specifically vedo_showable, then return
        if return_showables and not return_vedo_showables:
            return things_to_show

        vedo_things = dict()
        for key, gusobj in things_to_show.items():
            vedo_things.update({key : showmodule.make_showable(gusobj)})

        if lighting is not None:
            vedo_things["spline"].lighting(lighting)

        if spline.para_dim > 1:
            vedo_things["spline"].alpha(surface_alpha)

        if control_points and control_point_ids:
            vedo_things.update(
                control_point_ids=vedo_things["control_points"].labels("id")
            )

        # Add knots as "x" for curves
        if knots and spline.para_dim == 1:
            uks = spline.unique_knots[0]
            phys_uks = showmodule.make_showable(
                Vertices(spline.evaluate([[uk] for uk in uks]))
            )
            xs = ["x"] * len(uks)

            vedo_things.update(
                knots=phys_uks.labels(xs, justify="center", c="green")
            )

        if parametric_space and spline.para_dim > 1:
            from vedo.addons import Axes
            from gustav.create.splines import knot_vector_bounded

            kv_spline = knot_vector_bounded(spline.knot_vectors)
            kvs_showables = show(
                kv_spline,
                control_points=False,
                return_showables=True,
                return_vedo_showables=True,
                lighting=lighting,
                knots=knots,
            )
            # Make lines a bit thicker
            for l in naive_things[1:]: l.lw(3)

            # Trick to show begin/end value
            bs = np.asarray(naive_things[0].bounds()).reshape(-1,2).T
            bs_diff_001 = (bs[1] - bs[0]) * 0.001
            lowerb = bs[0] - bs_diff_001 
            upperb = bs[1] + bs_diff_001

            axes_config = dict(
                xtitle="u",
                ytitle="v",
                xrange=[lowerb[0], upperb[0]],
                yrange=[lowerb[1], upperb[1]],
                tipSize=0,
                xMinorTicks=3,
                yMinorTicks=3,
                xyGrid=False,
                yzGrid=False,
            )

            if self._para_dim == 3:
                axes_config.update(ztitle="w")
                axes_config.update(zrange=[lowerb[2], upperb[2]])
                axes_config.update(zMinorTicks=3)
                axes_config.update(zxGrid=False)

            naive_things.append(Axes(naive_things[0], **axes_config))

        # showable return
        if return_vedo_showables and return_showables:
            return vedo_things

        # now, show
        showmodule.show_vedo(vedo_things)

        return None

class BSpline(splinepy.BSpline, GustavBase):

    def __init__(
            self,
            degrees=None,
            knot_vectors=None,
            control_points=None,
    ):
        """
        BSpline of gustav. Inherited from splinepy.BSpline.

        Attributes
        -----------
        extract: _Extractor

        Parameters
        -----------
        degrees: (para_dim,) list-like
        knot_vectors: (para_dim, ...) list
        control_points: (m, dim) list-like

        Returns
        --------
        None
        """
        super().__init__(
            degrees=degrees,
            knot_vectors=knot_vectors,
            control_points=control_points
        )

        self._extractor = _Extractor(self)

    @property
    def extract(self):
        """
        Returns spline extracter.
        Can directly perform extractions available at
        `gustav/spline/extract.py`.
        For more info, take a look at `gustav/spline/extract.py`: _Extracter.

        Examples
        ---------
        >>> splinefaces = spline.extract.faces()

        Parameters
        -----------
        None

        Returns
        --------
        spline_extracter: _Extracter
        """
        return self._extractor

    def show(self, **kwargs):
        """
        """
        return show(self, **kwargs)

class NURBS(splinepy.NURBS, GustavBase):
    pass
