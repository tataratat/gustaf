"""gustav/spline/base.py

Base for splines.
Contains show and inherited classes from `spline`.
"""

import abc

import splinepy
import numpy as np

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
        return_discrete=False,
        return_showable=False,
        backend=None,
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
    return_discrete: bool
      Return dict of gustav discrete objects, for example,
      {Vertices, Edges, Faces}, instead of opening a window
    return_showable: bool
      Return dict of showable objects.
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
      iff return_discrete==True, dict of gustav objects that are showable.
      iff return_showable==True, dict of backend objects that are showable.
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

    # determine backend
    if backend is None:
        backend = settings.VISUALIZATION_BACKEND

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
        # If

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
            knot_lines.vis_dict.update(c="black", lw=3)
            things_to_show.update(knots=knot_lines)

    if show_fitting_queries and hasattr(spline, "_fitting_queries"):
        fitting_queries = Vertices(spline._fitting_queries)
        fitting_queries.vis_dict.update(c="blue", r=10)
        things_to_show.update(fitting_queries=fitting_queries)

    # Return here, if backend is not vedo        
    if not backend.startswith("vedo"):
        # turn everything into backend showables
        if return_showable:
            for key, gusobj in things_to_show.items():
                things_to_show.update({key : gusobj.showable(backend=backend)})

            return things_to_show

        elif return_discrete:
            return things_to_show

        else:
            showmodule.show(list(things_to_show.values()))
            return None

    # iff backend is vedo, we provide fancier visualization
    elif backend.startswith("vedo"):
        # return if showable is not desired
        # -> From now on we will directly work on vedo objects.
        if return_discrete and not return_showable:
            return things_to_show

        # turn all gus objects into gus objects.
        vedo_things = dict()
        for key, gusobj in things_to_show.items():
            vedo_things.update({key : gusobj.showable(backend=backend)})

        # apply lighting
        if lighting is not None:
            vedo_things["spline"].lighting(lighting)

        # adjust surface alpha
        if spline.para_dim > 1:
            vedo_things["spline"].alpha(surface_alpha)

        # add red points for at control points
        if control_points and control_point_ids:
            vedo_things.update(
                control_point_ids=vedo_things["control_points"].labels("id")
            )

        # add knots as "x" for curves
        if knots and spline.para_dim == 1:
            uks = spline.unique_knots[0]
            phys_uks = showmodule.make_showable(
                Vertices(spline.evaluate([[uk] for uk in uks])),
                backend=backend,
            )
            xs = ["x"] * len(uks)

            vedo_things.update(
                knots=phys_uks.labels(xs, justify="center", c="green")
            )

        # generate parametric view of spline
        if parametric_space and spline.para_dim > 1:
            from vedo.addons import Axes
            from gustav.create.spline import parametric_view
            from gustav.utils.arr import bounds

            para_spline = parametric_view(spline)
            para_showables = show(
                para_spline,
                control_points=False,
                return_showable=True,
                lighting=lighting,
                knots=knots,
                parametric_space=False,
                backend=backend,
            )
            # Make lines a bit thicker
            if knots:
                para_showables["knots"].lw(6)

            # Trick to show begin/end value
            bs = np.asarray(
                bounds(para_showables["spline"].points())
            )
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

            if spline.para_dim == 3:
                axes_config.update(ztitle="w")
                axes_config.update(zrange=[lowerb[2], upperb[2]])
                axes_config.update(zMinorTicks=3)
                axes_config.update(zxGrid=False)

            para_showables.update(
                axes=Axes(para_showables["spline"], **axes_config)
            )

        # showable return
        if return_showable:
            if parametric_space:
                vedo_things.update(parametric_spline=para_showables)
            return vedo_things

        # now, show
        if parametric_space:
            para_showables.update(description="Parametric View")
            vedo_things.update(description="Physical View")
            showmodule.show_vedo(para_showables, vedo_things)

        else:
            showmodule.show_vedo(vedo_things)

        return None


class GustavSpline(GustavBase):

    @abc.abstractmethod
    def __init__(self):
        """
        Contructor as abstractmethod.
        """
        pass

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
        Equivalent to `gustav.spline.base.show(**kwrags)`
        """
        return show(self, **kwargs)

    def showable(self, **kwargs):
        """
        Equivalent to `gustav.spline.base.show(return_showable=True, **kwargs)`
        """
        return show(self, return_showable=True, **kwargs)

    def copy(self):
        """
        """
        return type(self)(**self.todict())


class BSpline(splinepy.BSpline, GustavSpline):

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
    def nurbs(self):
        """
        Returns same nurbs.
        Overwrites one from splinepy to return correct type.

        Parameters
        -----------
        None

        Returns
        --------
        same_nurbs: NURBS
        """
        from copy import deepcopy

        return NURBS(
            degrees=deepcopy(self.degrees),
            knot_vectors=deepcopy(self.knot_vectors),
            control_points=deepcopy(self.control_points),
            # fix dtype to match splinepy's dtype.
            weights=np.ones(self.control_points.shape[0], dtype="float64")
        )


class NURBS(splinepy.NURBS, GustavSpline):

    def __init__(
            self,
            degrees=None,
            knot_vectors=None,
            control_points=None,
            weights=None,
    ):
        """
        NURBS of gustav. Inherited from splinepy.NURBS.

        Attributes
        -----------
        extract: _Extractor

        Parameters
        -----------
        degrees: (para_dim,) list-like
        knot_vectors: (para_dim,) list
        control_points: (m, dim) list-like
        weights: (m, 1) list-like

        Returns
        --------
        None
        """
        super().__init__(
            degrees=degrees,
            knot_vectors=knot_vectors,
            control_points=control_points,
            weights=weights,
        )

        self._extractor = _Extractor(self)
