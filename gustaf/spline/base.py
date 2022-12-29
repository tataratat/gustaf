"""gustaf/spline/base.py.

Base for splines. Contains show and inherited classes from `spline`.
"""
from copy import deepcopy

import numpy as np
import splinepy

from gustaf import settings
from gustaf import utils
from gustaf import show as showmodule
from gustaf._base import GustafBase
from gustaf.spline import visualize
from gustaf.spline.create import Creator
from gustaf.spline.extract import Extractor
from gustaf.spline.proximity import Proximity
from gustaf.utils.arr import enforce_len
from gustaf.helpers.data import SplineData 
from gustaf.vertices import Vertices


def show(
    spline,
#    resolutions=100,
#    control_points=True,
#    knots=True,
#    show_fitting_queries=True,
#    return_discrete=False,
#    return_showable=False,
#    backend=None,
    # From here, | only relevant if "vedo" is backend.
    #            V
#    parametric_space=False,
#    color=None,
#    surface_alpha=1,
#    control_points_alpha=0.8,
#    lighting="glossy",
#    control_point_ids=True,
#    color_spline=None,
#    cmap=None,  # only required
    **kwargs,
):
    """Shows splines with various options. They are excessively listed, so that
    it can be adjustable.

    Parameters
    -----------
    spline: BSpline or NURBS
    resolutions: int or (spline.para_dim,) array-like
    control_points: bool
    knots: bool
    show_fitting_queries: bool
    return_discrete: bool
      Return dict of gustaf discrete objects, for example,
      {Vertices, Edges, Faces}, instead of opening a window
    return_showable: bool
      Return dict of showable objects.
    parametric_space: bool
      Only relevant for `vedo` backend.
    color: str
      Default is None. Black for curves, else green.
    surface_alpha: float
      Only relevant for `vedo` backend. Effective range [0, 1].
    lighting: str
      Only relevant for `vedo` backend.
    control_point_ids: bool
    color_spline: str
    cmap: str
      Only relevant for `vedo` backend and color_spline is not None.

    Returns
    --------
    things_to_show: dict
      iff return_discrete==True, dict of gustaf objects that are showable.
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

    if kwargs:
        orig_show_options = spline.show_options
        spline._show_options = spline.__show_option__(spline)
        orig_show_options.copy_valid_options(spline.show_options)
        for key, value in kwargs.items():
            try:
                spline.show_options[key] = value
            except BaseException:
                utils.log.debug(
                    f"Skipping invalid option {key} for "
                    f"{spline.show_options._helps}."
                )
                continue

    # Prepare things to show dict.
    things_to_show = visualize.make_showable(spline)

#        # generate parametric view of spline
#        if parametric_space and spline.para_dim > 1:
#            from vedo.addons import Axes
#
#            from gustaf.spline.create import parametric_view
#            from gustaf.utils.arr import bounds
#
#            para_spline = parametric_view(spline)
#            para_showables = show(
#                para_spline,
#                control_points=False,
#                return_showable=True,
#                lighting=lighting,
#                knots=knots,
#                parametric_space=False,
#                backend=backend,
#            )
#            # Make lines a bit thicker
#            if knots:
#                para_showables["knots"].lw(6)
#
#            # Trick to show begin/end value
#            bs = np.asarray(bounds(para_showables["spline"].points()))
#            bs_diff_001 = (bs[1] - bs[0]) * 0.001
#            lowerb = bs[0] - bs_diff_001
#            upperb = bs[1] + bs_diff_001
#
#            axes_config = dict(
#                xtitle="u",
#                ytitle="v",
#                xrange=[lowerb[0], upperb[0]],
#                yrange=[lowerb[1], upperb[1]],
#                tipSize=0,
#                xMinorTicks=3,
#                yMinorTicks=3,
#                xyGrid=False,
#                yzGrid=False,
#            )
#
#            if spline.para_dim == 3:
#                axes_config.update(ztitle="w")
#                axes_config.update(zrange=[lowerb[2], upperb[2]])
#                axes_config.update(zMinorTicks=3)
#                axes_config.update(zxGrid=False)
#
#            para_showables.update(
#                axes=Axes(para_showables["spline"], **axes_config)
#            )
#
#        # showable return
#        if return_showable:
#            if parametric_space:
#                vedo_things.update(parametric_spline=para_showables)
#            return vedo_things
#
#        # now, show
#        if parametric_space:
#            para_showables.update(description="Parametric View")
#            vedo_things.update(description="Physical View")
#            plt = showmodule.show_vedo(para_showables, vedo_things, **kwargs)
#
#        else:
#            plt = showmodule.show_vedo(vedo_things, **kwargs)

    if kwargs.get("return_showable", False):
        return {key: value.showable() for key, value in things_to_show.items()}

    return showmodule.show_vedo(things_to_show)


class GustafSpline(GustafBase):
    __show_option__ = visualize.SplineShowOption

    def __init__(self):
        """Contructor as abstractmethod.

        This needs to be inherited first to make sure duplicating
        functions properly override splinepy.Spline
        """
        self._extractor = Extractor(self)
        self._proximity = Proximity(self)
        self._creator = Creator(self)
        self._show_options = self.__show_option__(self)
        self._splinedata = SplineData(self)

    @property
    def show_options(self):
        """
        Show option manager for splines.

        Parameters
        ----------
        None

        Returns
        -------
        show_options: SplineShowOption
        """
        return self._show_options

    @property
    def splinedata(self):
        """
        Spline data helper for splines.

        Parameters
        ----------
        None

        Returns
        -------
        splinedata: SplineData
        """
        return self._splinedata

    @property
    def extract(self):
        """Returns spline extracter. Can directly perform extractions available
        at `gustaf/spline/extract.py`. For more info, take a look at
        `gustaf/spline/extract.py`: Extracter.

        Examples
        ---------
        >>> splinefaces = spline.extract.faces()

        Parameters
        -----------
        None

        Returns
        --------
        spline_extracter: Extracter
        """
        return self._extractor

    @property
    def create(self):
        """Returns spline creator Can be used to create new splines using
        geometric relations.

        Examples
        --------
        >>> prism = spline.create.extrude(axis=[1,4,1])

        Parameters
        ----------
        None

        Returns
        spline.Creator
        """
        return self._creator

    @property
    def proximity(self):
        """Returns spline proximity helper. Can directly perform proximity
        queries available at `gustaf/spline/proximity.py`. For more info, take
        a look at `gustaf/spline/proximity.py`: Proximity.

        Examples
        ---------
        >>> closest_cp_ids = spline.proximity.closest_control_points(queries)
        >>> closest_cp_ids, distances =\
        ...    spline.proximity.closest_control_points(
        ...       queries,
        ...       return_distances=True
        ...    )

        Parameters
        -----------
        None

        Returns
        --------
        spline_proximity: Proximity
        """
        return self._proximity

    def show(self, **kwargs):
        """Equivalent to `gustaf.spline.base.show(**kwargs)`"""
        return show(self, **kwargs)

    def showable(self, **kwargs):
        """Equivalent to
        `gustaf.spline.base.show(return_showable=True,**kwargs)`"""
        return show(self, return_showable=True, **kwargs)

    def copy(self):
        """tmp copy from splinepy until #89 merges"""
        new = type(self)()
        new.new_core(**self._data["properties"], properties_round_trip=False)
        new._data = deepcopy(self._data)
        return new


class Bezier(GustafSpline, splinepy.Bezier):
    def __init__(
        self,
        degrees=None,
        control_points=None,
        spline=None,
    ):
        """Bezier of gustaf. Inherited from splinepy.Bezier and GustafSpline.

        Attributes
        -----------
        extract: Extractor
        create: Creator
        proximity: Proximity

        Parameters
        -----------
        degrees: (para_dim,) list-like
        control_points: (m, dim) list-like

        Returns
        --------
        None
        """
        splinepy.Bezier.__init__(
            self, degrees=degrees, control_points=control_points, spline=spline
        )
        GustafSpline.__init__(self)

    @property
    def bezier(self):
        """Returns same parametric representation as Bezier Spline.

        Parameters
        ----------
        None

        Returns
        -------
        same : Bezier
        """
        return self.copy()

    @property
    def rationalbezier(self):
        """Returns same parametric representation as Rational Bezier Spline.

        Parameters
        ----------
        None

        Returns
        -------
        same : RationalBezier
        """
        return RationalBezier(
            degrees=self.degrees,
            control_points=self.control_points,
            weights=np.ones(self.control_points.shape[0]),
        )

    @property
    def bspline(self):
        """Returns same parametric representation as BSpline.

        Parameters
        -----------
        None

        Returns
        --------
        same_bspline : BSpline
        """
        return BSpline(
            degrees=self.degrees,
            control_points=self.control_points,
            knot_vectors=[
                [0] * (self.degrees[i] + 1) + [1] * (self.degrees[i] + 1)
                for i in range(self.para_dim)
            ],
        )

    @property
    def nurbs(self):
        """Returns same parametric representation as nurbs.

        Parameters
        -----------
        None

        Returns
        --------
        same_nurbs: NURBS
        """
        return self.bspline.nurbs


class RationalBezier(GustafSpline, splinepy.RationalBezier):
    def __init__(
        self,
        degrees=None,
        control_points=None,
        weights=None,
        spline=None,
    ):
        """Rational Bezier of gustaf. Inherited from splinepy.RationalBezier
        and GustafSpline.

        Attributes
        -----------
        extract: Extractor
        create: Creator
        proximity: Proximity

        Parameters
        -----------
        degrees: (para_dim,) list-like
        control_points: (m, dim) list-like
        weights : (m) list-like

        Returns
        --------
        None
        """
        splinepy.RationalBezier.__init__(
            self,
            degrees=degrees,
            control_points=control_points,
            weights=weights,
            spline=spline,
        )
        GustafSpline.__init__(self)

    @property
    def rationalbezier(self):
        """Returns same parametric representation as Rational Bezier Spline.

        Parameters
        ----------
        None

        Returns
        -------
        same : RationalBezier
        """
        return self.copy()

    @property
    def nurbs(self):
        """Returns same parametric representation as nurbs.

        Parameters
        -----------
        None

        Returns
        --------
        same_nurbs: NURBS
        """
        return NURBS(
            degrees=self.degrees,
            control_points=self.control_points,
            knot_vectors=[
                [0] * (self.degrees[i] + 1) + [1] * (self.degrees[i] + 1)
                for i in range(self.para_dim)
            ],
            weights=self.weights,
        )


class BSpline(GustafSpline, splinepy.BSpline):
    def __init__(
        self,
        degrees=None,
        knot_vectors=None,
        control_points=None,
        spline=None,
    ):
        """BSpline of gustaf. Inherited from splinepy.BSpline and GustafSpline.

        Attributes
        -----------
        extract: Extractor
        create: Creator
        proximity: Proximity

        Parameters
        -----------
        degrees: (para_dim,) list-like
        knot_vectors: (para_dim, ...) list
        control_points: (m, dim) list-like

        Returns
        --------
        None
        """
        splinepy.BSpline.__init__(
            self,
            degrees=degrees,
            knot_vectors=knot_vectors,
            control_points=control_points,
            spline=spline,
        )

        GustafSpline.__init__(self)

    @property
    def bspline(self):
        """Returns same parametric representation as BSpline.

        Parameters
        -----------
        None

        Returns
        --------
        same_bspline : BSpline
        """
        return self.copy()

    @property
    def nurbs(self):
        """Returns same nurbs. Overwrites one from splinepy to return correct
        type.

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
            weights=np.ones(self.control_points.shape[0], dtype="float64"),
        )


class NURBS(GustafSpline, splinepy.NURBS):
    def __init__(
        self,
        degrees=None,
        knot_vectors=None,
        control_points=None,
        weights=None,
        spline=None,
    ):
        """NURBS of gustaf. Inherited from splinepy.NURBS.

        Attributes
        -----------
        extract: Extractor
        create: Creator
        proximity: Proximity

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
        splinepy.NURBS.__init__(
            self,
            degrees=degrees,
            knot_vectors=knot_vectors,
            control_points=control_points,
            weights=weights,
            spline=spline,
        )
        GustafSpline.__init__(self)

    @property
    def _mfem_ids(self):
        """Returns mfem index mapping. For ease of use.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        if self.para_dim != 2:
            raise NotImplementedError(
                "Sorry, only available for para_dim = 2 splines"
            )

        gustaf2mfem, mfem2gustaf = splinepy.io.mfem.mfem_index_mapping(
            self.para_dim,
            self.degrees,
            self.knot_vectors,
        )

        return gustaf2mfem, mfem2gustaf

    @property
    def nurbs(self):
        """Returns same parametric representation as nurbs.

        Parameters
        -----------
        None

        Returns
        --------
        same_nurbs: NURBS
        """
        return self.copy()


def from_mfem(nurbs_dict):
    """Construct a gustaf NURBS. Reorganizes control points and weights.

    Parameters
    -----------
    nurbs_dict: dict

    Returns
    --------
    nurbs: NURBS
    """
    _, m2gus = splinepy.io.mfem.mfem_index_mapping(
        len(nurbs_dict["degrees"]),
        nurbs_dict["degrees"],
        nurbs_dict["knot_vectors"],
    )

    return NURBS(
        degrees=nurbs_dict["degrees"],
        knot_vectors=nurbs_dict["knot_vectors"],
        control_points=nurbs_dict["control_points"][m2gus],
        weights=nurbs_dict["weights"][m2gus],
    )


def load_splines(fname):
    """Loads and creates gustaf NURBS. Does not perform any check or tests.

    Parameters
    -----------
    fname: str

    Returns
    --------
    gussplines: list
    """
    # first get dict_splines using splinepy
    dictsplines = splinepy.load_splines(fname, as_dict=True)

    # try to initialize with correct spline type
    gussplines = list()
    for dics in dictsplines:
        is_bspline = "knot_vectors" in dics
        is_nurbs = "weights" in dics

        if is_nurbs:
            gussplines.append(NURBS(**dics))
        elif is_bspline:
            gussplines.append(BSpline(**dics))
        else:
            gussplines.append(Bezier(**dics))

    return gussplines
