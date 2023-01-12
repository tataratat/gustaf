"""gustaf/spline/base.py.

Base for splines. Contains show and inherited classes from `spline`.
"""
from copy import deepcopy

import numpy as np
import splinepy

from gustaf import show as showmodule
from gustaf import utils
from gustaf._base import GustafBase
from gustaf.helpers.data import SplineData
from gustaf.spline import visualize
from gustaf.spline.create import Creator
from gustaf.spline.extract import Extractor
from gustaf.spline.proximity import Proximity


def show(spline, **kwargs):
    """Shows splines with various options. They are excessively listed, so that
    it can be adjustable.

    Parameters
    -----------
    spline: BSpline or NURBS
    resolutions: int or (spline.para_dim,) array-like
    control_points: bool
    knots: bool
    fitting_queries: bool
    return_discrete: bool
      Return dict of gustaf discrete objects, for example,
      {Vertices, Edges, Faces}, instead of opening a window
    return_showable: bool
      Return dict of showable objects.
    parametric_space: bool
      Only relevant for `vedo` backend.
    c: str
      Default is None. Black for curves, else green.
    alpha: float
    lighting: str
    control_point_ids: bool
    color_spline: str
    cmap: str

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

    para_space = kwargs.pop("parametric_space", False)
    if para_space:
        p_spline = spline.create.parametric_view()
        things_to_show["parametric_spline"] = p_spline

    if kwargs.get("return_discrete", False):
        return things_to_show

    if kwargs.get("return_showable", False):
        return {key: value.showable() for key, value in things_to_show.items()}

    if para_space:
        things_to_show.pop("parametric_spline")

        return showmodule.show_vedo(
            ["Parametric Space", p_spline],
            ["Physical Space", *things_to_show.values()],
            **kwargs,
        )

    return showmodule.show_vedo(things_to_show, **kwargs)


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
