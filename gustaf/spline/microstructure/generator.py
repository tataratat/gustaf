import numpy as np

from gustaf._base import GustafBase
from gustaf.spline import base


class Generator(GustafBase):
    """
    Helper class to facilitatae the construction of microstructures
    """

    def __init__(
            self,
            deformation_function=None,
            tiling=None,
            microtile=None,
            parametrization_function=None
    ):
        """
        Helper class to facilitatae the construction of microstructures

        Parameters
        ----------
        deformation_function : spline
          Outer function that describes the contour of the microstructured
          geometry
        tiling : list of integers
          microtiles per parametric dimension
        microtile : spline or list of splines
          Representation of the building block defined in the unit cube
        parametrization_function : Callable (optional)
          Function to describe spline parameters
        """
        if deformation_function is not None:
            self.deformation_function = deformation_function

        if tiling is not None:
            self.tiling = tiling

        if microtile is not None:
            self.microtile = microtile

        if parametrization_function is not None:
            self.parametrization_function = parametrization_function

    @property
    def deformation_function(self):
        """
        Deformation function defining the outer geometry (contour) of the
        microstructure.

        Parameters
        ----------
        None

        Returns
        -------
        deformation_function : spline
        """
        if hasattr(self, "_deformation_function"):
            return self._deformation_function
        else:
            return None

    @deformation_function.setter
    def deformation_function(self, deformation_function):
        """
        Deformation function setter defining the outer geometry of the
        microstructure. Must be spline type and as such inherit from
        gustaf.GustafSpline

        Parameters
        ----------
        deformation_function : spline

        Returns
        -------
        None
        """
        if not issubclass(type(deformation_function), base.GustafSpline):
            raise ValueError(
                    "Deformation function must be Gustaf-Spline."
                    " e.g. gustaf.NURBS"
            )
        self._deformation_function = deformation_function
        self._sanity_check()

    @property
    def tiling(self):
        """
        Number of microtiles per parametric dimension

        Parameters
        ----------
        None

        Returns
        -------
        tiling : list<int>
        """
        if hasattr(self, "_tiling"):
            return self._tiling
        else:
            return None

    @tiling.setter
    def tiling(self, tiling, knot_span_wise=True):
        """
        Setter for the tiling attribute, defining the number of microtiles per
        parametric dimension

        Parameters
        ----------
        tiling : int / list<int>
          Number of tiles for each dimension respectively
        knot_span_wise : bool
          Insertion per knotspan vs. total number per paradim
        """
        if not isinstance(tiling, list):
            if not isinstance(tiling, int):
                raise ValueError(
                        "Tiling mus be either list of integers of integer "
                        "value"
                )
        self._tiling = tiling
        # Is defaulted to False using function arguments
        self._tiling_per_knot_span = bool(knot_span_wise)
        self._sanity_check()
        self._logd(f"Successfully set tiling to : {self.tiling}")

    @property
    def microtile(self):
        """
        Microtile that is either a spline, a list of splines, or a class that
        provides a `create_tile` function.
        """
        if hasattr(self, "_microtile"):
            return self._microtile
        else:
            return None

    @microtile.setter
    def microtile(self, microtile):
        """
        Setter for microtile

        Microtile must be either a spline, a list of splines, or a class that
        provides (at least) a `create_tile` function and a `dim` member.

        Parameters
        ----------
        microtile : spline / list<splines> / user-object
          arbitrary long list of splines that define the microtile

        Returns
        -------
        None
        """
        # place single tiles into a list to provide common interface
        if (
                isinstance(microtile, list)
                or issubclass(type(microtile), base.GustafSpline)
        ):
            microtile = self._make_microtilable(microtile)
        # Assign Microtile object to member variable
        self._microtile = microtile

        self._sanity_check()

    @property
    def parametrization_function(self):
        """
        Optional function, that - if required - parametrizes the microtiles

        In order to use said function, the Microtile needs to provide a couple
        of attributes
          1. `evaluation_points` - a list of points defined in the unit cube
             that will be evaluated in the parametrization function to provide
             the required set of data points
          2. `parameter_space_dimension` - dimensionality of the
              parametrization function and number of design variables for said
              microtile

        Parameters
        ----------
        None

        Returns
        -------
        parametrization_function : Callable


        """
        if hasattr(self, "_parametrization_function"):
            return self._parametrization_function
        else:
            return None

    @parametrization_function.setter
    def parametrization_function(self, parametrization_function):
        """
        Optional function, that - if required - parametrizes the microtiles

        In order to use said function, the Microtile needs to provide a couple
        of attributes
          1. `evaluation_points` - a list of points defined in the unit cube
             that will be evaluated in the parametrization function to provide
             the required set of data points
          2. `parameter_space_dimension` - dimensionality of the
              parametrization function and number of design variables for said
              microtile

        Parameters
        ----------
        Callable

        Returns
        None
        """
        if not callable(parametrization_function):
            raise ("parametrization_function must be callable")
        self._parametrization_function = parametrization_function
        self._sanity_check()

    def create(self, closing_faces=None, **kwargs):
        """
        Create a Microstructure

        Parameters
        ----------
        closing_faces : int
          If not None, Microtile must provide a function `closing_tile`
        **kwargs
          will be passed to `create_tile` function

        Returns
        -------
        Microstructure : list<spline>
          finished microstructure based on object requirements
        """
        import itertools

        # Check if all information is gathered
        if not self._sanity_check():
            raise ValueError("Not enough information provided, abort")
        # check if user wants closed structure
        is_closed = closing_faces is not None
        if is_closed:
            is_closed = True
            if closing_faces >= self._deformation_function.para_dim:
                raise ValueError(
                        "closing face must be smaller than the deformation "
                        "function's parametric dimension"
                )
            if self._parametrization_function is None:
                raise ValueError(
                        "Faceclosure is currently only implemented for "
                        "parametrized microstructures"
                )

        # Prepare the deformation function
        # Transform into a non-uniform splinetype and make sure to work on copy
        if hasattr(self._deformation_function, "bspline"):
            deformation_function_copy_ = self._deformation_function.bspline
        else:
            deformation_function_copy_ = self._deformation_function.nurbs
        # Create Spline that will be used to iterate over parametric space
        ukvs = deformation_function_copy_.unique_knots
        if self._tiling_per_knot_span:
            for i_pd in range(deformation_function_copy_.para_dim):
                if self.tiling[i_pd] == 1:
                    continue
                inv_t = 1 / self.tiling[i_pd]
                new_knots = [
                        j * inv_t * (ukvs[i_pd][i] - ukvs[i_pd][i - 1])
                        for i in range(1, len(ukvs[i_pd]))
                        for j in range(1, self.tiling[i_pd])
                ]
                # insert knots in both the deformation function
                deformation_function_copy_.insert_knots(i_pd, new_knots)
            def_fun_patches = deformation_function_copy_.extract.beziers()
        else:
            raise NotImplementedError(
                    "Currently only knot-span wise insertion is implemented"
            )

        # Calculate parametric space representation for parametrized
        # microstructures
        is_parametrized = self.parametrization_function is not None
        if is_parametrized:
            para_space_dimensions = [[u[0], u[-1]] for u in ukvs]
            # Trust me @j042
            def_fun_para_space = base.Bezier(
                    degrees=[1] * deformation_function_copy_.para_dim,
                    control_points=np.array(
                            list(
                                    itertools.product(
                                            *para_space_dimensions[::-1]
                                    )
                            )
                    )[:, ::-1]
            ).bspline
            for i_pd in range(deformation_function_copy_.para_dim):
                if self.tiling[i_pd] != 1:
                    def_fun_para_space.insert_knots(
                            i_pd,
                            deformation_function_copy_.unique_knots[i_pd][1:-1]
                    )
            def_fun_para_space = def_fun_para_space.extract.beziers()

        # Determine element resolution
        element_resolutions = [
                len(c) - 1 for c in deformation_function_copy_.unique_knots
        ]

        # Start actual composition
        microstructure = []
        if is_parametrized:
            for i, (def_fun, def_fun_para) in enumerate(
                    zip(def_fun_patches, def_fun_para_space)
            ):
                # Evaluate tile parameters
                positions = def_fun_para.evaluate(
                        self._microtile.evaluation_points
                )
                tile_parameters = self._parametrization_function(positions)

                # Check if center or closing tile
                if is_closed:
                    # check index
                    index = i
                    for ipd in range(closing_faces):
                        index -= index % element_resolutions[ipd]
                        index /= element_resolutions[ipd]
                    index = index % element_resolutions[closing_faces]
                    if index == 0:
                        # Closure at minimum id
                        tile = self._microtile.closing_tile(
                                parameters=tile_parameters,
                                closure=closing_faces,
                                **kwargs
                        )
                    elif (index + 1) == element_resolutions[closing_faces]:
                        # Closure at minimum id
                        tile = self._microtile.closing_tile(
                                parameters=tile_parameters,
                                closure=-closing_faces,
                                **kwargs
                        )
                    else:
                        tile = self._microtile.create_tile(
                                parameters=tile_parameters, **kwargs
                        )
                else:
                    tile = self._microtile.create_tile(
                            parameters=tile_parameters, **kwargs
                    )

                # Perform composition
                for tile_patch in tile:
                    microstructure.append(def_fun.compose(tile_patch))
        # Not parametrized
        else:
            # Tile can be computed once (prevent to many evaluations)
            tile = self._microtile.create_tile(*kwargs)
            for def_fun in def_fun_patches:
                for t in tile:
                    microstructure.append(def_fun.compose(t))

        return microstructure

    def _sanity_check(self) -> bool:
        """
        Check all members and consistency of user data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if (
                (self.deformation_function is None)
                or (self.microtile is None) or (self.tiling is None)
        ):
            self._logd(
                    "Current information not sufficient,"
                    " awaiting further assignments"
            )
            return False
        # Check if microtile object fulfils requirements
        if not hasattr(self._microtile, "create_tile"):
            raise ValueError(
                    "Microtile class does not provide the necessary "
                    "attribute `create_tile`, that is required for "
                    "microstructure construction"
            )
        if not hasattr(self._microtile, "dim"):
            raise ValueError(
                    "Microtile class does not provide the necessary "
                    "attribute `dim`, defining the dimensionality of "
                    "the created tile"
            )

        # Check if parametric dimensions are consistent
        if not self.deformation_function.para_dim == self._microtile.dim:
            raise ValueError(
                    "Microtile dimension must match parametric dimension of "
                    "deformation function to enable composition"
            )

        # Check if tiling is consistent
        if isinstance(self.tiling, int):
            self.tiling = [self.tiling] * self.deformation_function.para_dim
        if len(self.tiling) != self.deformation_function.para_dim:
            raise ValueError(
                    "Tiling list must have one entry per parametric dimension"
                    " of the deformation function"
            )
        if self.parametrization_function is not None:
            self._logd("Checking compatibility of parametrization function")
            if not hasattr(self._microtile, "evaluation_points"):
                raise ValueError(
                        "Microtile class does not provide the necessary "
                        "attribute `evaluation_points`, that is required for"
                        " a parametrized microstructure construction"
                )
            if not hasattr(self._microtile, "parameter_space_dimension"):
                raise ValueError(
                        "Microtile class does not provide the necessary "
                        "attribute `parameter_space_dimension`, that is "
                        "required for a parametrized microstructure "
                        "construction"
                )
            result = self._parametrization_function(
                    self._microtile.evaluation_points
            )
            if not isinstance(result, tuple):
                raise ValueError(
                        "Function outline of parametrization function must be "
                        "`f(np.ndarray)->tuple`"
                )
            if not len(result) == self._microtile.parameter_space_dimension:
                raise ValueError(
                        "Return type of Parametrization function is "
                        "insufficient, check documentation of Microtile for "
                        "dimensionality"
                )
        # Complete check
        return True

    def _make_microtilable(self, microtile):
        """
        Private function that creates a Microtile object on the fly if user
        only provides (a list of) splines

        Parameters
        ----------
        microtile : spline / list<splines>
          Microtile definition of a spline
        """

        class _UserTile():

            def __init__(self, microtile):
                """
                On the fly created class of a user tile
                Parameters
                ----------
                microtile : spline , list<splines>
                """
                # Assign microtiles
                self._user_tile = []

                if not isinstance(microtile, list):
                    microtile = [microtile]

                for m in microtile:
                    if not issubclass(type(m), base.GustafSpline):
                        raise ValueError(
                                "Microtiles must be (list of) "
                                "gustaf.GustafSplines. e.g. gustaf.NURBS"
                        )
                    # Extract beziers for every non Bezier patch else this just
                    # returns itself
                    self._user_tile.extend(m.extract.beziers())
                self._dim = microtile[0].dim
                for m in microtile:
                    if m.dim != self._dim:
                        raise ValueError(
                                "Dimensions of spline lists inconsistent"
                        )

            @property
            def dim(self):
                return self._dim

            def create_tile(self, *kwargs):
                """
                Create a tile on the fly
                """
                return self._user_tile.copy()

        return _UserTile(microtile)
