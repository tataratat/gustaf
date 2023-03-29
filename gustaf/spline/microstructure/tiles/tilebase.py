from gustaf.spline import base


class TileBase(base.GustafBase):
    """
    Base class for tile objects
    """

    def __init__(self):
        """
        Init Values to None
        """
        self._dim = None
        self._evaluation_points = None
        self._parameter_space_dimension = None

    @property
    def parameter_space_dimension(self):
        """Number of parameters per evaluation point."""
        if self._parameter_space_dimension is None:
            raise TypeError(
                "Inherited Tile-types need to provide "
                "_parameter_space_dimension, see documentation."
            )
        return self._parameter_space_dimension

    @property
    def evaluation_points(self):
        """Positions in the parametrization function to be evaluated when tile
        " "is constructed prior to composition.

        Parameters
        ----------
        None

        Returns
        -------
        evaluation_points : np.ndarray(6,3)
        """
        if self._evaluation_points is None:
            raise TypeError(
                "Inherited Tile-types need to provide _evaluation_points, see "
                "documentation."
            )
        return self._evaluation_points

    @property
    def dim(self):
        """Returns dimensionality in physical space of the Microtile.

        Parameters
        ----------
        None

        Returns
        -------
        dim : int
        """
        if self._dim is None:
            raise TypeError(
                "Inherited Tile-types need to provide _dim, see documentation."
            )
        return self._dim

    def check_params(self, params):
        """Checks if the parameters have the correct format and shape

        Parameters
        ----------
        params: tuple(np.ndarray)
            the parameters for the tile

        Returns
        -------
        True: Boolean
        """
        # check if tuple
        if not (isinstance(params, tuple)):
            raise TypeError("parameters must be a Tuple with array as entries")
        # check if the tuple has the correct number of entries
        if not (len(params) == self._parameter_space_dimension):
            raise TypeError(
                f"Size mismatch in param, expected "
                f"{self._parameter_space_dimension} "
                f"entries got {len(params)} entries"
            )
        # check if all entries in the tuple has the correct length
        if not (all([len(self._evaluation_points) == e.size for e in params])):
            raise TypeError(
                f"Mismatch amount of parameter entries, expected "
                f"{self._evaluation_points}"
            )

        return True

    def check_param_derivatives(self, derivatives):
        """Checks if all derivatives have the correct format and shape

        Parameters
        ----------
        derivatives: list(tuple(np.ndarray)
            all derivatives as list

        Returns
        -------
        True: Boolean
        """
        if derivatives is None:
            return False

        if not (isinstance(derivatives, list)):
            raise TypeError(
                f"The parameter_sensitives passed have the wrong "
                "format. The expected format is list(tuple(np.ndarray)),"
                f" found type({type(derivatives) })"
            )
        for i in derivatives:
            self.check_params(i)

        return True
