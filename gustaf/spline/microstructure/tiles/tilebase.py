import numpy as np

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
        # check if tuple
        if not (type(params) == tuple):
            raise TypeError("parameters must be a Tuple with array as entries")
        # check if the tuple has the correct number of entries
        if not (len(params) == self._parameter_space_dimension):
            raise TypeError("There unexpected entries in the parameter tuple")
        # check if all entries in the tuple has the correct length
        if not (all([len(self._evaluation_points) == len(e) for e in params])):
            raise TypeError("Wrong number of evaluation points")

        return

    def check_param_values(self, params, value_range):
        if not (
            np.all(params[0] > value_range[0])
            and np.all(params[0] < value_range[1])
        ):
            raise ValueError(
                f"Value of a parameter is out of range ("
                f"{value_range[0]}, {value_range[1]})"
            )
