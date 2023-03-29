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

        if not (isinstance(params, np.ndarray) and params.ndim == 1):
            raise TypeError("parameters must be a array as entries")
        # check if the tuple has the correct number of entries

            # check if all entries in the tuple has the correct length
        if not ([len(self._evaluation_points) == e.size for e in params]):
            raise TypeError(
                f"Mismatch amount of parameter entries, expected "
                f"{self._evaluation_points}")

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

        if not (isinstance(derivatives, np.ndarray) and derivatives.ndim == 2):
            raise TypeError(
                f"The parameter_sensitives passed have the wrong "
                "format. The expected format is list(tuple(np.ndarray)),"
                f" found type({type(derivatives)})"
            )
        for i in derivatives:
            self.check_params(i)

        return True
