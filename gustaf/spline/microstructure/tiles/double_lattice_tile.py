import numpy as np

from gustaf.spline import base
from gustaf.spline.microstructure.tiles.tilebase import TileBase


class CrossTile2D(TileBase):
    def __init__(self):
        """Simple crosstile with linear-quadratic branches and a trilinear
        center spline."""
        self._dim = 2
        self._evaluation_points = np.array(
            [
                [0.5, 0.5],
            ]
        )
        self._parameter_space_dimension = 1

    def create_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        contact_length=0.5,
        **kwargs,
    ):
        """Create a microtile based on the parameters that describe the branch
        thicknesses.

        Thickness parameters are used to describe the inner radius of the
        outward facing branches

        Parameters
        ----------
        parameters : tuple(np.array)
          only first entry is used, defines the internal radii of the
          branches
        parameter_sensitivities: list(tuple(np.ndarray))
          correlates with thickness of branches and entouring wall
        contact_length : double
          required for conformity between tiles, sets the length of the center
          block on the tiles boundary
        Returns
        -------
        microtile_list : list(splines)
        """

        if not isinstance(contact_length, float):
            raise ValueError("Invalid Type")
        if not ((contact_length > 0.0) and (contact_length < 0.5)):
            raise ValueError("Center Expansion must be in (0.,1.)")

        # set to default if nothing is given
        if parameters is None:
            self._logd("Tile request is not parametrized, setting default 0.2")
            parameters = tuple([np.ones(1) * 0.1])
        else:
            # @Lukas, bitte hier type checks!
            # shape muss(1,) sein mit werten zwischen (0, .25)
            pass

        # Check if user requests derivative splines
        if parameter_sensitivities is not None:
            # Check format
            if not (
                isinstance(parameter_sensitivities, list)
                and isinstance(parameter_sensitivities[0], tuple)
            ):
                raise TypeError(
                    "The parameter_sensitivities passed have the wrong "
                    "format. The expected format is "
                    "list(tuple(np.ndarray)), where each list entry "
                )
            # Else set number of parameters here
            n_derivatives = len(parameter_sensitivities)
        else:
            n_derivatives = 0

        derivatives = []
        splines = []
        for i_derivative in range(n_derivatives + 1):
            # Constant auxiliary values
            if i_derivative == 0:
                cl = contact_length
                pp = parameters[0][0]  # Should be the only one
                v_one_half = 0.5
                v_one = 1.0
                v_zero = 0.0
            else:
                cl = 0.0
                pp = parameter_sensitivities[i_derivative - 1][0][0]
                v_one_half = 0.0
                v_one = 0.0
                v_zero = 0.0

            # Set variables
            a01 = v_zero
            a02 = pp
            a03 = 2 * pp
            a04 = (v_one + cl) * 0.5
            a05 = v_one_half - pp
            a06 = v_one_half
            a07 = v_one_half + pp
            a08 = (v_one - cl) * 0.5
            a09 = v_one - 2 * pp
            a10 = v_one - pp
            a11 = v_one

            # Init return value
            spline_list = []

            ######################################
            # @Lukas hier die Splines definieren #
            ######################################
            # Dabei bitte ausschliesslich die Hilfsvariablen A0 bis A11
            # benutzen und keine Rechnungen durchfuehren

            # Beispiel (Spline mit nr 1 im Bild)
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1],
                    control_points=[
                        [a01, a01],
                        [a02, a02],
                        [a01, a04],
                        [a02, a03],
                    ],
                )
            )

            # Pass to output
            if i_derivative == 0:
                splines = spline_list.copy()
            else:
                derivatives.append(spline_list)

        # Return results
        if i_derivative == 0:
            return splines
        else:
            return (splines, derivatives)
