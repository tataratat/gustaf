import numpy as np

from gustaf.spline import base
from gustaf.spline.microstructure.tiles.tilebase import TileBase


class Armadillo(TileBase):
    def __init__(self):
        """Simple tile - looks like a nut"""
        self._dim = 3
        self._evaluation_points = np.array(
            [
                [0.0, 0.5, 0.5],
                [1.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 1.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 1.0],
            ]
        )
        self._parameter_space_dimension = 1

    def closing_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        contact_length=0.2,
        closure=None,
        **kwargs,
    ):
        """Create a closing tile to match with closed surface.

        Parameters
        ----------
        parameters: tuple(np.ndarray)
            thickness of the wall
        parameter_sensitivities: list(tuple(np.ndarray))
          Describes the parameter sensitivities with respect to some design
          variable. In case the design variables directly apply to the
          parameter itself, they evaluate as delta_ij
        closure : int
          parametric dimension that needs to be closed. Positive values mean
          that minimum parametric dimension is requested. That means,
          i.e. -2 closes the tile at maximum z-coordinate.
          (must currently be either -2 or 2)
        contact_length: float
          the length of the wall that contacts the other microstructure

        Results
        -------
        spline_list : list
        """
        if closure is None:
            raise ValueError("No closing direction given")

        if not isinstance(contact_length, float):
            raise ValueError("Invalid Type for radius")

        if not ((contact_length > 0) and (contact_length < 0.99)):
            raise ValueError("The length of a side must be in (0.01, 0.99)")

        if parameters is None:
            self._logd("Setting parameters to default values (0.2)")
            parameters = tuple([np.ones(1) * 0.2])

        v_h_void = parameters[0][0]
        if not ((v_h_void > 0.01) and (v_h_void < 0.5)):
            raise ValueError(
                "The thickness of the wall must be in (0.01 and 0.49)"
            )

        spline_list = []
        v_zero = 0.0
        v_one_half = 0.5
        v_one = 1.0
        v_outer_c_h = contact_length * 0.5
        v_half_contact_length = contact_length * 0.5
        v_inner_c_h = contact_length * parameters[0][0]

        if closure == "x_min":
            # set points:
            right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            connection_front_right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            front = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            connection_back_left = np.array(
                [
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_one,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_zero,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            left = np.array(
                [
                    [
                        v_zero,
                        v_one,
                        v_one,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_one,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_zero,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_zero,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_front_left = np.array(
                [
                    [
                        v_zero,
                        v_one,
                        v_one,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_zero,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            back = np.array(
                [
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_right = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            bottom = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            top = np.array(
                [
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                ]
            )

            connection_front_bottom = np.array(
                [
                    [
                        v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_front_top = np.array(
                [
                    [
                        v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                ]
            )

            connection_back_bottom = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_top = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_half_contact_length + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_top_right = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one,
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one,
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_top_left = np.array(
                [
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_bottom_left = np.array(
                [
                    [
                        v_zero,
                        v_one,
                        v_zero,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_bottom_right = np.array(
                [
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

        elif closure == "x_max":
            # set points:
            right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_one,
                        v_one,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_one,
                        v_zero,
                    ],
                ]
            )

            connection_front_right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_one,
                        v_one,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_one,
                        v_zero,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            front = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            connection_back_left = np.array(
                [
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            left = np.array(
                [
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_front_left = np.array(
                [
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            back = np.array(
                [
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_right = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_zero,
                    ],
                ]
            )

            bottom = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            top = np.array(
                [
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                ]
            )

            connection_front_bottom = np.array(
                [
                    [
                        v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_front_top = np.array(
                [
                    [
                        v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                ]
            )

            connection_back_bottom = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_top = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_half_contact_length + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_top_right = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one,
                        v_one,
                        v_one,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_top_left = np.array(
                [
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_zero,
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_bottom_left = np.array(
                [
                    [
                        v_zero,
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_zero,
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_bottom_right = np.array(
                [
                    [
                        v_one,
                        v_one,
                        v_zero,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

        elif closure == "y_min":
            # set points:
            right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            connection_front_right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            front = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            connection_back_left = np.array(
                [
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_one,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_zero,
                    ],
                ]
            )

            left = np.array(
                [
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_front_left = np.array(
                [
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            back = np.array(
                [
                    [
                        v_one,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_one,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_zero,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_right = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            bottom = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            top = np.array(
                [
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                ]
            )

            connection_front_bottom = np.array(
                [
                    [
                        v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_front_top = np.array(
                [
                    [
                        v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                ]
            )

            connection_back_bottom = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_top = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_top_right = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one,
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one,
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_top_left = np.array(
                [
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_zero,
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_bottom_left = np.array(
                [
                    [
                        v_zero,
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_zero,
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_bottom_right = np.array(
                [
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

        elif closure == "y_max":
            # set points:
            # set points:
            right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            connection_front_right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_one,
                        v_one,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_one,
                        v_zero,
                    ],
                ]
            )

            front = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_one,
                        v_one,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_one,
                        v_zero,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_one,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_zero,
                    ],
                ]
            )

            connection_back_left = np.array(
                [
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            left = np.array(
                [
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_top_left = np.array(
                [
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_one,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_zero,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            back = np.array(
                [
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_right = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            bottom = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            top = np.array(
                [
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                ]
            )

            connection_front_bottom = np.array(
                [
                    [
                        v_one,
                        v_one,
                        v_zero,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_front_top = np.array(
                [
                    [
                        v_one,
                        v_one,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_one,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                ]
            )

            connection_back_bottom = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_top = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_half_contact_length + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_top_right = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one,
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one,
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_front_left = np.array(
                [
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_zero,
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_bottom_left = np.array(
                [
                    [
                        v_zero,
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_zero,
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_bottom_right = np.array(
                [
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

        elif closure == "z_max":
            # set points:
            right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            connection_front_right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            front = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            connection_back_left = np.array(
                [
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            left = np.array(
                [
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_front_left = np.array(
                [
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            back = np.array(
                [
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_right = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            bottom = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            top = np.array(
                [
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one,
                        v_one,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_one,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_one,
                    ],
                ]
            )

            connection_front_bottom = np.array(
                [
                    [
                        v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_front_top = np.array(
                [
                    [
                        v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_one,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                ]
            )

            connection_back_bottom = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_top = np.array(
                [
                    [
                        v_one,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_half_contact_length + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_top_right = np.array(
                [
                    [
                        v_one,
                        v_one,
                        v_one,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one,
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one,
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_top_left = np.array(
                [
                    [
                        v_zero,
                        v_one,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_zero,
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_bottom_left = np.array(
                [
                    [
                        v_zero,
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_zero,
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_bottom_right = np.array(
                [
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

        elif closure == "z_min":
            # set points:
            # set points:
            right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            connection_front_right = np.array(
                [
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            front = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            connection_back_left = np.array(
                [
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            left = np.array(
                [
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_front_left = np.array(
                [
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_zero,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            back = np.array(
                [
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_right = np.array(
                [
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        -v_h_void + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_outer_c_h + v_one_half,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                ]
            )

            bottom = np.array(
                [
                    [
                        v_one,
                        v_one,
                        v_zero,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_zero,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            top = np.array(
                [
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                ]
            )

            connection_front_bottom = np.array(
                [
                    [
                        v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one,
                        v_one,
                        v_zero,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_zero,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_front_top = np.array(
                [
                    [
                        v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_one,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                ]
            )

            connection_back_bottom = np.array(
                [
                    [
                        v_one,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_zero,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                    ],
                ]
            )

            connection_back_top = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_half_contact_length + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        -v_half_contact_length + v_one_half,
                        v_zero,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_top_right = np.array(
                [
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one,
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one,
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half + v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_top_left = np.array(
                [
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                        v_one,
                    ],
                    [
                        v_zero,
                        v_one_half + v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_zero,
                        v_one_half - v_half_contact_length,
                        v_one_half + v_half_contact_length,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_h_void,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                    ],
                ]
            )

            connection_bottom_left = np.array(
                [
                    [
                        v_zero,
                        v_one_half + v_half_contact_length,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_zero,
                        v_one_half - v_half_contact_length,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_zero,
                        v_one,
                        v_zero,
                    ],
                    [
                        v_zero,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_h_void,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half - v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

            connection_bottom_right = np.array(
                [
                    [
                        v_one,
                        v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_one,
                        -v_outer_c_h + v_one_half,
                        v_one_half - v_half_contact_length,
                    ],
                    [
                        v_h_void + v_one_half,
                        v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_h_void + v_one_half,
                        -v_inner_c_h + v_one_half,
                        v_one_half - v_inner_c_h,
                    ],
                    [
                        v_one,
                        v_one,
                        v_zero,
                    ],
                    [
                        v_one,
                        v_zero,
                        v_zero,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half + v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                    [
                        v_one_half + v_inner_c_h,
                        v_one_half - v_inner_c_h,
                        v_one_half - v_h_void,
                    ],
                ]
            )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_bottom_left
            )
        )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_bottom_right
            )
        )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_back_bottom
            )
        )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_front_bottom
            )
        )
        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=connection_front_top)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=connection_back_top)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=connection_top_right)
        )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_front_left
            )
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=right)
        )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_front_right
            )
        )

        spline_list.append(base.Bezier(degrees=[1, 1, 1], control_points=back))

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=connection_back_left)
        )

        spline_list.append(base.Bezier(degrees=[1, 1, 1], control_points=left))

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=connection_top_left)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=front)
        )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_back_right
            )
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=bottom)
        )

        spline_list.append(base.Bezier(degrees=[1, 1, 1], control_points=top))

        return spline_list

    def create_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        contact_length=0.3,
        **kwargs,
    ):
        """Create a microtile based on the parameters that describe the wall
        thicknesses.

        Thickness parameters are used to describe the inner radius of the
        outward facing branches

        Parameters
        ----------
        parameters : tuple(np.array)
          only first entry is used, defines the thickness of the
          wall
        parameter_sensitivities: list(tuple(np.ndarray))
          Describes the parameter sensitivities with respect to some design
          variable. In case the design variables directly apply to the
          parameter itself, they evaluate as delta_ij
        contact_length : float
            the length of the wall that contacts the other microstructure
        Returns
        -------
        microtile_list : list(splines)
        """

        if not isinstance(contact_length, float):
            raise ValueError("Invalid Type for radius")

        if not ((contact_length > 0) and (contact_length < 0.99)):
            raise ValueError("The length of a side must be in (0.01, 0.99)")

        if parameters is None:
            self._logd("Setting parameters to default values (0.2)")
            parameters = tuple([np.ones(1) * 0.2])

        v_h_void = parameters[0][0]
        if not ((v_h_void > 0.01) and (v_h_void < 0.5)):
            raise ValueError(
                "The thickness of the wall must be in (0.01 and 0.49)"
            )

        v_zero = 0.0
        v_one_half = 0.5
        v_one = 1.0
        v_outer_c_h = contact_length * 0.5
        v_half_contact_length = contact_length * 0.5
        v_inner_c_h = contact_length * parameters[0][0]

        spline_list = []

        # set points:
        right = np.array(
            [
                [
                    v_h_void + v_one_half,
                    -v_inner_c_h + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_one,
                    -v_outer_c_h + v_one_half,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_h_void + v_one_half,
                    -v_inner_c_h + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_one,
                    -v_outer_c_h + v_one_half,
                    v_one_half - v_half_contact_length,
                ],
                [
                    v_h_void + v_one_half,
                    v_inner_c_h + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_one,
                    v_outer_c_h + v_one_half,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_h_void + v_one_half,
                    v_inner_c_h + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_one,
                    v_outer_c_h + v_one_half,
                    v_one_half - v_half_contact_length,
                ],
            ]
        )

        connection_front_right = np.array(
            [
                [
                    v_h_void + v_one_half,
                    v_inner_c_h + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_one,
                    v_outer_c_h + v_one_half,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_h_void + v_one_half,
                    v_inner_c_h + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_one,
                    v_outer_c_h + v_one_half,
                    v_one_half - v_half_contact_length,
                ],
                [
                    v_inner_c_h + v_one_half,
                    v_h_void + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_outer_c_h + v_one_half,
                    v_one,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_inner_c_h + v_one_half,
                    v_h_void + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_outer_c_h + v_one_half,
                    v_one,
                    v_one_half - v_half_contact_length,
                ],
            ]
        )

        front = np.array(
            [
                [
                    v_inner_c_h + v_one_half,
                    v_h_void + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_outer_c_h + v_one_half,
                    v_one,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_inner_c_h + v_one_half,
                    v_h_void + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_outer_c_h + v_one_half,
                    v_one,
                    v_one_half - v_half_contact_length,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    v_h_void + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    -v_outer_c_h + v_one_half,
                    v_one,
                    v_one_half + v_half_contact_length,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    v_h_void + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    -v_outer_c_h + v_one_half,
                    v_one,
                    v_one_half - v_half_contact_length,
                ],
            ]
        )

        connection_back_left = np.array(
            [
                [
                    -v_h_void + v_one_half,
                    -v_inner_c_h + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_zero,
                    -v_outer_c_h + v_one_half,
                    v_one_half + v_half_contact_length,
                ],
                [
                    -v_h_void + v_one_half,
                    -v_inner_c_h + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_zero,
                    -v_outer_c_h + v_one_half,
                    v_one_half - v_half_contact_length,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    -v_h_void + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    -v_outer_c_h + v_one_half,
                    v_zero,
                    v_one_half + v_half_contact_length,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    -v_h_void + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    -v_outer_c_h + v_one_half,
                    v_zero,
                    v_one_half - v_half_contact_length,
                ],
            ]
        )

        left = np.array(
            [
                [
                    v_zero,
                    -v_outer_c_h + v_one_half,
                    v_one_half + v_half_contact_length,
                ],
                [
                    -v_h_void + v_one_half,
                    -v_inner_c_h + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_zero,
                    -v_outer_c_h + v_one_half,
                    v_one_half - v_half_contact_length,
                ],
                [
                    -v_h_void + v_one_half,
                    -v_inner_c_h + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_zero,
                    v_outer_c_h + v_one_half,
                    v_one_half + v_half_contact_length,
                ],
                [
                    -v_h_void + v_one_half,
                    v_inner_c_h + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_zero,
                    v_outer_c_h + v_one_half,
                    v_one_half - v_half_contact_length,
                ],
                [
                    -v_h_void + v_one_half,
                    v_inner_c_h + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
            ]
        )

        connection_front_left = np.array(
            [
                [
                    v_zero,
                    v_outer_c_h + v_one_half,
                    v_one_half + v_half_contact_length,
                ],
                [
                    -v_h_void + v_one_half,
                    v_inner_c_h + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_zero,
                    v_outer_c_h + v_one_half,
                    v_one_half - v_half_contact_length,
                ],
                [
                    -v_h_void + v_one_half,
                    v_inner_c_h + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    -v_outer_c_h + v_one_half,
                    v_one,
                    v_one_half + v_half_contact_length,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    v_h_void + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    -v_outer_c_h + v_one_half,
                    v_one,
                    v_one_half - v_half_contact_length,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    v_h_void + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
            ]
        )

        back = np.array(
            [
                [
                    v_outer_c_h + v_one_half,
                    v_zero,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_inner_c_h + v_one_half,
                    -v_h_void + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_outer_c_h + v_one_half,
                    v_zero,
                    v_one_half - v_half_contact_length,
                ],
                [
                    v_inner_c_h + v_one_half,
                    -v_h_void + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    -v_outer_c_h + v_one_half,
                    v_zero,
                    v_one_half + v_half_contact_length,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    -v_h_void + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    -v_outer_c_h + v_one_half,
                    v_zero,
                    v_one_half - v_half_contact_length,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    -v_h_void + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
            ]
        )

        connection_back_right = np.array(
            [
                [
                    v_inner_c_h + v_one_half,
                    -v_h_void + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_outer_c_h + v_one_half,
                    v_zero,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_inner_c_h + v_one_half,
                    -v_h_void + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_outer_c_h + v_one_half,
                    v_zero,
                    v_one_half - v_half_contact_length,
                ],
                [
                    v_h_void + v_one_half,
                    -v_inner_c_h + v_one_half,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_one,
                    -v_outer_c_h + v_one_half,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_h_void + v_one_half,
                    -v_inner_c_h + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_one,
                    -v_outer_c_h + v_one_half,
                    v_one_half - v_half_contact_length,
                ],
            ]
        )

        bottom = np.array(
            [
                [
                    v_one_half + v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_zero,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_zero,
                ],
                [
                    v_one_half + v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_zero,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_zero,
                ],
                [
                    v_one_half + v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half - v_h_void,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half - v_h_void,
                ],
                [
                    v_one_half + v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half - v_h_void,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half - v_h_void,
                ],
            ]
        )

        top = np.array(
            [
                [
                    v_one_half + v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half + v_h_void,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half + v_h_void,
                ],
                [
                    v_one_half + v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half + v_h_void,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half + v_h_void,
                ],
                [
                    v_one_half + v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_one,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_one,
                ],
                [
                    v_one_half + v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_one,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_one,
                ],
            ]
        )

        connection_front_bottom = np.array(
            [
                [
                    v_half_contact_length + v_one_half,
                    v_one,
                    v_one_half - v_half_contact_length,
                ],
                [
                    -v_half_contact_length + v_one_half,
                    v_one,
                    v_one_half - v_half_contact_length,
                ],
                [
                    v_one_half + v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_zero,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_zero,
                ],
                [
                    v_inner_c_h + v_one_half,
                    v_one_half + v_h_void,
                    v_one_half - v_inner_c_h,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    v_one_half + v_h_void,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_one_half + v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half - v_h_void,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half - v_h_void,
                ],
            ]
        )

        connection_front_top = np.array(
            [
                [
                    v_half_contact_length + v_one_half,
                    v_one,
                    v_one_half + v_half_contact_length,
                ],
                [
                    -v_half_contact_length + v_one_half,
                    v_one,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_inner_c_h + v_one_half,
                    v_one_half + v_h_void,
                    v_one_half + v_inner_c_h,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    v_one_half + v_h_void,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_one_half + v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_one,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_one,
                ],
                [
                    v_one_half + v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half + v_h_void,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half + v_h_void,
                ],
            ]
        )

        connection_back_bottom = np.array(
            [
                [
                    v_one_half + v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_zero,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_zero,
                ],
                [
                    v_one_half + v_half_contact_length,
                    v_zero,
                    v_one_half - v_half_contact_length,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_zero,
                    v_one_half - v_half_contact_length,
                ],
                [
                    v_one_half + v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half - v_h_void,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half - v_h_void,
                ],
                [
                    v_inner_c_h + v_one_half,
                    v_one_half - v_h_void,
                    v_one_half - v_inner_c_h,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    v_one_half - v_h_void,
                    v_one_half - v_inner_c_h,
                ],
            ]
        )

        connection_back_top = np.array(
            [
                [
                    v_one_half + v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_one,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_one,
                ],
                [
                    v_one_half + v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half + v_h_void,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half + v_h_void,
                ],
                [
                    v_half_contact_length + v_one_half,
                    v_zero,
                    v_one_half + v_half_contact_length,
                ],
                [
                    -v_half_contact_length + v_one_half,
                    v_zero,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_inner_c_h + v_one_half,
                    v_one_half - v_h_void,
                    v_one_half + v_inner_c_h,
                ],
                [
                    -v_inner_c_h + v_one_half,
                    v_one_half - v_h_void,
                    v_one_half + v_inner_c_h,
                ],
            ]
        )

        connection_top_right = np.array(
            [
                [
                    v_one_half + v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_one,
                ],
                [
                    v_one_half + v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_one,
                ],
                [
                    v_one_half + v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half + v_h_void,
                ],
                [
                    v_one_half + v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half + v_h_void,
                ],
                [
                    v_one,
                    v_one_half + v_half_contact_length,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_one,
                    v_one_half - v_half_contact_length,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_one_half + v_h_void,
                    v_one_half + v_inner_c_h,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_one_half + v_h_void,
                    v_one_half - v_inner_c_h,
                    v_one_half + v_inner_c_h,
                ],
            ]
        )

        connection_top_left = np.array(
            [
                [
                    v_one_half - v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_one,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_one,
                ],
                [
                    v_zero,
                    v_one_half + v_half_contact_length,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_zero,
                    v_one_half - v_half_contact_length,
                    v_one_half + v_half_contact_length,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half + v_h_void,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half + v_h_void,
                ],
                [
                    v_one_half - v_h_void,
                    v_one_half + v_inner_c_h,
                    v_one_half + v_inner_c_h,
                ],
                [
                    v_one_half - v_h_void,
                    v_one_half - v_inner_c_h,
                    v_one_half + v_inner_c_h,
                ],
            ]
        )

        connection_bottom_left = np.array(
            [
                [
                    v_zero,
                    v_one_half + v_half_contact_length,
                    v_one_half - v_half_contact_length,
                ],
                [
                    v_zero,
                    v_one_half - v_half_contact_length,
                    v_one_half - v_half_contact_length,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_zero,
                ],
                [
                    v_one_half - v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_zero,
                ],
                [
                    v_one_half - v_h_void,
                    v_one_half + v_inner_c_h,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_one_half - v_h_void,
                    v_one_half - v_inner_c_h,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half - v_h_void,
                ],
                [
                    v_one_half - v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half - v_h_void,
                ],
            ]
        )

        connection_bottom_right = np.array(
            [
                [
                    v_one,
                    v_outer_c_h + v_one_half,
                    v_one_half - v_half_contact_length,
                ],
                [
                    v_one,
                    -v_outer_c_h + v_one_half,
                    v_one_half - v_half_contact_length,
                ],
                [
                    v_h_void + v_one_half,
                    v_inner_c_h + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_h_void + v_one_half,
                    -v_inner_c_h + v_one_half,
                    v_one_half - v_inner_c_h,
                ],
                [
                    v_one_half + v_half_contact_length,
                    v_one_half + v_half_contact_length,
                    v_zero,
                ],
                [
                    v_one_half + v_half_contact_length,
                    v_one_half - v_half_contact_length,
                    v_zero,
                ],
                [
                    v_one_half + v_inner_c_h,
                    v_one_half + v_inner_c_h,
                    v_one_half - v_h_void,
                ],
                [
                    v_one_half + v_inner_c_h,
                    v_one_half - v_inner_c_h,
                    v_one_half - v_h_void,
                ],
            ]
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=right)
        )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_front_right
            )
        )

        spline_list.append(base.Bezier(degrees=[1, 1, 1], control_points=back))

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=connection_back_left)
        )

        spline_list.append(base.Bezier(degrees=[1, 1, 1], control_points=left))

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_front_left
            )
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=front)
        )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_back_right
            )
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=bottom)
        )

        spline_list.append(base.Bezier(degrees=[1, 1, 1], control_points=top))

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=connection_front_top)
        )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_front_bottom
            )
        )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_back_bottom
            )
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=connection_back_top)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=connection_top_right)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=connection_top_left)
        )

        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_bottom_left
            )
        )
        spline_list.append(
            base.Bezier(
                degrees=[1, 1, 1], control_points=connection_bottom_right
            )
        )

        return spline_list