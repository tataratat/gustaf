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
                [0.0, 0.5],
                [1.0, 0.5],
                [0.5, 0.0],
                [0.5, 1.0],
            ]
        )
        self._parameter_space_dimension = 1

    def closing_tile(
            self,
            parameters=None,
            parameter_sensitivities=None,
            closure=None,
            boundary_width=0.1,
            filling_height=0.5,
            **kwargs,
    ):
        """Create a closing tile to match with closed surface.

        Parameters
        ----------
        parameters : tuple(np.ndarray)
          radii of fitting cylinder at evaluation points
        parameter_sensitivities: list(tuple(np.ndarray))
          Describes the parameter sensitivities with respect to some design
          variable. In case the design variables directly apply to the
          parameter itself, they evaluate as delta_ij
        closure : int
          parametric dimension that needs to be closed. Positiv values mean
          that minimum parametric dimension is requested. That means,
          i.e. -2 closes the tile at maximum z-coordinate.
          (must currently be either -2 or 2)
        boundary_width : float
          with of the boundary surronding branch
        filling_height : float
          portion of the height that is filled in parametric domain

        Returns
        -------
        list_of_splines : list
        """
        # Check parameters
        if closure is None:
            raise ValueError("No closing direction given")

        if parameters is None:
            self._logd("Tile request is not parametrized, setting default 0.2")
            parameters = tuple([np.ones(6) * 0.2])
        if not (np.all(parameters[0] > 0) and np.all(parameters[0] < .5)):
            raise ValueError("Thickness out of range (0, .5)")

        if not (0. < float(boundary_width) < .5):
            raise ValueError("Boundary Width is out of range")

        if not (0. < float(filling_height) < 1.):
            raise ValueError("Filling must  be in (0,1)")

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
            n_derivatives = len(parameter_sensitivities)
        else:
            n_derivatives = 0

        derivatives = []
        splines = []
        for i_derivative in range(n_derivatives + 1):
            # Constant auxiliary values
            if i_derivative == 0:
                fill_height = filling_height
                bound_width = boundary_width
                inv_bound_width = 1. - bound_width
                inv_fill_height = 1. - fill_height
                ctps_mid_height_top = (1 + fill_height) * .5
                ctps_mid_height_bottom = 1. - ctps_mid_height_top
                v_one_half = .5
                v_one = 1.
                v_zero = 0.
                parameters = parameters[0]
            else:
                # Set constant values to zero for derivatives
                fill_height = 0.
                bound_width = 0.
                inv_bound_width = 0.
                inv_fill_height = 0.
                ctps_mid_height_top = 0.
                ctps_mid_height_bottom = 0.
                v_one_half = 0.
                v_one = 0.
                v_zero = 0.
                parameters = parameter_sensitivities[i_derivative - 1][0]

            spline_list = []
            if closure == "x_min":
                # Minimum x position
                branch_thickness = parameters[1]

                block0_ctps = np.array(
                        [
                                [v_zero, v_zero],
                                [fill_height, v_zero],
                                [v_zero, bound_width],
                                [fill_height, bound_width],
                        ]
                )

                block1_ctps = np.array(
                        [
                                [v_zero, bound_width],
                                [fill_height, bound_width],
                                [v_zero, inv_bound_width],
                                [fill_height, inv_bound_width],
                        ]
                )

                block2_ctps = np.array(
                        [
                                [v_zero, inv_bound_width],
                                [fill_height, inv_bound_width],
                                [v_zero, v_one], [fill_height, v_one]
                        ]
                )

                branch_ctps = np.array(
                        [
                                [fill_height, bound_width],
                                [
                                        ctps_mid_height_top,
                                        v_one_half - branch_thickness
                                ], [v_one, v_one_half - branch_thickness],
                                [fill_height, inv_bound_width],
                                [
                                        ctps_mid_height_top,
                                        v_one_half + branch_thickness
                                ], [v_one, v_one_half + branch_thickness]
                        ]
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block0_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block1_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block2_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[2, 1], control_points=branch_ctps
                        )
                )
            elif closure == "x_max":
                # Maximum x position
                branch_thickness = parameters[0]

                block0_ctps = np.array(
                        [
                                [inv_fill_height, v_zero],
                                [v_one, v_zero],
                                [inv_fill_height, bound_width],
                                [v_one, bound_width],
                        ]
                )

                block1_ctps = np.array(
                        [
                                [inv_fill_height, bound_width],
                                [v_one, bound_width],
                                [inv_fill_height, inv_bound_width],
                                [v_one, inv_bound_width],
                        ]
                )

                block2_ctps = np.array(
                        [
                                [inv_fill_height, inv_bound_width],
                                [v_one, inv_bound_width],
                                [inv_fill_height, v_one],
                                [v_one, v_one],
                        ]
                )

                branch_ctps = np.array(
                        [
                                [0, v_one_half - branch_thickness],
                                [
                                        ctps_mid_height_bottom,
                                        v_one_half - branch_thickness
                                ],
                                [inv_fill_height, bound_width],
                                [v_zero, v_one_half + branch_thickness],
                                [
                                        ctps_mid_height_bottom,
                                        v_one_half + branch_thickness
                                ],
                                [inv_fill_height, inv_bound_width],
                        ]
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block0_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block1_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block2_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[2, 1], control_points=branch_ctps
                        )
                )
            elif closure == "y_min":
                # Minimum y position
                branch_thickness = parameters[3]

                block0_ctps = np.array(
                        [
                                [v_zero, v_zero],
                                [bound_width, v_zero],
                                [v_zero, fill_height],
                                [bound_width, fill_height],
                        ]
                )

                block1_ctps = np.array(
                        [
                                [bound_width, v_zero],
                                [inv_bound_width, v_zero],
                                [bound_width, fill_height],
                                [inv_bound_width, fill_height],
                        ]
                )

                block2_ctps = np.array(
                        [
                                [inv_bound_width, v_zero],
                                [v_one, v_zero],
                                [inv_bound_width, fill_height],
                                [v_one, fill_height],
                        ]
                )

                branch_ctps = np.array(
                        [
                                [bound_width, fill_height],
                                [inv_bound_width, fill_height],
                                [
                                        v_one_half - branch_thickness,
                                        ctps_mid_height_top
                                ],
                                [
                                        v_one_half + branch_thickness,
                                        ctps_mid_height_top
                                ],
                                [v_one_half - branch_thickness, v_one],
                                [v_one_half + branch_thickness, v_one],
                        ]
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block0_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block1_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block2_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 2], control_points=branch_ctps
                        )
                )
            elif closure == "y_max":
                # Maximum y position
                branch_thickness = parameters[2]

                block0_ctps = np.array(
                        [
                                [v_zero, inv_fill_height],
                                [bound_width, inv_fill_height],
                                [v_zero, v_one],
                                [bound_width, v_one],
                        ]
                )

                block1_ctps = np.array(
                        [
                                [bound_width, inv_fill_height],
                                [inv_bound_width, inv_fill_height],
                                [bound_width, v_one],
                                [inv_bound_width, v_one],
                        ]
                )

                block2_ctps = np.array(
                        [
                                [inv_bound_width, inv_fill_height],
                                [v_one, inv_fill_height],
                                [inv_bound_width, v_one],
                                [v_one, v_one],
                        ]
                )

                branch_ctps = np.array(
                        [
                                [v_one_half - branch_thickness, v_zero],
                                [v_one_half + branch_thickness, v_zero],
                                [
                                        v_one_half - branch_thickness,
                                        ctps_mid_height_bottom
                                ],
                                [
                                        v_one_half + branch_thickness,
                                        ctps_mid_height_bottom
                                ],
                                [bound_width, inv_fill_height],
                                [inv_bound_width, inv_fill_height],
                        ]
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block0_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block1_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 1], control_points=block2_ctps
                        )
                )

                spline_list.append(
                        base.Bezier(
                                degrees=[1, 2], control_points=branch_ctps
                        )
                )
            else:
                raise NotImplementedError(
                        "Requested closing dimension is not supported"
                )

            if i_derivative == 0:
                splines = spline_list.copy()
            else:
                derivatives.append(spline_list)
        # Return results
        if i_derivative == 0:
            return splines
        else:
            return (splines, derivatives)

    def create_tile(
            self,
            parameters=None,
            parameter_sensitivities=None,
            center_expansion=1.,
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
          Describes the parameter sensitivities with respect to some design
          variable. In case the design variables directly apply to the
          parameter itself, they evaluate as delta_ij
        center_expansion : float
            thickness of center is expanded by a factor
        Returns
        -------
        microtile_list : list(splines)
        """

        if not isinstance(center_expansion, float):
            raise ValueError("Invalid Type")
        if not ((center_expansion > .5) and (center_expansion < 1.5)):
            raise ValueError("Center Expansion must be in (.5,1.5)")
        max_radius = min(.5, (.5 / center_expansion))
        # set to default if nothing is given
        if parameters is None:
            self._logd("Setting branch thickness to default 0.2")
            parameters = tuple([np.ones(4) * 0.2])
        for radius in parameters[0].tolist():
            if not isinstance(radius, float):
                raise ValueError("Invalid type")
            if not (radius > 0 and radius < max_radius):
                raise ValueError(
                        f"Radii must be in (0,{max_radius}) for "
                        f"center_expansion {center_expansion}"
                )

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
            n_derivatives = len(parameter_sensitivities)
        else:
            n_derivatives = 0

        derivatives = []
        splines = []
        for i_derivative in range(n_derivatives + 1):
            # Constant auxiliary values
            if i_derivative == 0:
                [x_min_r, x_max_r, y_min_r, y_max_r] = parameters[0].tolist()
                parameters = parameters[0]
                v_one_half = .5
                # center radius
                center_r = (
                        x_min_r + x_max_r + y_min_r + y_max_r
                ) / 4. * center_expansion
                hd_center = 0.5 * (0.5 + center_r)
            else:
                [x_min_r, x_max_r, y_min_r, y_max_r
                 ] = parameter_sensitivities[i_derivative - 1][0].tolist()
                v_one_half = 0.
                # center radius
                center_r = (
                        x_min_r + x_max_r + y_min_r + y_max_r
                ) / 4. * center_expansion
                hd_center = 0.5 * center_r

            # Init return value
            spline_list = []

            # Create the center-tile
            center_points = np.array(
                    [
                            [-center_r, -center_r], [center_r, -center_r],
                            [-center_r, center_r], [center_r, center_r]
                    ]
            ) + np.array([v_one_half, v_one_half])

            y_min_ctps = np.array(
                    [
                            [-y_min_r, -v_one_half], [y_min_r, -v_one_half],
                            [-y_min_r, -hd_center], [y_min_r, -hd_center],
                            [-center_r, -center_r], [center_r, -center_r]
                    ]
            ) + np.array([v_one_half, v_one_half])

            y_max_ctps = np.array(
                    [
                            [-center_r, center_r],
                            [center_r, center_r],
                            [-y_max_r, hd_center],
                            [y_max_r, hd_center],
                            [-y_max_r, v_one_half],
                            [y_max_r, v_one_half],
                    ]
            ) + np.array([v_one_half, v_one_half])

            x_min_ctps = np.array(
                    [
                            [-v_one_half, -x_min_r], [-hd_center, -x_min_r],
                            [-center_r, -center_r], [-v_one_half, x_min_r],
                            [-hd_center, x_min_r], [-center_r, center_r]
                    ]
            ) + np.array([v_one_half, v_one_half])

            x_max_ctps = np.array(
                    [
                            [center_r, -center_r],
                            [hd_center, -x_max_r],
                            [v_one_half, -x_max_r],
                            [center_r, center_r],
                            [hd_center, x_max_r],
                            [v_one_half, x_max_r],
                    ]
            ) + np.array([v_one_half, v_one_half])

            spline_list.append(
                    base.Bezier(degrees=[1, 1], control_points=center_points)
            )

            spline_list.append(
                    base.Bezier(degrees=[2, 1], control_points=x_min_ctps)
            )

            spline_list.append(
                    base.Bezier(degrees=[2, 1], control_points=x_max_ctps)
            )

            spline_list.append(
                    base.Bezier(degrees=[1, 2], control_points=y_min_ctps)
            )

            spline_list.append(
                    base.Bezier(degrees=[1, 2], control_points=y_max_ctps)
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
