import numpy as np

from gustaf.spline import base
from gustaf.spline.microstructure.tiles.tilebase import TileBase


class Cube3D(TileBase):
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
                [0.5, 0.5, 0.5],
            ]
        )
        self._n_info_per_eval_point = 1

    def create_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        contact_length=0.1,
        **kwargs
    ):
        """Create a microtile based on the parameters that describe the strut
        thicknesses.


        Parameters
        ----------
        parameters :
          specifies the hole-thickness on the tile interfaces. Last parameter
          is used for center dimensions
        parameter_sensitivities: np.ndarray
          Describes the parameter sensitivities with respect to some design
          variable. In case the design variables directly apply to the
          parameter itself, they evaluate to delta_ij
        contact_length : float
            the length of the wall that contacts the other microstructure

        Returns
        -------
        microtile_list : list(splines)
        """

        if parameters is None:
            self._logd("Setting parameters to default values (0.2)")
            parameters = (
                np.ones(
                    (len(self._evaluation_points), self._n_info_per_eval_point)
                ).reshape(-1, 1)
                * 0.1
            )

        self.check_params(parameters)

        v_h_void = parameters[0, 0]
        if not ((v_h_void > 0.01) and (v_h_void < 0.5)):
            raise ValueError(
                "The thickness of the wall must be in (0.01 and 0.49)"
            )

        if self.check_param_derivatives(parameter_sensitivities):
            n_derivatives = parameter_sensitivities.shape[2]
        else:
            n_derivatives = 0

        derivatives = []
        splines = []

        for i_derivative in range(n_derivatives + 1):
            spline_list = []
            # Constant auxiliary values
            if i_derivative == 0:
                v_zero = 0.0
                v_one = 1.0
                [
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    z_min,
                    z_max,
                    center,
                ] = parameters[:, 0].flatten()
            else:
                v_zero = 0.0
                v_one = 0.0
                [
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    z_min,
                    z_max,
                    center,
                ] = parameter_sensitivities[:, 0, i_derivative - 1]

            # x_max_z_max
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_one - z_max, z_max, v_one],
                            [v_one, contact_length, v_one],
                            [v_one - z_max, v_one - z_max, v_one],
                            [v_one, v_one - contact_length, v_one],
                            [v_one - center, center, v_one - center],
                            [v_one, x_max, v_one - x_max],
                            [v_one - center, v_one - center, v_one - center],
                            [v_one, v_one - x_max, v_one - x_max],
                        ]
                    ),
                )
            )

            # x_min_z_min
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_zero, x_min, x_min],
                            [center, center, center],
                            [v_zero, v_one - x_min, x_min],
                            [center, v_one - center, center],
                            [v_zero, contact_length, v_zero],
                            [z_min, z_min, v_zero],
                            [v_zero, v_one - contact_length, v_zero],
                            [z_min, v_one - z_min, v_zero],
                        ]
                    ),
                )
            )

            # x_min_z_max
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_zero, contact_length, v_one],
                            [z_max, z_max, v_one],
                            [v_zero, v_one - contact_length, v_one],
                            [z_max, v_one - z_max, v_one],
                            [v_zero, x_min, v_one - x_min],
                            [center, center, v_one - center],
                            [v_zero, v_one - x_min, v_one - x_min],
                            [center, v_one - center, v_one - center],
                        ]
                    ),
                )
            )

            # x_max_z_min
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_one - center, center, center],
                            [v_one, x_max, x_max],
                            [v_one - center, v_one - center, center],
                            [v_one, v_one - x_max, x_max],
                            [v_one - z_min, z_min, v_zero],
                            [v_one, contact_length, v_zero],
                            [v_one - z_min, v_one - z_min, v_zero],
                            [v_one, v_one - contact_length, v_zero],
                        ]
                    ),
                )
            )

            # x_max_y_max
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_one - y_max, v_one, y_max],
                            [v_one, v_one, contact_length],
                            [v_one - y_max, v_one, v_one - y_max],
                            [v_one, v_one, v_one - contact_length],
                            [v_one - center, v_one - center, center],
                            [v_one, v_one - x_max, x_max],
                            [v_one - center, v_one - center, v_one - center],
                            [v_one, v_one - x_max, v_one - x_max],
                        ]
                    ),
                )
            )

            # y_max_z_min
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [contact_length, v_one, v_zero],
                            [v_one - contact_length, v_one, v_zero],
                            [y_max, v_one, y_max],
                            [v_one - y_max, v_one, y_max],
                            [z_min, v_one - z_min, v_zero],
                            [v_one - z_min, v_one - z_min, v_zero],
                            [center, v_one - center, center],
                            [v_one - center, v_one - center, center],
                        ]
                    ),
                )
            )

            # x_min_y_max
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_zero, v_one, contact_length],
                            [y_max, v_one, y_max],
                            [v_zero, v_one, v_one - contact_length],
                            [y_max, v_one, v_one - y_max],
                            [v_zero, v_one - x_min, x_min],
                            [center, v_one - center, center],
                            [v_zero, v_one - x_min, v_one - x_min],
                            [center, v_one - center, v_one - center],
                        ]
                    ),
                )
            )

            # y_max_z_max
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [y_max, v_one, v_one - y_max],
                            [v_one - y_max, v_one, v_one - y_max],
                            [contact_length, v_one, v_one],
                            [v_one - contact_length, v_one, v_one],
                            [center, v_one - center, v_one - center],
                            [v_one - center, v_one - center, v_one - center],
                            [z_max, v_one - z_max, v_one],
                            [v_one - z_max, v_one - z_max, v_one],
                        ]
                    ),
                )
            )

            # x_max_y_min
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_one - center, center, center],
                            [v_one, x_max, x_max],
                            [v_one - center, center, v_one - center],
                            [v_one, x_max, v_one - x_max],
                            [v_one - y_min, v_zero, y_min],
                            [v_one, v_zero, contact_length],
                            [v_one - y_min, v_zero, v_one - y_min],
                            [v_one, v_zero, v_one - contact_length],
                        ]
                    ),
                )
            )

            # x_min_y_min
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_zero, x_min, x_min],
                            [center, center, center],
                            [v_zero, x_min, v_one - x_min],
                            [center, center, v_one - center],
                            [v_zero, v_zero, contact_length],
                            [y_min, v_zero, y_min],
                            [v_zero, v_zero, v_one - contact_length],
                            [y_min, v_zero, v_one - y_min],
                        ]
                    ),
                )
            )

            # y_min_z_max
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [center, center, v_one - center],
                            [v_one - center, center, v_one - center],
                            [z_max, z_max, v_one],
                            [v_one - z_max, z_max, v_one],
                            [y_min, v_zero, v_one - y_min],
                            [v_one - y_min, v_zero, v_one - y_min],
                            [contact_length, v_zero, v_one],
                            [v_one - contact_length, v_zero, v_one],
                        ]
                    ),
                )
            )

            # y_min_z_min
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [z_min, z_min, v_zero],
                            [v_one - z_min, z_min, v_zero],
                            [center, center, center],
                            [v_one - center, center, center],
                            [contact_length, v_zero, v_zero],
                            [v_one - contact_length, v_zero, v_zero],
                            [y_min, v_zero, y_min],
                            [v_one - y_min, v_zero, y_min],
                        ]
                    ),
                )
            )

            # x_min_y_min_z_min
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_zero, v_zero, v_zero],
                            [contact_length, v_zero, v_zero],
                            [v_zero, contact_length, v_zero],
                            [z_min, z_min, v_zero],
                            [v_zero, v_zero, contact_length],
                            [y_min, v_zero, y_min],
                            [v_zero, x_min, x_min],
                            [center, center, center],
                        ]
                    ),
                )
            )

            # x_max_y_min_z_min
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_one - contact_length, v_zero, v_zero],
                            [v_one, v_zero, v_zero],
                            [v_one - z_min, z_min, v_zero],
                            [v_one, contact_length, v_zero],
                            [v_one - y_min, v_zero, y_min],
                            [v_one, v_zero, contact_length],
                            [v_one - center, center, center],
                            [v_one, x_max, x_max],
                        ]
                    ),
                )
            )

            # x_min_y_max_z_min
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_zero, v_one - contact_length, v_zero],
                            [z_min, v_one - z_min, v_zero],
                            [v_zero, v_one, v_zero],
                            [contact_length, v_one, v_zero],
                            [v_zero, v_one - x_min, x_min],
                            [center, v_one - center, center],
                            [v_zero, v_one, contact_length],
                            [y_max, v_one, y_max],
                        ]
                    ),
                )
            )

            # x_max_y_max_z_min
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_one - z_min, v_one - z_min, v_zero],
                            [v_one, v_one - contact_length, v_zero],
                            [v_one - contact_length, v_one, v_zero],
                            [v_one, v_one, v_zero],
                            [v_one - center, v_one - center, center],
                            [v_one, v_one - x_max, x_max],
                            [v_one - y_max, v_one, y_max],
                            [v_one, v_one, contact_length],
                        ]
                    ),
                )
            )

            # x_min_y_min_z_max
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_zero, v_zero, v_one - contact_length],
                            [y_min, v_zero, v_one - y_min],
                            [v_zero, x_min, v_one - x_min],
                            [center, center, v_one - center],
                            [v_zero, v_zero, v_one],
                            [contact_length, v_zero, v_one],
                            [v_zero, contact_length, v_one],
                            [z_max, z_max, v_one],
                        ]
                    ),
                )
            )

            # x_max_y_min_z_max
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_one - y_min, v_zero, v_one - y_min],
                            [v_one, v_zero, v_one - contact_length],
                            [v_one - center, center, v_one - center],
                            [v_one, x_max, v_one - x_max],
                            [v_one - contact_length, v_zero, v_one],
                            [v_one, v_zero, v_one],
                            [v_one - z_max, z_max, v_one],
                            [v_one, contact_length, v_one],
                        ]
                    ),
                )
            )

            # x_min_y_max_z_max
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_zero, v_one - x_min, v_one - x_min],
                            [center, v_one - center, v_one - center],
                            [v_zero, v_one, v_one - contact_length],
                            [y_max, v_one, v_one - y_max],
                            [v_zero, v_one - contact_length, v_one],
                            [z_max, v_one - z_max, v_one],
                            [v_zero, v_one, v_one],
                            [contact_length, v_one, v_one],
                        ]
                    ),
                )
            )

            # x_max_y_max_z_max
            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.array(
                        [
                            [v_one - center, v_one - center, v_one - center],
                            [v_one, v_one - x_max, v_one - x_max],
                            [v_one - y_max, v_one, v_one - y_max],
                            [v_one, v_one, v_one - contact_length],
                            [v_one - z_max, v_one - z_max, v_one],
                            [v_one, v_one - contact_length, v_one],
                            [v_one - contact_length, v_one, v_one],
                            [v_one, v_one, v_one],
                        ]
                    ),
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
