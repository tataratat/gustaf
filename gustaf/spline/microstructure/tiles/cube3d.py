import numpy as np

from gustaf.spline import base
from gustaf.spline.microstructure.tiles.tilebase import TileBase


class Cube3D(TileBase):
    def __init__(self):
        """Simple tile - looks like a nut"""
        self._dim = 3
        self._evaluation_points = np.array(
            [
                [0.5, 0.5, 0.5],
            ]
        )
        self._n_info_per_eval_point = 1

    def create_tile(
        self, parameters=None, parameter_sensitivities=None, **kwargs
    ):
        """Create a microtile based on the parameters that describe the strut
        thicknesses.


        Parameters
        ----------
        parameters : np.array(1, 1)
          One evaluation point with one parameter is used. This parameter
          specifies the thickness of the strut, where the value must be
          between 0.01 and 0.49.
        parameter_sensitivities: np.ndarray
          Describes the parameter sensitivities with respect to some design
          variable. In case the design variables directly apply to the
          parameter itself, they evaluate as delta_ij
        contact_length : float
            the length of the wall that contacts the other microstructure
        Returns
        -------
        microtile_list : list(splines)
        """

        if parameters is None:
            self._logd("Setting parameters to default values (0.2)")
            parameters = np.array(
                np.ones(
                    (len(self._evaluation_points), self._n_info_per_eval_point)
                )
                * 0.2
            )

        self.check_params(parameters)

        v_h_void = parameters[0, 0]
        if not ((v_h_void > 0.01) and (v_h_void < 0.5)):
            raise ValueError(
                "The thickness of the wall must be in (0.01 and 0.49)"
            )

        v_zero = 0.0
        v_one = 1.0
        # thickness
        v_t = parameters[0, 0]

        spline_list = []

        # set points:

        right_top = np.array(
            [
                [v_one - v_t, v_t, v_one],
                [v_one, v_t, v_one],
                [v_one - v_t, v_one - v_t, v_one],
                [v_one, v_one - v_t, v_one],
                [v_one - v_t, v_t, v_one - v_t],
                [v_one, v_t, v_one - v_t],
                [v_one - v_t, v_one - v_t, v_one - v_t],
                [v_one, v_one - v_t, v_one - v_t],
            ]
        )

        right_bottom = np.array(
            [
                [v_zero, v_t, v_t],
                [v_t, v_t, v_t],
                [v_zero, v_one - v_t, v_t],
                [v_t, v_one - v_t, v_t],
                [v_zero, v_t, v_zero],
                [v_t, v_t, v_zero],
                [v_zero, v_one - v_t, v_zero],
                [v_t, v_one - v_t, v_zero],
            ]
        )

        left_top = np.array(
            [
                [v_zero, v_t, v_one],
                [v_t, v_t, v_one],
                [v_zero, v_one - v_t, v_one],
                [v_t, v_one - v_t, v_one],
                [v_zero, v_t, v_one - v_t],
                [v_t, v_t, v_one - v_t],
                [v_zero, v_one - v_t, v_one - v_t],
                [v_t, v_one - v_t, v_one - v_t],
            ]
        )

        left_bottom = np.array(
            [
                [v_one - v_t, v_t, v_t],
                [v_one, v_t, v_t],
                [v_one - v_t, v_one - v_t, v_t],
                [v_one, v_one - v_t, v_t],
                [v_one - v_t, v_t, v_zero],
                [v_one, v_t, v_zero],
                [v_one - v_t, v_one - v_t, v_zero],
                [v_one, v_one - v_t, v_zero],
            ]
        )

        front_right = np.array(
            [
                [v_one - v_t, v_one, v_t],
                [v_one, v_one, v_t],
                [v_one - v_t, v_one, v_one - v_t],
                [v_one, v_one, v_one - v_t],
                [v_one - v_t, v_one - v_t, v_t],
                [v_one, v_one - v_t, v_t],
                [v_one - v_t, v_one - v_t, v_one - v_t],
                [v_one, v_one - v_t, v_one - v_t],
            ]
        )

        front_bottom = np.array(
            [
                [v_zero, v_one, v_zero],
                [v_one, v_one, v_zero],
                [v_zero, v_one, v_t],
                [v_one, v_one, v_t],
                [v_zero, v_one - v_t, v_zero],
                [v_one, v_one - v_t, v_zero],
                [v_zero, v_one - v_t, v_t],
                [v_one, v_one - v_t, v_t],
            ]
        )

        front_left = np.array(
            [
                [v_zero, v_one, v_t],
                [v_t, v_one, v_t],
                [v_zero, v_one, v_one - v_t],
                [v_t, v_one, v_one - v_t],
                [v_zero, v_one - v_t, v_t],
                [v_t, v_one - v_t, v_t],
                [v_zero, v_one - v_t, v_one - v_t],
                [v_t, v_one - v_t, v_one - v_t],
            ]
        )

        front_top = np.array(
            [
                [v_zero, v_one, v_one - v_t],
                [v_one, v_one, v_one - v_t],
                [v_zero, v_one, v_one],
                [v_one, v_one, v_one],
                [v_zero, v_one - v_t, v_one - v_t],
                [v_one, v_one - v_t, v_one - v_t],
                [v_zero, v_one - v_t, v_one],
                [v_one, v_one - v_t, v_one],
            ]
        )

        back_right = np.array(
            [
                [v_one - v_t, v_t, v_t],
                [v_one, v_t, v_t],
                [v_one - v_t, v_t, v_one - v_t],
                [v_one, v_t, v_one - v_t],
                [v_one - v_t, v_zero, v_t],
                [v_one, v_zero, v_t],
                [v_one - v_t, v_zero, v_one - v_t],
                [v_one, v_zero, v_one - v_t],
            ]
        )

        back_left = np.array(
            [
                [v_zero, v_t, v_t],
                [v_t, v_t, v_t],
                [v_zero, v_t, v_one - v_t],
                [v_t, v_t, v_one - v_t],
                [v_zero, v_zero, v_t],
                [v_t, v_zero, v_t],
                [v_zero, v_zero, v_one - v_t],
                [v_t, v_zero, v_one - v_t],
            ]
        )

        back_top = np.array(
            [
                [v_zero, v_t, v_one - v_t],
                [v_one, v_t, v_one - v_t],
                [v_zero, v_t, v_one],
                [v_one, v_t, v_one],
                [v_zero, v_zero, v_one - v_t],
                [v_one, v_zero, v_one - v_t],
                [v_zero, v_zero, v_one],
                [v_one, v_zero, v_one],
            ]
        )

        back_bottom = np.array(
            [
                [v_zero, v_t, v_zero],
                [v_one, v_t, v_zero],
                [v_zero, v_t, v_t],
                [v_one, v_t, v_t],
                [v_zero, v_zero, v_zero],
                [v_one, v_zero, v_zero],
                [v_zero, v_zero, v_t],
                [v_one, v_zero, v_t],
            ]
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=right_top)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=front_right)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=front_bottom)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=front_left)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=front_top)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=right_bottom)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=left_top)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=left_bottom)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=back_top)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=back_right)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=back_left)
        )

        spline_list.append(
            base.Bezier(degrees=[1, 1, 1], control_points=back_bottom)
        )

        return spline_list


def closing_tile(
    self,
    parameters=None,
    parameter_sensitivities=None,
    closure=None,
):
    pass
