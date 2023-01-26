import math

import numpy as np

from gustaf.spline import base
from gustaf.spline.microstructure.tiles.tilebase import TileBase


class NutTile2D(TileBase):
    def __init__(self):
        """Simple tile - looks like a nut"""
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
        contact_length=0.49,
        **kwargs
    ):

        if not isinstance(contact_length, float):
            raise ValueError("Invalid Type for radius")

        if not ((contact_length > 0) and (contact_length < 0.99)):
            raise ValueError("The length of a side must be in (0.01, 0.99)")

        if parameters is None:
            self._logd("Setting parameters to default values (0.5)")
            parameters = tuple([np.ones(1) * 0.49])

        v_h_void = parameters[0][0]
        if not ((v_h_void > 0.01) and (v_h_void < 0.5)):
            raise ValueError(
                "The thickness of the wall must be in (0.01 and 0.49)"
            )

        v_zero = 0.
        v_one_half = 0.5
        v_one = 1.
        v_outer_c_h = contact_length * 0.5
        v_inner_c_h = contact_length * parameters[0][0]

        spline_list = []

        # set points:
        right = np.array(
            [
                [v_h_void+v_one_half, -v_inner_c_h+v_one_half],
                [v_one, -v_outer_c_h+v_one_half],
                [v_h_void+v_one_half, v_inner_c_h+v_one_half],
                [v_one, v_outer_c_h+v_one_half],
            ]
        )

        right_top = np.array(
            [
                [v_h_void + v_one_half, v_inner_c_h + v_one_half],
                [v_one, v_outer_c_h + v_one_half],
                [v_inner_c_h + v_one_half, v_h_void+v_one_half],
                [v_outer_c_h + v_one_half, v_one],
            ]
        )

        top = np.array(
            [
                [v_inner_c_h + v_one_half, v_h_void + v_one_half],
                [v_outer_c_h + v_one_half, v_one],
                [-v_inner_c_h + v_one_half, v_h_void + v_one_half],
                [-v_outer_c_h + v_one_half, v_one]
            ]
        )

        bottom_left = np.array(
            [
                [-v_h_void + v_one_half, -v_inner_c_h + v_one_half],
                [v_zero, -v_outer_c_h + v_one_half],
                [-v_inner_c_h + v_one_half, -v_h_void + v_one_half],
                [-v_outer_c_h + v_one_half, v_zero]
            ]
        )

        left = np.array(
            [
                [-v_h_void + v_one_half, -v_inner_c_h + v_one_half],
                [v_zero, -v_outer_c_h + v_one_half],
                [-v_h_void + v_one_half, v_inner_c_h + v_one_half],
                [v_zero, v_outer_c_h + v_one_half]
            ]
        )

        top_left = np.array(
            [
                [-v_h_void + v_one_half, v_inner_c_h + v_one_half],
                [v_zero, v_outer_c_h + v_one_half],
                [-v_inner_c_h + v_one_half, v_h_void + v_one_half],
                [-v_outer_c_h + v_one_half, v_one]
            ]
        )

        bottom = np.array(
            [
                [v_inner_c_h + v_one_half, -v_h_void + v_one_half],
                [v_outer_c_h + v_one_half, v_zero],
                [-v_inner_c_h + v_one_half, -v_h_void + v_one_half],
                [-v_outer_c_h + v_one_half, v_zero]
            ]
        )

        bottom_right = np.array(
            [
                [v_inner_c_h + v_one_half, -v_h_void + v_one_half],
                [v_outer_c_h + v_one_half, v_zero],
                [v_h_void + v_one_half, -v_inner_c_h + v_one_half],
                [v_one, -v_outer_c_h + v_one_half]
            ]
        )

        spline_list.append(base.Bezier(degrees=[1, 1], control_points=right))

        spline_list.append(
            base.Bezier(degrees=[1, 1], control_points=right_top)
        )

        spline_list.append(base.Bezier(degrees=[1, 1], control_points=bottom))

        spline_list.append(
            base.Bezier(degrees=[1, 1], control_points=bottom_left)
        )

        spline_list.append(base.Bezier(degrees=[1, 1], control_points=left))

        spline_list.append(
            base.Bezier(degrees=[1, 1], control_points=top_left)
        )

        spline_list.append(base.Bezier(degrees=[1, 1], control_points=top))

        spline_list.append(
            base.Bezier(degrees=[1, 1], control_points=bottom_right)
        )

        return spline_list
