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
        contact_length=0.5,
        **kwargs
    ):
        if not isinstance(contact_length, float):
            raise ValueError("Invalid Type for radius")

        # if not ((wall_thickness > 0.01) and (wall_thickness < 0.99)):
        #     raise ValueError(
        #         "The thickness of the wall must be in (0.01 and 0.99)"
        #     )
        if not ((contact_length > 0) and (contact_length < 0.99)):
            raise ValueError("The length of a side must be in (0.01, 0.99)")

        if parameters is None:
            self._logd("Setting parameters to default values (0.5)")
            parameters = tuple([np.ones(1) * 0.4])

        v_zero= 0.
        v_one = 1.
        v_outer_c_h = contact_length * 0.5
        v_h_void = parameters[0][0]
        v_inner_c_h = contact_length * parameters[0][0]



        center = 0.5
        param = 1 - parameters[0][0]
        delta_out = center * contact_length
        # alpha = math.asin(delta_out / center)
        delta_in = (delta_out  * param)

        spline_list = []
        v_one_half = 0.5
        assert(v_h_void < 0.5)

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
                [center, delta_out],
                [center * param, delta_in],
                [delta_out, center],
                [delta_in, center * param],
            ]
        ) + np.array([v_one_half, v_one_half])

        bottom = np.array(
            [
                [delta_in, -center * param],
                [delta_out, -center],
                [-delta_in, -center * param],
                [-delta_out, -center],
            ]
        ) + np.array([v_one_half, v_one_half])

        bottom_left = np.array(
            [
                [-delta_in, -center * param],
                [-delta_out, -center],
                [-center * param, -delta_in],
                [-center, -delta_out],
            ]
        ) + np.array([v_one_half, v_one_half])

        left = np.array(
            [
                [-center, delta_out],
                [-center * param, delta_in],
                [-center, -delta_out],
                [-center * param, -delta_in],
            ]
        ) + np.array([v_one_half, v_one_half])

        top_left = np.array(
            [
                [-center, delta_out],
                [-center * param, delta_in],
                [-delta_out, center],
                [-delta_in, center * param],
            ]
        ) + np.array([v_one_half, v_one_half])

        top = np.array(
            [
                [delta_out, center],
                [delta_in, center * param],
                [-delta_out, center],
                [-delta_in, center * param],
            ]
        ) + np.array([v_one_half, v_one_half])

        bottom_right = np.array(
            [
                [delta_out, -center],
                [delta_in, -center * param],
                [center, -delta_out],
                [center * param, -delta_in],
            ]
        ) + np.array([v_one_half, v_one_half])

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
