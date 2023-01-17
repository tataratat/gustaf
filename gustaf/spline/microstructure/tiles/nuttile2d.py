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
                [0.0, 0.5],
                [1.0, 0.5],
                [0.5, 0.0],
                [0.5, 1.0],
            ]
        )
        self._parameter_space_dimension = 1

    def create_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        wall_thickness=0.5,
        side_length=0.5,
        **kwargs
    ):
        if not isinstance(wall_thickness, float):
            raise ValueError("Invalid Type for width")
        if not isinstance(side_length, float):
            raise ValueError("Invalid Type for radius")

        if not ((wall_thickness > 0.01) and (wall_thickness < 0.99)):
            raise ValueError(
                "The thickness of the wall must be in (0.01 and 0.99)"
            )
        if not ((side_length > 0) and (side_length < 0.99)):
            raise ValueError("The length of a side must be in (0.01, 0.99)")

        if parameters is None:
            self._logd("Setting parameters to default values (0.5, 0.5)")
            parameters = tuple([np.ones(5) * 0.5])

            """
                    for radius in parameters[0].tolist():
                    if not isinstance(radius, float):
                        raise ValueError("Invalid type")
            """
        center_expansion = 0.5

        [x_min_r, x_max_r, y_min_r, y_max_r] = [1.0, 1.0, 1.0, 1.0]

        center = (
            (x_min_r + x_max_r + y_min_r + y_max_r) / 4.0 * center_expansion
        )
        param = 1 - wall_thickness
        delta_out = center * side_length
        alpha = math.asin(delta_out / center)
        delta_in = math.sin(alpha) * (center * param)

        spline_list = []
        v_one_half = 0.5

        # set points:
        right = np.array(
            [
                [center, delta_out],
                [center * param, delta_in],
                [center, -delta_out],
                [center * param, -delta_in],
            ]
        ) + np.array([v_one_half, v_one_half])

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
