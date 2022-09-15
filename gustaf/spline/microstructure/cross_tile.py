import numpy as np

from gustaf.spline import base


class CrossTile():

    def parametrized_closing_tile(
        self,
        branch_thickness=.2,
        closure=None,
        boundary_width=0.1,
        filling_height=0.75
    ):
        """
        Create a closing tile to match with closed surface

        Parameters
        ----------
        branch_thickness : float
          radius of fitting cylinder in branch (0,0.5)
        closure : str
          ["x_min","x_max"] Rest not yet implemented
        boundary_width : float
          with of the boundary surronding branch
        filling_height : float
          portion of the height that is filled in parametric domain

        Results
        -------
        list_of_splines : list
        """
        # Check parameters
        if closure is None:
            raise ValueError("No closing direction given")

        if not (0. < float(branch_thickness) < .5):
            raise ValueError("Thickness out of range (0, .5)")

        if not (0. < float(boundary_width) < .5):
            raise ValueError("Boundary Width is out of range")

        if not (0. < float(filling_height) < 1.):
            raise ValueError("Filling must  be in (0,1)")

        inv_boundary_width = 1. - boundary_width
        inv_filling_height = 1. - filling_height
        center_width = 1. - 2 * boundary_width
        ctps_mid_height_top = (1+filling_height)*.5
        ctps_mid_height_bottom = 1. - ctps_mid_height_top
        r_center = center_width * .5

        spline_list = []
        if closure == "z_min":
            ctps_corner = np.array([
                [0., 0., 0.],
                [boundary_width, 0., 0.],
                [0., boundary_width, 0.],
                [boundary_width, boundary_width, 0.],
                [0., 0., filling_height],
                [boundary_width, 0., filling_height],
                [0., boundary_width, filling_height],
                [boundary_width, boundary_width, filling_height]
            ])

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=ctps_corner
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + np.array([0., inv_boundary_width, 0.])
                    )
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + np.array([inv_boundary_width, 0., 0.])
                    )
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + np.array([inv_boundary_width,
                                   inv_boundary_width, 0.])
                    )
                )
            )

            center_ctps = np.array([
                [boundary_width,     boundary_width,     0.],
                [inv_boundary_width, boundary_width,     0.],
                [boundary_width,     inv_boundary_width, 0.],
                [inv_boundary_width, inv_boundary_width, 0.],
                [boundary_width,     boundary_width,     filling_height],
                [inv_boundary_width, boundary_width,     filling_height],
                [boundary_width,     inv_boundary_width, filling_height],
                [inv_boundary_width, inv_boundary_width, filling_height]
            ])

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=center_ctps
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.maximum(
                        center_ctps - np.array([center_width, 0, 0]), 0)
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.maximum(
                        center_ctps - np.array([0, center_width, 0]), 0)
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.minimum(
                        center_ctps + np.array([center_width, 0, 0]), 1.)
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.minimum(
                        center_ctps + np.array([0, center_width, 0]), 1.)
                )
            )
            branch_ctps = np.array([
                [-r_center, -r_center, filling_height],
                [r_center, -r_center, filling_height],
                [-r_center, r_center, filling_height],
                [r_center, r_center, filling_height],
                [-branch_thickness, -branch_thickness, ctps_mid_height_top],
                [branch_thickness, -branch_thickness, ctps_mid_height_top],
                [-branch_thickness, branch_thickness, ctps_mid_height_top],
                [branch_thickness, branch_thickness, ctps_mid_height_top],
                [-branch_thickness, -branch_thickness, 1.],
                [branch_thickness, -branch_thickness, 1.],
                [-branch_thickness, branch_thickness, 1.],
                [branch_thickness, branch_thickness, 1.]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 2],
                    control_points=branch_ctps
                )
            )

            return spline_list
        elif closure == "z_max":
            ctps_corner = np.array([
                [0., 0., inv_filling_height],
                [boundary_width, 0., inv_filling_height],
                [0., boundary_width, inv_filling_height],
                [boundary_width, boundary_width, inv_filling_height],
                [0., 0., 1.],
                [boundary_width, 0., 1.],
                [0., boundary_width, 1.],
                [boundary_width, boundary_width, 1.]
            ])

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=ctps_corner
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + np.array([0., inv_boundary_width, 0.])
                    )
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + np.array([inv_boundary_width, 0., 0.])
                    )
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + np.array([inv_boundary_width,
                                   inv_boundary_width, 0.])
                    )
                )
            )

            center_ctps = np.array([
                [boundary_width,     boundary_width,     inv_filling_height],
                [inv_boundary_width, boundary_width,     inv_filling_height],
                [boundary_width,     inv_boundary_width, inv_filling_height],
                [inv_boundary_width, inv_boundary_width, inv_filling_height],
                [boundary_width,     boundary_width,     1.],
                [inv_boundary_width, boundary_width,     1.],
                [boundary_width,     inv_boundary_width, 1.],
                [inv_boundary_width, inv_boundary_width, 1.]
            ])

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=center_ctps
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.maximum(
                        center_ctps - np.array([center_width, 0, 0]), 0)
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.maximum(
                        center_ctps - np.array([0, center_width, 0]), 0)
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.minimum(
                        center_ctps + np.array([center_width, 0, 0]), 1.)
                )
            )

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 1],
                    control_points=np.minimum(
                        center_ctps + np.array([0, center_width, 0]), 1.)
                )
            )

            branch_ctps = np.array([
                [-branch_thickness, -branch_thickness, 0.],
                [branch_thickness, -branch_thickness, 0.],
                [-branch_thickness, branch_thickness, 0.],
                [branch_thickness, branch_thickness, 0.],
                [-branch_thickness, -branch_thickness, ctps_mid_height_bottom],
                [branch_thickness, -branch_thickness, ctps_mid_height_bottom],
                [-branch_thickness, branch_thickness, ctps_mid_height_bottom],
                [branch_thickness, branch_thickness, ctps_mid_height_bottom],
                [-r_center, -r_center, inv_filling_height],
                [r_center, -r_center, inv_filling_height],
                [-r_center, r_center, inv_filling_height],
                [r_center, r_center, inv_filling_height]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[1, 1, 2],
                    control_points=branch_ctps
                )
            )

            return spline_list
        else:
            raise ValueError("Corner Type not supported")

    def parametrized_microtile(
        self,
        x_min_r=0.2,
        x_max_r=0.2,
        y_min_r=0.2,
        y_max_r=0.2,
        z_min_r=0.2,
        z_max_r=0.2,
        center_expansion=1.
    ):
        """
        Create a microtile based on the parameters that describe the branch
        thicknesses

        Thickness parameters are used to describe the inner radius of the outward
        facing branches
        Parameters
        ----------
        x_min_r : float
            thickness at branch with coords x_min
        x_max_r : float
            thickness at branch with coords x_max
        y_min_r : float
            thickness at branch with coords y_min
        y_max_r : float
            thickness at branch with coords y_max
        z_min_r : float
            thickness at branch with coords z_min
        z_max_r : float
            thickness at branch with coords z_max
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
        max_radius = min(.5, (.5/center_expansion))
        for radius in [x_min_r, x_max_r, y_min_r, y_max_r, z_min_r, z_max_r]:
            if not isinstance(radius, float):
                raise ValueError("Invalid type")
            if not (radius > 0 and radius < max_radius):
                raise ValueError(
                    f"Radii must be in (0,{max_radius}) for "
                    f"center_expansion {center_expansion}")

        # center radius
        center_r = (x_min_r +
                    x_max_r +
                    y_min_r +
                    y_max_r +
                    z_min_r +
                    z_max_r) / 6. * center_expansion
        hd_center = 0.5 * (0.5 + center_r)

        # Create the center-tile
        center_points = np.array([
            [-center_r, -center_r, -center_r],
            [center_r, -center_r, -center_r],
            [-center_r,  center_r, -center_r],
            [center_r,  center_r, -center_r],
            [-center_r, -center_r,  center_r],
            [center_r, -center_r,  center_r],
            [-center_r,  center_r,  center_r],
            [center_r,  center_r,  center_r]
        ])

        center_spline = base.Bezier(
            degrees=[1, 1, 1],
            control_points=center_points+[.5, .5, .5]
        )

        # X-Axis branches
        # X-Min-Branch
        aux_x_min = min(x_min_r, center_r)
        x_min_ctps = np.array([
            [-.5, -x_min_r, -x_min_r],
            [-hd_center, -aux_x_min,  -aux_x_min],
            center_points[0, :],
            [-.5, x_min_r, -x_min_r],
            [-hd_center, aux_x_min, -aux_x_min],
            center_points[2, :],
            [-.5,        -x_min_r,  x_min_r],
            [-hd_center, -aux_x_min, aux_x_min],
            center_points[4, :],
            [-.5,        x_min_r,   x_min_r],
            [-hd_center, aux_x_min, aux_x_min],
            center_points[6, :]
        ])
        x_min_spline = base.Bezier(
            degrees=[2, 1, 1],
            control_points=x_min_ctps+[.5, .5, .5]
        )
        # X-Min-Branch
        aux_x_max = min(x_max_r, center_r)
        x_max_ctps = np.array([
            center_points[1, :],
            [hd_center, -aux_x_max, -aux_x_max],
            [.5, -x_max_r, -x_max_r],
            center_points[3, :],
            [hd_center, aux_x_max, -aux_x_max],
            [.5, x_max_r, -x_max_r],
            center_points[5, :],
            [hd_center, -aux_x_max, aux_x_max],
            [.5, -x_max_r, x_max_r],
            center_points[7, :],
            [hd_center, aux_x_max, aux_x_max],
            [.5, x_max_r, x_max_r]
        ])
        x_max_spline = base.Bezier(
            degrees=[2, 1, 1],
            control_points=x_max_ctps + [.5, .5, .5]
        )

        # Y-Axis branches
        # Y-Min-Branch
        aux_y_min = min(y_min_r, center_r)
        y_min_ctps = np.array([
            [-y_min_r, -.5,  -y_min_r],
            [y_min_r, -.5, -y_min_r],
            [-aux_y_min, -hd_center, -aux_y_min],
            [aux_y_min, -hd_center,  -aux_y_min],
            center_points[0, :],
            center_points[1, :],
            [-y_min_r, -.5, y_min_r],
            [y_min_r, -.5, y_min_r],
            [-aux_y_min, -hd_center, aux_y_min],
            [aux_y_min, -hd_center, aux_y_min],
            center_points[4, :],
            center_points[5, :]
        ])
        y_min_spline = base.Bezier(
            degrees=[1, 2,  1],
            control_points=y_min_ctps+[.5, .5, .5]
        )
        # Y-Min-Branch
        aux_y_max = min(y_max_r, center_r)
        y_max_ctps = np.array([
            center_points[2, :],
            center_points[3, :],
            [-aux_y_max, hd_center, -aux_y_max],
            [aux_y_max, hd_center, -aux_y_max],
            [-y_max_r, .5, -y_max_r],
            [y_max_r, .5,  -y_max_r],
            center_points[6, :],
            center_points[7, :],
            [-aux_y_max, hd_center, aux_y_max],
            [aux_y_max, hd_center, aux_y_max],
            [-y_max_r, .5,  y_max_r],
            [y_max_r, .5,  y_max_r]
        ])
        y_max_spline = base.Bezier(
            degrees=[1, 2, 1],
            control_points=y_max_ctps + [.5, .5, .5]
        )

        # Y-Axis branches
        # Y-Min-Branch
        aux_z_min = min(z_min_r, center_r)
        z_min_ctps = np.array([
            [-z_min_r,  -z_min_r, -.5],
            [z_min_r, -z_min_r, -.5],
            [-z_min_r, z_min_r, -.5],
            [z_min_r, z_min_r, -.5],
            [-aux_z_min, - aux_z_min, -hd_center],
            [aux_z_min, - aux_z_min, -hd_center],
            [-aux_z_min, aux_z_min, -hd_center],
            [aux_z_min, aux_z_min, -hd_center],
            center_points[0, :],
            center_points[1, :],
            center_points[2, :],
            center_points[3, :]
        ])
        z_min_spline = base.Bezier(
            degrees=[1,  1, 2],
            control_points=z_min_ctps+[.5, .5, .5]
        )
        # Y-Min-Branch
        aux_z_max = min(z_max_r, center_r)
        z_max_ctps = np.array([
            center_points[4, :],
            center_points[5, :],
            center_points[6, :],
            center_points[7, :],
            [-aux_z_max, -aux_z_max, hd_center],
            [aux_z_max, -aux_z_max, hd_center],
            [-aux_z_max, aux_z_max, hd_center],
            [aux_z_max, aux_z_max, hd_center],
            [-z_max_r, -z_max_r, .5],
            [z_max_r,  -z_max_r, .5],
            [-z_max_r,   z_max_r, .5],
            [z_max_r,   z_max_r, .5]
        ])
        z_max_spline = base.Bezier(
            degrees=[1, 1, 2],
            control_points=z_max_ctps + [.5, .5, .5]
        )

        return [
            center_spline,
            x_min_spline,
            x_max_spline,
            y_min_spline,
            y_max_spline,
            z_min_spline,
            z_max_spline
        ]


class InverseCrossTile():
    """
    Class that provides necessary functions to create inverse microtile, that
    can be used to describe the domain within a microstructure
    """

    def parametrized_closing_tile(
        self,
        branch_thickness=.2,
        closure=None,
        boundary_width=0.1,
        filling_height=0.5,
        seperator_distance=None
    ):
        """
        Create a closing tile to match with closed surface

        Parameters
        ----------
        branch_thickness : float
          radius of fitting cylinder in branch (0,0.5)
        closure : str
          ["x_min","x_max"] Rest not yet implemented
        boundary_width : float
          with of the boundary surronding branch
        filling_height : float
          portion of the height that is filled in parametric domain

        Results
        -------
        list_of_splines : list
        """
        # Check parameters
        if closure is None:
            raise ValueError("No closing direction given")

        if seperator_distance is None:
            raise ValueError(
                "Seperator Distance is missing. The value is required to "
                "create watertight connections with neighboring elements. The"
                " value should be greater than the biggest branch radius"
            )

        if not (0. < float(branch_thickness) < seperator_distance):
            raise ValueError("Thickness out of range (0, .5)")

        if not (0. < float(boundary_width) < .5):
            raise ValueError("Boundary Width is out of range")

        if not (0. < float(filling_height) < 1.):
            raise ValueError("Filling must  be in (0,1)")

        # Precompute auxiliary values
        inv_filling_height = 1. - filling_height
        ctps_mid_height_top = (1+filling_height)*.5
        ctps_mid_height_bottom = 1. - ctps_mid_height_top
        center_width = 1. - 2 * boundary_width
        r_center = center_width * .5
        half_r_center = (r_center + .5) * .5
        aux_column_width = .5 - 2*(.5 - seperator_distance)

        spline_list = []
        if closure == "z_min":
            branch_neighbor_x_min_ctps = np.array([
                [-.5, -r_center, filling_height],
                [-half_r_center, -r_center, filling_height],
                [-r_center, -r_center, filling_height],
                [-.5, r_center, filling_height],
                [-half_r_center, r_center, filling_height],
                [-r_center, r_center, filling_height],
                [-.5, -aux_column_width, ctps_mid_height_top],
                [-seperator_distance, -aux_column_width, ctps_mid_height_top],
                [-branch_thickness, -branch_thickness, ctps_mid_height_top],
                [-.5, aux_column_width, ctps_mid_height_top],
                [-seperator_distance, aux_column_width, ctps_mid_height_top],
                [-branch_thickness, branch_thickness, ctps_mid_height_top],
                [-.5, -aux_column_width, 1.],
                [-seperator_distance, -aux_column_width, 1.],
                [-branch_thickness, -branch_thickness, 1.],
                [-.5, aux_column_width, 1.],
                [-seperator_distance, aux_column_width, 1.],
                [-branch_thickness, branch_thickness, 1.]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 1, 2],
                    control_points=branch_neighbor_x_min_ctps
                )
            )

            branch_neighbor_x_max_ctps = np.array([
                [r_center, -r_center, filling_height],
                [half_r_center, -r_center, filling_height],
                [.5, -r_center, filling_height],
                [r_center, r_center, filling_height],
                [half_r_center, r_center, filling_height],
                [.5, r_center, filling_height],
                [branch_thickness, -branch_thickness, ctps_mid_height_top],
                [seperator_distance, -aux_column_width, ctps_mid_height_top],
                [.5, -aux_column_width, ctps_mid_height_top],
                [branch_thickness, branch_thickness, ctps_mid_height_top],
                [seperator_distance, aux_column_width, ctps_mid_height_top],
                [.5, aux_column_width, ctps_mid_height_top],
                [branch_thickness, -branch_thickness, 1.],
                [seperator_distance, -aux_column_width, 1.],
                [.5, -aux_column_width, 1.],
                [branch_thickness, branch_thickness, 1.],
                [seperator_distance, aux_column_width, 1.],
                [.5, aux_column_width, 1.]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 1, 2],
                    control_points=branch_neighbor_x_max_ctps
                )
            )

            branch_neighbor_y_min_ctps = np.array([
                [-r_center, -.5, filling_height],
                [r_center, -.5, filling_height],
                [-r_center, -half_r_center, filling_height],
                [r_center, -half_r_center, filling_height],
                [-r_center, -r_center, filling_height],
                [r_center, -r_center, filling_height],

                [-aux_column_width, -.5, ctps_mid_height_top],
                [aux_column_width, -.5, ctps_mid_height_top],
                [-aux_column_width, -seperator_distance, ctps_mid_height_top],
                [aux_column_width, -seperator_distance, ctps_mid_height_top],
                [-branch_thickness, -branch_thickness, ctps_mid_height_top],
                [branch_thickness, -branch_thickness, ctps_mid_height_top],

                [-aux_column_width, -.5, 1.],
                [aux_column_width, -.5, 1.],
                [-aux_column_width, -seperator_distance, 1.],
                [aux_column_width, -seperator_distance, 1.],
                [-branch_thickness, -branch_thickness, 1.],
                [branch_thickness, -branch_thickness, 1.]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[1, 2, 2],
                    control_points=branch_neighbor_y_min_ctps
                )
            )

            branch_neighbor_y_max_ctps = np.array([
                [-r_center, r_center, filling_height],
                [r_center, r_center, filling_height],
                [-r_center, half_r_center, filling_height],
                [r_center, half_r_center, filling_height],
                [-r_center, .5, filling_height],
                [r_center, .5, filling_height],
                [-branch_thickness, branch_thickness, ctps_mid_height_top],
                [branch_thickness, branch_thickness, ctps_mid_height_top],
                [-aux_column_width, seperator_distance, ctps_mid_height_top],
                [aux_column_width, seperator_distance, ctps_mid_height_top],
                [-aux_column_width, .5, ctps_mid_height_top],
                [aux_column_width, .5, ctps_mid_height_top],
                [-branch_thickness, branch_thickness, 1.],
                [branch_thickness, branch_thickness, 1.],
                [-aux_column_width, seperator_distance, 1.],
                [aux_column_width, seperator_distance, 1.],
                [-aux_column_width, .5, 1.],
                [aux_column_width, .5, 1.]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[1, 2, 2],
                    control_points=branch_neighbor_y_max_ctps
                )
            )

            branch_x_min_y_min_ctps = np.array([
                [-.5, -.5, filling_height],
                [-half_r_center, -.5, filling_height],
                [-r_center, -.5, filling_height],
                [-.5, -half_r_center, filling_height],
                [-half_r_center, -half_r_center, filling_height],
                [-r_center, -half_r_center, filling_height],
                [-.5, -r_center, filling_height],
                [-half_r_center, -r_center, filling_height],
                [-r_center, -r_center, filling_height],

                [-.5, -.5, ctps_mid_height_top],
                [-seperator_distance, -.5, ctps_mid_height_top],
                [-aux_column_width, -.5, ctps_mid_height_top],
                [-.5, -seperator_distance, ctps_mid_height_top],
                [-seperator_distance, -seperator_distance, ctps_mid_height_top],
                [-aux_column_width, -seperator_distance, ctps_mid_height_top],
                [-.5, -aux_column_width, ctps_mid_height_top],
                [-seperator_distance, -aux_column_width, ctps_mid_height_top],
                [-branch_thickness, -branch_thickness, ctps_mid_height_top],

                [-.5, -.5, 1.],
                [-seperator_distance, -.5, 1.],
                [-aux_column_width, -.5, 1.],
                [-.5, -seperator_distance, 1.],
                [-seperator_distance, -seperator_distance, 1.],
                [-aux_column_width, -seperator_distance, 1.],
                [-.5, -aux_column_width, 1.],
                [-seperator_distance, -aux_column_width, 1.],
                [-branch_thickness, -branch_thickness, 1.]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 2, 2],
                    control_points=branch_x_min_y_min_ctps
                )
            )

            branch_x_min_y_max_ctps = np.array([
                [-.5, r_center, filling_height],
                [-half_r_center, r_center, filling_height],
                [-r_center, r_center, filling_height],
                [-.5, half_r_center, filling_height],
                [-half_r_center, half_r_center, filling_height],
                [-r_center, half_r_center, filling_height],
                [-.5, .5, filling_height],
                [-half_r_center, .5, filling_height],
                [-r_center, .5, filling_height],

                [-.5, aux_column_width, ctps_mid_height_top],
                [-seperator_distance, aux_column_width, ctps_mid_height_top],
                [-branch_thickness, branch_thickness, ctps_mid_height_top],
                [-.5, seperator_distance, ctps_mid_height_top],
                [-seperator_distance, seperator_distance, ctps_mid_height_top],
                [-aux_column_width, seperator_distance, ctps_mid_height_top],
                [-.5, .5, ctps_mid_height_top],
                [-seperator_distance, .5, ctps_mid_height_top],
                [-aux_column_width, .5, ctps_mid_height_top],

                [-.5, aux_column_width, 1.],
                [-seperator_distance, aux_column_width, 1.],
                [-branch_thickness, branch_thickness, 1.],
                [-.5, seperator_distance, 1.],
                [-seperator_distance, seperator_distance, 1.],
                [-aux_column_width, seperator_distance, 1.],
                [-.5, .5, 1.],
                [-seperator_distance, .5, 1.],
                [-aux_column_width, .5, 1.]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 2, 2],
                    control_points=branch_x_min_y_max_ctps
                )
            )

            branch_x_max_y_min_ctps = np.array([
                [r_center, -.5, filling_height],
                [half_r_center, -.5, filling_height],
                [.5, -.5, filling_height],
                [r_center, -half_r_center, filling_height],
                [half_r_center, -half_r_center, filling_height],
                [.5, -half_r_center, filling_height],
                [r_center, -r_center, filling_height],
                [half_r_center, -r_center, filling_height],
                [.5, -r_center, filling_height],

                [aux_column_width, -.5, ctps_mid_height_top],
                [seperator_distance, -.5, ctps_mid_height_top],
                [.5, -.5, ctps_mid_height_top],
                [aux_column_width, -seperator_distance, ctps_mid_height_top],
                [seperator_distance, -seperator_distance, ctps_mid_height_top],
                [.5, -seperator_distance, ctps_mid_height_top],
                [branch_thickness, -branch_thickness, ctps_mid_height_top],
                [seperator_distance, -aux_column_width, ctps_mid_height_top],
                [.5, -aux_column_width, ctps_mid_height_top],

                [aux_column_width, -.5, 1.],
                [seperator_distance, -.5, 1.],
                [.5, -.5, 1.],
                [aux_column_width, -seperator_distance, 1.],
                [seperator_distance, -seperator_distance, 1.],
                [.5, -seperator_distance, 1.],
                [branch_thickness, -branch_thickness, 1.],
                [seperator_distance, -aux_column_width, 1.],
                [.5, -aux_column_width, 1.],
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 2, 2],
                    control_points=branch_x_max_y_min_ctps
                )
            )

            branch_x_max_y_max_ctps = np.array([
                [r_center, r_center, filling_height],
                [half_r_center, r_center, filling_height],
                [.5, r_center, filling_height],
                [r_center, half_r_center, filling_height],
                [half_r_center, half_r_center, filling_height],
                [.5, half_r_center, filling_height],
                [r_center, .5, filling_height],
                [half_r_center, .5, filling_height],
                [.5, .5, filling_height],

                [branch_thickness, branch_thickness, ctps_mid_height_top],
                [seperator_distance, aux_column_width, ctps_mid_height_top],
                [.5, aux_column_width, ctps_mid_height_top],
                [aux_column_width, seperator_distance, ctps_mid_height_top],
                [seperator_distance, seperator_distance, ctps_mid_height_top],
                [.5, seperator_distance, ctps_mid_height_top],
                [aux_column_width, .5, ctps_mid_height_top],
                [seperator_distance, .5, ctps_mid_height_top],
                [.5, .5, ctps_mid_height_top],

                [branch_thickness, branch_thickness, 1.],
                [seperator_distance, aux_column_width, 1.],
                [.5, aux_column_width, 1.],
                [aux_column_width, seperator_distance, 1.],
                [seperator_distance, seperator_distance, 1.],
                [.5, seperator_distance, 1.],
                [aux_column_width, .5, 1.],
                [seperator_distance, .5, 1.],
                [.5, .5, 1.]

            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 2, 2],
                    control_points=branch_x_max_y_max_ctps
                )
            )

            return spline_list

        elif closure == "z_max":
            branch_neighbor_x_min_ctps = np.array([
                [-.5, -aux_column_width, 0.],
                [-seperator_distance, -aux_column_width, 0.],
                [-branch_thickness, -branch_thickness, 0.],
                [-.5, aux_column_width, 0.],
                [-seperator_distance, aux_column_width, 0.],
                [-branch_thickness, branch_thickness, 0.],

                [-.5, -aux_column_width, ctps_mid_height_bottom],
                [-seperator_distance, -aux_column_width, ctps_mid_height_bottom],
                [-branch_thickness, -branch_thickness, ctps_mid_height_bottom],
                [-.5, aux_column_width, ctps_mid_height_bottom],
                [-seperator_distance, aux_column_width, ctps_mid_height_bottom],
                [-branch_thickness, branch_thickness, ctps_mid_height_bottom],

                [-.5, -r_center, inv_filling_height],
                [-half_r_center, -r_center, inv_filling_height],
                [-r_center, -r_center, inv_filling_height],
                [-.5, r_center, inv_filling_height],
                [-half_r_center, r_center, inv_filling_height],
                [-r_center, r_center, inv_filling_height]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 1, 2],
                    control_points=branch_neighbor_x_min_ctps
                )
            )

            branch_neighbor_x_max_ctps = np.array([
                [branch_thickness, -branch_thickness, 0.],
                [seperator_distance, -aux_column_width,  0.],
                [.5, -aux_column_width,  0.],
                [branch_thickness, branch_thickness,  0.],
                [seperator_distance, aux_column_width,  0.],
                [.5, aux_column_width,  0.],
                [branch_thickness, -branch_thickness, ctps_mid_height_bottom],
                [seperator_distance, -aux_column_width, ctps_mid_height_bottom],
                [.5, -aux_column_width, ctps_mid_height_bottom],
                [branch_thickness, branch_thickness, ctps_mid_height_bottom],
                [seperator_distance, aux_column_width, ctps_mid_height_bottom],
                [.5, aux_column_width, ctps_mid_height_bottom],
                [r_center, -r_center, inv_filling_height],
                [half_r_center, -r_center, inv_filling_height],
                [.5, -r_center, inv_filling_height],
                [r_center, r_center, inv_filling_height],
                [half_r_center, r_center, inv_filling_height],
                [.5, r_center, inv_filling_height]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 1, 2],
                    control_points=branch_neighbor_x_max_ctps
                )
            )

            branch_neighbor_y_min_ctps = np.array([
                [-aux_column_width, -.5, 0.],
                [aux_column_width, -.5, 0.],
                [-aux_column_width, -seperator_distance, 0.],
                [aux_column_width, -seperator_distance, 0.],
                [-branch_thickness, -branch_thickness, 0.],
                [branch_thickness, -branch_thickness, 0.],
                [-aux_column_width, -.5, ctps_mid_height_bottom],
                [aux_column_width, -.5, ctps_mid_height_bottom],
                [-aux_column_width, -seperator_distance, ctps_mid_height_bottom],
                [aux_column_width, -seperator_distance, ctps_mid_height_bottom],
                [-branch_thickness, -branch_thickness, ctps_mid_height_bottom],
                [branch_thickness, -branch_thickness, ctps_mid_height_bottom],
                [-r_center, -.5, inv_filling_height],
                [r_center, -.5, inv_filling_height],
                [-r_center, -half_r_center, inv_filling_height],
                [r_center, -half_r_center, inv_filling_height],
                [-r_center, -r_center, inv_filling_height],
                [r_center, -r_center, inv_filling_height]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[1, 2, 2],
                    control_points=branch_neighbor_y_min_ctps
                )
            )

            branch_neighbor_y_max_ctps = np.array([
                [-branch_thickness, branch_thickness, 0.],
                [branch_thickness, branch_thickness,  0.],
                [-aux_column_width, seperator_distance,  0.],
                [aux_column_width, seperator_distance,  0.],
                [-aux_column_width, .5,  0.],
                [aux_column_width, .5,  0.],

                [-branch_thickness, branch_thickness, ctps_mid_height_bottom],
                [branch_thickness, branch_thickness, ctps_mid_height_bottom],
                [-aux_column_width, seperator_distance, ctps_mid_height_bottom],
                [aux_column_width, seperator_distance, ctps_mid_height_bottom],
                [-aux_column_width, .5, ctps_mid_height_bottom],
                [aux_column_width, .5, ctps_mid_height_bottom],

                [-r_center, r_center, inv_filling_height],
                [r_center, r_center, inv_filling_height],
                [-r_center, half_r_center, inv_filling_height],
                [r_center, half_r_center, inv_filling_height],
                [-r_center, .5, inv_filling_height],
                [r_center, .5, inv_filling_height]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[1, 2, 2],
                    control_points=branch_neighbor_y_max_ctps
                )
            )

            branch_x_min_y_min_ctps = np.array([
                [-.5, -.5, 0.],
                [-seperator_distance, -.5,  0.],
                [-aux_column_width, -.5,  0.],
                [-.5, -seperator_distance,  0.],
                [-seperator_distance, -seperator_distance,  0.],
                [-aux_column_width, -seperator_distance,  0.],
                [-.5, -aux_column_width,  0.],
                [-seperator_distance, -aux_column_width,  0.],
                [-branch_thickness, -branch_thickness,  0.],

                [-.5, -.5, ctps_mid_height_bottom],
                [-seperator_distance, -.5, ctps_mid_height_bottom],
                [-aux_column_width, -.5, ctps_mid_height_bottom],
                [-.5, -seperator_distance, ctps_mid_height_bottom],
                [-seperator_distance, -seperator_distance, ctps_mid_height_bottom],
                [-aux_column_width, -seperator_distance, ctps_mid_height_bottom],
                [-.5, -aux_column_width, ctps_mid_height_bottom],
                [-seperator_distance, -aux_column_width, ctps_mid_height_bottom],
                [-branch_thickness, -branch_thickness, ctps_mid_height_bottom],

                [-.5, -.5, inv_filling_height],
                [-half_r_center, -.5, inv_filling_height],
                [-r_center, -.5, inv_filling_height],
                [-.5, -half_r_center, inv_filling_height],
                [-half_r_center, -half_r_center, inv_filling_height],
                [-r_center, -half_r_center, inv_filling_height],
                [-.5, -r_center, inv_filling_height],
                [-half_r_center, -r_center, inv_filling_height],
                [-r_center, -r_center, inv_filling_height]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 2, 2],
                    control_points=branch_x_min_y_min_ctps
                )
            )

            branch_x_max_y_max_ctps = np.array([
                [branch_thickness, branch_thickness, 0.],
                [seperator_distance, aux_column_width,  0.],
                [.5, aux_column_width,  0.],
                [aux_column_width, seperator_distance,  0.],
                [seperator_distance, seperator_distance,  0.],
                [.5, seperator_distance,  0.],
                [aux_column_width, .5,  0.],
                [seperator_distance, .5,  0.],
                [.5, .5,  0.],

                [branch_thickness, branch_thickness, ctps_mid_height_bottom],
                [seperator_distance, aux_column_width, ctps_mid_height_bottom],
                [.5, aux_column_width, ctps_mid_height_bottom],
                [aux_column_width, seperator_distance, ctps_mid_height_bottom],
                [seperator_distance, seperator_distance, ctps_mid_height_bottom],
                [.5, seperator_distance, ctps_mid_height_bottom],
                [aux_column_width, .5, ctps_mid_height_bottom],
                [seperator_distance, .5, ctps_mid_height_bottom],
                [.5, .5, ctps_mid_height_bottom],

                [r_center, r_center, inv_filling_height],
                [half_r_center, r_center, inv_filling_height],
                [.5, r_center, inv_filling_height],
                [r_center, half_r_center, inv_filling_height],
                [half_r_center, half_r_center, inv_filling_height],
                [.5, half_r_center, inv_filling_height],
                [r_center, .5, inv_filling_height],
                [half_r_center, .5, inv_filling_height],
                [.5, .5, inv_filling_height]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 2, 2],
                    control_points=branch_x_max_y_max_ctps
                )
            )

            branch_x_max_y_min_ctps = np.array([
                [aux_column_width, -.5, 0.],
                [seperator_distance, -.5,  0.],
                [.5, -.5,  0.],
                [aux_column_width, -seperator_distance,  0.],
                [seperator_distance, -seperator_distance,  0.],
                [.5, -seperator_distance,  0.],
                [branch_thickness, -branch_thickness,  0.],
                [seperator_distance, -aux_column_width,  0.],
                [.5, -aux_column_width,  0.],

                [aux_column_width, -.5, ctps_mid_height_bottom],
                [seperator_distance, -.5, ctps_mid_height_bottom],
                [.5, -.5, ctps_mid_height_bottom],
                [aux_column_width, -seperator_distance, ctps_mid_height_bottom],
                [seperator_distance, -seperator_distance, ctps_mid_height_bottom],
                [.5, -seperator_distance, ctps_mid_height_bottom],
                [branch_thickness, -branch_thickness, ctps_mid_height_bottom],
                [seperator_distance, -aux_column_width, ctps_mid_height_bottom],
                [.5, -aux_column_width, ctps_mid_height_bottom],

                [r_center, -.5, inv_filling_height],
                [half_r_center, -.5, inv_filling_height],
                [.5, -.5, inv_filling_height],
                [r_center, -half_r_center, inv_filling_height],
                [half_r_center, -half_r_center, inv_filling_height],
                [.5, -half_r_center, inv_filling_height],
                [r_center, -r_center, inv_filling_height],
                [half_r_center, -r_center, inv_filling_height],
                [.5, -r_center, inv_filling_height]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 2, 2],
                    control_points=branch_x_max_y_min_ctps
                )
            )

            branch_x_min_y_max_ctps = np.array([
                [-.5, aux_column_width, 0.],
                [-seperator_distance, aux_column_width,  0.],
                [-branch_thickness, branch_thickness,  0.],
                [-.5, seperator_distance,  0.],
                [-seperator_distance, seperator_distance,  0.],
                [-aux_column_width, seperator_distance,  0.],
                [-.5, .5,  0.],
                [-seperator_distance, .5,  0.],
                [-aux_column_width, .5,  0.],

                [-.5, aux_column_width, ctps_mid_height_bottom],
                [-seperator_distance, aux_column_width, ctps_mid_height_bottom],
                [-branch_thickness, branch_thickness, ctps_mid_height_bottom],
                [-.5, seperator_distance, ctps_mid_height_bottom],
                [-seperator_distance, seperator_distance, ctps_mid_height_bottom],
                [-aux_column_width, seperator_distance, ctps_mid_height_bottom],
                [-.5, .5, ctps_mid_height_bottom],
                [-seperator_distance, .5, ctps_mid_height_bottom],
                [-aux_column_width, .5, ctps_mid_height_bottom],

                [-.5, r_center, inv_filling_height],
                [-half_r_center, r_center, inv_filling_height],
                [-r_center, r_center, inv_filling_height],
                [-.5, half_r_center, inv_filling_height],
                [-half_r_center, half_r_center, inv_filling_height],
                [-r_center, half_r_center, inv_filling_height],
                [-.5, .5, inv_filling_height],
                [-half_r_center, .5, inv_filling_height],
                [-r_center, .5, inv_filling_height]
            ]) + np.array([.5, .5, 0.])

            spline_list.append(
                base.Bezier(
                    degrees=[2, 2, 2],
                    control_points=branch_x_min_y_max_ctps
                )
            )

            return spline_list
        else:
            raise ValueError("Corner Type not supported")

    def parametrized_microtile(
        self,
        x_min_r=0.2,
        x_max_r=0.2,
        y_min_r=0.2,
        y_max_r=0.2,
        z_min_r=0.2,
        z_max_r=0.2,
        seperator_distance=None,
        center_expansion=1.
    ):
        """
        Create an inverse microtile based on the parameters that describe the
        branch thicknesses

        Thickness parameters are used to describe the inner radius of the outward
        facing branches
        Parameters
        ----------
        x_min_r : float
            thickness at branch with coords x_min
        x_max_r : float
            thickness at branch with coords x_max
        y_min_r : float
            thickness at branch with coords y_min
        y_max_r : float
            thickness at branch with coords y_max
        z_min_r : float
            thickness at branch with coords z_min
        z_max_r : float
            thickness at branch with coords z_max
        seperator_distance : float
            position of the control points for higher order elements
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

        if seperator_distance is None:
            raise ValueError(
                "Seperator Distance is missing. The value is required to "
                "create watertight connections with neighboring elements. The"
                " value should be greater than the biggest branch radius"
            )

        # Check if all radii are in allowed range
        max_radius = min(.5, (.5/center_expansion))
        max_radius = min(max_radius, seperator_distance)
        for radius in [x_min_r, x_max_r, y_min_r, y_max_r, z_min_r, z_max_r]:
            if not isinstance(radius, float):
                raise ValueError("Invalid type")
            if not (radius > 0 and radius < max_radius):
                raise ValueError(
                    f"Radii must be in (0,{max_radius}) for "
                    f"center_expansion {center_expansion}")

        # center radius
        center_r = (x_min_r +
                    x_max_r +
                    y_min_r +
                    y_max_r +
                    z_min_r +
                    z_max_r) / 6. * center_expansion

        # Auxiliary values for smooothing (mid-branch thickness)
        aux_x_min = min(x_min_r, center_r)
        aux_x_max = min(x_max_r, center_r)
        aux_y_min = min(y_min_r, center_r)
        aux_y_max = min(y_max_r, center_r)
        aux_z_min = min(z_min_r, center_r)
        aux_z_max = min(z_max_r, center_r)
        # Branch midlength
        hd_center = 0.5 * (0.5 + center_r)

        #
        if seperator_distance is None:
            seperator_distance = .45
        aux_column_width = .5 - 2*(.5 - seperator_distance)

        # Init return type
        spline_list = []

        # Start with branch interconnections
        x_min_y_min = np.array([
            [-.5, -.5, -aux_column_width],
            [-seperator_distance, -.5, -aux_column_width],
            [-y_min_r, -.5, -y_min_r],
            [-.5, -seperator_distance, -aux_column_width],
            [-hd_center, -hd_center, -aux_column_width],
            [-aux_y_min, -hd_center, -aux_y_min],
            [-.5, -x_min_r, -x_min_r],
            [-hd_center, -aux_x_min, -aux_x_min],
            [-center_r, -center_r, -center_r],
            [-.5, -.5, aux_column_width],
            [-seperator_distance, -.5, aux_column_width],
            [-y_min_r, -.5, y_min_r],
            [-.5, -seperator_distance, aux_column_width],
            [-hd_center, -hd_center, aux_column_width],
            [-aux_y_min, -hd_center, aux_y_min],
            [-.5, -x_min_r, x_min_r],
            [-hd_center, -aux_x_min, aux_x_min],
            [-center_r, -center_r, center_r]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 1],
                control_points=x_min_y_min
            )
        )

        x_max_y_min = np.array([
            [y_min_r, -.5, -y_min_r],
            [seperator_distance, -.5, -aux_column_width],
            [.5, -.5, -aux_column_width],
            [aux_y_min, -hd_center, -aux_y_min],
            [hd_center, -hd_center, -aux_column_width],
            [.5, -seperator_distance, -aux_column_width],
            [center_r, -center_r, -center_r],
            [hd_center, -aux_x_max, -aux_x_max],
            [.5, -x_max_r, -x_max_r],
            [y_min_r, -.5, y_min_r],
            [seperator_distance, -.5, aux_column_width],
            [.5, -.5, aux_column_width],
            [aux_y_min, -hd_center, aux_y_min],
            [hd_center, -hd_center, aux_column_width],
            [.5, -seperator_distance, aux_column_width],
            [center_r, -center_r, center_r],
            [hd_center, -aux_x_max, aux_x_max],
            [.5, -x_max_r, x_max_r]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 1],
                control_points=x_max_y_min
            )
        )

        x_min_y_max = np.array([
            [-.5, x_min_r, -x_min_r],
            [-hd_center, aux_x_min, -aux_x_min],
            [-center_r, center_r, -center_r],
            [-.5, seperator_distance, -aux_column_width],
            [-hd_center, hd_center, -aux_column_width],
            [-aux_y_max, hd_center, -aux_y_max],
            [-.5, .5, -aux_column_width],
            [-seperator_distance, .5, -aux_column_width],
            [-y_max_r, .5, -y_max_r],
            [-.5, x_min_r, x_min_r],
            [-hd_center, aux_x_min, aux_x_min],
            [-center_r, center_r, center_r],
            [-.5, seperator_distance, aux_column_width],
            [-hd_center, hd_center, aux_column_width],
            [-aux_y_max, hd_center, aux_y_max],
            [-.5, .5, aux_column_width],
            [-seperator_distance, .5, aux_column_width],
            [-y_max_r, .5, y_max_r]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 1],
                control_points=x_min_y_max
            )
        )

        x_max_y_max = np.array([
            [center_r, center_r, -center_r],
            [hd_center, aux_x_max, -aux_x_max],
            [.5, x_max_r, -x_max_r],
            [aux_y_max, hd_center, -aux_y_max],
            [hd_center, hd_center, -aux_column_width],
            [.5, seperator_distance, -aux_column_width],
            [y_max_r, .5, -y_max_r],
            [seperator_distance, .5, -aux_column_width],
            [.5, .5, -aux_column_width],
            [center_r, center_r, center_r],
            [hd_center, aux_x_max, aux_x_max],
            [.5, x_max_r, x_max_r],
            [aux_y_max, hd_center, aux_y_max],
            [hd_center, hd_center, aux_column_width],
            [.5, seperator_distance, aux_column_width],
            [y_max_r, .5, y_max_r],
            [seperator_distance, .5, aux_column_width],
            [.5, .5, aux_column_width]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 1],
                control_points=x_max_y_max
            )
        )

        x_min_z_min = np.array([
            [-.5, -aux_column_width, -.5],
            [-seperator_distance, -aux_column_width, -.5],
            [-z_min_r, -z_min_r, -.5],
            [-.5, aux_column_width, -.5],
            [-seperator_distance, aux_column_width, -.5],
            [-z_min_r, z_min_r, -.5],
            [-.5, -aux_column_width, -seperator_distance],
            [-hd_center, -aux_column_width, -hd_center],
            [-aux_z_min, -aux_z_min, -hd_center],
            [-.5, aux_column_width, -seperator_distance],
            [-hd_center, aux_column_width, -hd_center],
            [-aux_z_min, aux_z_min, -hd_center],
            [-.5, -x_min_r, -x_min_r],
            [-hd_center, -aux_x_min, -aux_x_min],
            [-center_r, -center_r, -center_r],
            [-.5, x_min_r, -x_min_r],
            [-hd_center, aux_x_min, -aux_x_min],
            [-center_r, center_r, -center_r]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 1, 2],
                control_points=x_min_z_min
            )
        )

        x_max_z_min = np.array([
            [z_min_r, -z_min_r, -.5],
            [seperator_distance, -aux_column_width, -.5],
            [.5, -aux_column_width, -.5],
            [z_min_r, z_min_r, -.5],
            [seperator_distance, aux_column_width, -.5],
            [.5, aux_column_width, -.5],
            [aux_z_min, -aux_z_min, -hd_center],
            [hd_center, -aux_column_width, -hd_center],
            [.5, -aux_column_width, -seperator_distance],
            [aux_z_min, aux_z_min, -hd_center],
            [hd_center, aux_column_width, -hd_center],
            [.5, aux_column_width, -seperator_distance],
            [center_r, -center_r, -center_r],
            [hd_center, -aux_x_max, -aux_x_max],
            [.5, -x_max_r, -x_max_r],
            [center_r, center_r, -center_r],
            [hd_center, aux_x_max, -aux_x_max],
            [.5, x_max_r, -x_max_r]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 1, 2],
                control_points=x_max_z_min
            )
        )

        x_min_z_max = np.array([
            [-.5, -x_min_r, x_min_r],
            [-hd_center, -aux_x_min, aux_x_min],
            [-center_r, -center_r, center_r],
            [-.5, x_min_r, x_min_r],
            [-hd_center, aux_x_min, aux_x_min],
            [-center_r, center_r, center_r],
            [-.5, -aux_column_width, seperator_distance],
            [-hd_center, -aux_column_width, hd_center],
            [-aux_z_max, -aux_z_max, hd_center],
            [-.5, aux_column_width, seperator_distance],
            [-hd_center, aux_column_width, hd_center],
            [-aux_z_max, aux_z_max, hd_center],
            [-.5, -aux_column_width, .5],
            [-seperator_distance, -aux_column_width, .5],
            [-z_max_r, -z_max_r, .5],
            [-.5, aux_column_width, .5],
            [-seperator_distance, aux_column_width, .5],
            [-z_max_r, z_max_r, .5]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 1, 2],
                control_points=x_min_z_max
            )
        )

        x_max_z_max = np.array([
            [center_r, -center_r, center_r],
            [hd_center, -aux_x_max, aux_x_max],
            [.5, -x_max_r, x_max_r],
            [center_r, center_r, center_r],
            [hd_center, aux_x_max, aux_x_max],
            [.5, x_max_r, x_max_r],
            [aux_z_max, -aux_z_max, hd_center],
            [hd_center, -aux_column_width, hd_center],
            [.5, -aux_column_width, seperator_distance],
            [aux_z_max, aux_z_max, hd_center],
            [hd_center, aux_column_width, hd_center],
            [.5, aux_column_width, seperator_distance],
            [z_max_r, -z_max_r, .5],
            [seperator_distance, -aux_column_width, .5],
            [.5, -aux_column_width, .5],
            [z_max_r, z_max_r, .5],
            [seperator_distance, aux_column_width, .5],
            [.5, aux_column_width, .5]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 1, 2],
                control_points=x_max_z_max
            )
        )

        y_min_z_min = np.array([
            [-aux_column_width, -.5, -.5],
            [aux_column_width, -.5, -.5],
            [-aux_column_width, -seperator_distance, -.5],
            [aux_column_width, -seperator_distance, -.5],
            [-z_min_r, -z_min_r, -.5],
            [z_min_r, -z_min_r, -.5],
            [-aux_column_width, -.5, -seperator_distance],
            [aux_column_width, -.5, -seperator_distance],
            [-aux_column_width, -hd_center, -hd_center],
            [aux_column_width, -hd_center, -hd_center],
            [-aux_z_min, -aux_z_min, -hd_center],
            [aux_z_min, -aux_z_min, -hd_center],
            [-y_min_r, -.5, -y_min_r],
            [y_min_r, -.5, -y_min_r],
            [-aux_y_min, -hd_center, -aux_y_min],
            [aux_y_min, -hd_center, -aux_y_min],
            [-center_r, -center_r, -center_r],
            [center_r, -center_r, -center_r]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[1, 2, 2],
                control_points=y_min_z_min
            )
        )

        y_max_z_min = np.array([
            [-z_min_r, z_min_r, -.5],
            [z_min_r, z_min_r, -.5],
            [-aux_column_width, seperator_distance, -.5],
            [aux_column_width, seperator_distance, -.5],
            [-aux_column_width, .5, -.5],
            [aux_column_width, .5, -.5],
            [-aux_z_min, aux_z_min, -hd_center],
            [aux_z_min, aux_z_min, -hd_center],
            [-aux_column_width, hd_center, -hd_center],
            [aux_column_width, hd_center, -hd_center],
            [-aux_column_width, .5, -seperator_distance],
            [aux_column_width, .5, -seperator_distance],
            [-center_r, center_r, -center_r],
            [center_r, center_r, -center_r],
            [-aux_y_max, hd_center, -aux_y_max],
            [aux_y_max, hd_center, -aux_y_max],
            [-y_max_r, .5, -y_max_r],
            [y_max_r, .5, -y_max_r]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[1, 2, 2],
                control_points=y_max_z_min
            )
        )

        y_min_z_max = np.array([
            [-y_min_r, -.5, y_min_r],
            [y_min_r, -.5, y_min_r],
            [-aux_y_min, -hd_center, aux_y_min],
            [aux_y_min, -hd_center, aux_y_min],
            [-center_r, -center_r, center_r],
            [center_r, -center_r, center_r],
            [-aux_column_width, -.5, seperator_distance],
            [aux_column_width, -.5, seperator_distance],
            [-aux_column_width, -hd_center, hd_center],
            [aux_column_width, -hd_center, hd_center],
            [-aux_z_max, -aux_z_max, hd_center],
            [aux_z_max, -aux_z_max, hd_center],
            [-aux_column_width, -.5, .5],
            [aux_column_width, -.5, .5],
            [-aux_column_width, -seperator_distance, .5],
            [aux_column_width, -seperator_distance, .5],
            [-z_max_r, -z_max_r, .5],
            [z_max_r, -z_max_r, .5],
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[1, 2, 2],
                control_points=y_min_z_max
            )
        )

        y_max_z_max = np.array([
            [-center_r, center_r, center_r],
            [center_r, center_r, center_r],
            [-aux_y_max, hd_center, aux_y_max],
            [aux_y_max, hd_center, aux_y_max],
            [-y_max_r, .5, y_max_r],
            [y_max_r, .5, y_max_r],
            [-aux_z_max, aux_z_max, hd_center],
            [aux_z_max, aux_z_max, hd_center],
            [-aux_column_width, hd_center, hd_center],
            [aux_column_width, hd_center, hd_center],
            [-aux_column_width, .5, seperator_distance],
            [aux_column_width, .5, seperator_distance],
            [-z_max_r, z_max_r, .5],
            [z_max_r, z_max_r, .5],
            [-aux_column_width, seperator_distance, .5],
            [aux_column_width, seperator_distance, .5],
            [-aux_column_width, .5, .5],
            [aux_column_width, .5, .5]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[1, 2, 2],
                control_points=y_max_z_max
            )
        )

        x_min_y_min_z_min = np.array([
            [-.5, -.5, -.5],
            [-seperator_distance, -.5, -.5],
            [-aux_column_width, -.5, -.5],
            [-.5, -seperator_distance, -.5],
            [-seperator_distance, -seperator_distance, -.5],
            [-aux_column_width, -seperator_distance, -.5],
            [-.5, -aux_column_width, -.5],
            [-seperator_distance, -aux_column_width, -.5],
            [-z_min_r, -z_min_r, -.5],
            [-.5, -.5, -seperator_distance],
            [-seperator_distance, -.5, -seperator_distance],
            [-aux_column_width, -.5, -seperator_distance],
            [-.5, -seperator_distance, -seperator_distance],
            [-hd_center, -hd_center, -hd_center],
            [-aux_column_width, -hd_center, -hd_center],
            [-.5, -aux_column_width, -seperator_distance],
            [-hd_center, -aux_column_width, -hd_center],
            [-aux_z_min, -aux_z_min, -hd_center],
            [-.5, -.5, -aux_column_width],
            [-seperator_distance, -.5, -aux_column_width],
            [-y_min_r, -.5, -y_min_r],
            [-.5, -seperator_distance, -aux_column_width],
            [-hd_center, -hd_center, -aux_column_width],
            [-aux_y_min, -hd_center, -aux_y_min],
            [-.5, -x_min_r, -x_min_r],
            [-hd_center, -aux_x_min, -aux_x_min],
            [-center_r, -center_r, -center_r]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 2],
                control_points=x_min_y_min_z_min
            )
        )

        x_max_y_min_z_min = np.array([
            [aux_column_width, -.5, -.5],
            [seperator_distance, -.5, -.5],
            [.5, -.5, -.5],
            [aux_column_width, -seperator_distance, -.5],
            [seperator_distance, -seperator_distance, -.5],
            [.5, -seperator_distance, -.5],
            [z_min_r, -z_min_r, -.5],
            [seperator_distance, -aux_column_width, -.5],
            [.5, -aux_column_width, -.5],
            [aux_column_width, -.5, -seperator_distance],
            [seperator_distance, -.5, -seperator_distance],
            [.5, -.5, -seperator_distance],
            [aux_column_width, -hd_center, -hd_center],
            [hd_center, -hd_center, -hd_center],
            [.5, -seperator_distance, -seperator_distance],
            [aux_z_min, -aux_z_min, -hd_center],
            [hd_center, -aux_column_width, -hd_center],
            [.5, -aux_column_width, -seperator_distance],
            [y_min_r, -.5, -y_min_r],
            [seperator_distance, -.5, -aux_column_width],
            [.5, -.5, -aux_column_width],
            [aux_y_min, -hd_center, -aux_y_min],
            [hd_center, -hd_center, -aux_column_width],
            [.5, -seperator_distance, -aux_column_width],
            [center_r, -center_r, -center_r],
            [hd_center, -aux_x_max, -aux_x_max],
            [.5, -x_max_r, -x_max_r]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 2],
                control_points=x_max_y_min_z_min
            )
        )

        x_min_y_max_z_min = np.array([
            [-.5, aux_column_width, -.5],
            [-seperator_distance, aux_column_width, -.5],
            [-z_min_r, z_min_r, -.5],
            [-.5, seperator_distance, -.5],
            [-seperator_distance, seperator_distance, -.5],
            [-aux_column_width, seperator_distance, -.5],
            [-.5, .5, -.5],
            [-seperator_distance, .5, -.5],
            [-aux_column_width, .5, -.5],
            [-.5, aux_column_width, -seperator_distance],
            [-hd_center, aux_column_width, -hd_center],
            [-aux_z_min, aux_z_min, -hd_center],
            [-.5, seperator_distance, -seperator_distance],
            [-hd_center, hd_center, -hd_center],
            [-aux_column_width, hd_center, -hd_center],
            [-.5, .5, -seperator_distance],
            [-seperator_distance, .5, -seperator_distance],
            [-aux_column_width, .5, -seperator_distance],
            [-.5, x_min_r, -x_min_r],
            [-hd_center, aux_x_min, -aux_x_min],
            [-center_r, center_r, -center_r],
            [-.5, seperator_distance, -aux_column_width],
            [-hd_center, hd_center, -aux_column_width],
            [-aux_y_max, hd_center, -aux_y_max],
            [-.5, .5, -aux_column_width],
            [-seperator_distance, .5, -aux_column_width],
            [-y_max_r, .5, -y_max_r]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 2],
                control_points=x_min_y_max_z_min
            )
        )

        x_max_y_max_z_min = np.array([
            [z_min_r, z_min_r, -.5],
            [seperator_distance, aux_column_width, -.5],
            [.5, aux_column_width, -.5],
            [aux_column_width, seperator_distance, -.5],
            [seperator_distance, seperator_distance, -.5],
            [.5, seperator_distance, -.5],
            [aux_column_width, .5, -.5],
            [seperator_distance, .5, -.5],
            [.5, .5, -.5],
            [aux_z_min, aux_z_min, -hd_center],
            [hd_center, aux_column_width, -hd_center],
            [.5, aux_column_width, -seperator_distance],
            [aux_column_width, hd_center, -hd_center],
            [hd_center, hd_center, -hd_center],
            [.5, seperator_distance, -seperator_distance],
            [aux_column_width, .5, -seperator_distance],
            [seperator_distance, .5, -seperator_distance],
            [.5, .5, -seperator_distance],
            [center_r, center_r, -center_r],
            [hd_center, aux_x_max, -aux_x_max],
            [.5, x_max_r, -x_max_r],
            [aux_y_max, hd_center, -aux_y_max],
            [hd_center, hd_center, -aux_column_width],
            [.5, seperator_distance, -aux_column_width],
            [y_max_r, .5, -y_max_r],
            [seperator_distance, .5, -aux_column_width],
            [.5, .5, -aux_column_width]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 2],
                control_points=x_max_y_max_z_min
            )
        )

        x_min_y_min_z_max = np.array([
            [-.5, -.5, aux_column_width],
            [-seperator_distance, -.5, aux_column_width],
            [-y_min_r, -.5, y_min_r],
            [-.5, -seperator_distance, aux_column_width],
            [-hd_center, -hd_center, aux_column_width],
            [-aux_y_min, -hd_center, aux_y_min],
            [-.5, -x_min_r, x_min_r],
            [-hd_center, -aux_x_min, aux_x_min],
            [-center_r, -center_r, center_r],
            [-.5, -.5, seperator_distance],
            [-seperator_distance, -.5, seperator_distance],
            [-aux_column_width, -.5, seperator_distance],
            [-.5, -seperator_distance, seperator_distance],
            [-hd_center, -hd_center, hd_center],
            [-aux_column_width, -hd_center, hd_center],
            [-.5, -aux_column_width, seperator_distance],
            [-hd_center, -aux_column_width, hd_center],
            [-aux_z_max, -aux_z_max, hd_center],
            [-.5, -.5, .5],
            [-seperator_distance, -.5, .5],
            [-aux_column_width, -.5, .5],
            [-.5, -seperator_distance, .5],
            [-seperator_distance, -seperator_distance, .5],
            [-aux_column_width, -seperator_distance, .5],
            [-.5, -aux_column_width, .5],
            [-seperator_distance, -aux_column_width, .5],
            [-z_max_r, -z_max_r, .5]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 2],
                control_points=x_min_y_min_z_max
            )
        )

        x_max_y_min_z_max = np.array([
            [y_min_r, -.5, y_min_r],
            [seperator_distance, -.5, aux_column_width],
            [.5, -.5, aux_column_width],
            [aux_y_min, -hd_center, aux_y_min],
            [hd_center, -hd_center, aux_column_width],
            [.5, - seperator_distance, aux_column_width],
            [center_r, -center_r, center_r],
            [hd_center, -aux_x_max, aux_x_max],
            [.5, -x_max_r, x_max_r],
            [aux_column_width, -.5, seperator_distance],
            [seperator_distance, - .5, seperator_distance],
            [.5, -.5, seperator_distance],
            [aux_column_width, -hd_center, hd_center],
            [hd_center, -hd_center, hd_center],
            [.5, -seperator_distance, seperator_distance],
            [aux_z_max, - aux_z_max, hd_center],
            [hd_center, - aux_column_width, hd_center],
            [.5, -aux_column_width, seperator_distance],
            [aux_column_width, -.5, .5],
            [seperator_distance, -.5, .5],
            [.5, -.5, .5],
            [aux_column_width, -seperator_distance, .5],
            [seperator_distance, -seperator_distance, .5],
            [.5, -seperator_distance, .5],
            [z_max_r, - z_max_r, .5],
            [seperator_distance, -aux_column_width, .5],
            [.5, -aux_column_width, .5],
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 2],
                control_points=x_max_y_min_z_max
            )
        )

        x_min_y_max_z_max = np.array([
            [-.5, x_min_r, x_min_r],
            [-hd_center, aux_x_min, aux_x_min],
            [-center_r, center_r, center_r],
            [-.5, seperator_distance, aux_column_width],
            [-hd_center, hd_center, aux_column_width],
            [-aux_y_max, hd_center, aux_y_max],
            [-.5, .5, aux_column_width],
            [-seperator_distance, .5, aux_column_width],
            [-y_max_r, .5, y_max_r],

            [-.5, aux_column_width, seperator_distance],
            [-hd_center, aux_column_width, hd_center],
            [-aux_z_max, aux_z_max, hd_center],
            [-.5, seperator_distance, seperator_distance],
            [-hd_center, hd_center, hd_center],
            [-aux_column_width, hd_center, hd_center],
            [-.5, .5, seperator_distance],
            [-seperator_distance, .5, seperator_distance],
            [-aux_column_width, .5, seperator_distance],

            [-.5, aux_column_width, .5],
            [-seperator_distance, aux_column_width, .5],
            [-z_max_r, z_max_r, .5],
            [-.5, seperator_distance, .5],
            [-seperator_distance, seperator_distance, .5],
            [-aux_column_width, seperator_distance, .5],
            [-.5, .5, .5],
            [-seperator_distance, .5, .5],
            [-aux_column_width, .5, .5]
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 2],
                control_points=x_min_y_max_z_max
            )
        )

        x_max_y_max_z_max = np.array([
            [center_r, center_r, center_r],
            [hd_center, aux_x_max, aux_x_max],
            [.5, x_max_r, x_max_r],
            [aux_y_max, hd_center, aux_y_max],
            [hd_center, hd_center, aux_column_width],
            [.5, seperator_distance, aux_column_width],
            [y_max_r, .5, y_max_r],
            [seperator_distance, .5, aux_column_width],
            [.5, .5, aux_column_width],
            [aux_z_max, aux_z_max, hd_center],
            [hd_center, aux_column_width, hd_center],
            [.5, aux_column_width, seperator_distance],
            [aux_column_width, hd_center, hd_center],
            [hd_center, hd_center, hd_center],
            [.5, seperator_distance, seperator_distance],
            [aux_column_width, .5, seperator_distance],
            [seperator_distance, .5, seperator_distance],
            [.5, .5, seperator_distance],
            [z_max_r, z_max_r, .5],
            [seperator_distance, aux_column_width, .5],
            [.5, aux_column_width, .5],
            [aux_column_width, seperator_distance, .5],
            [seperator_distance, seperator_distance, .5],
            [.5, seperator_distance, .5],
            [aux_column_width, .5, .5],
            [seperator_distance, .5, .5],
            [.5, .5, .5],
        ]) + np.array([.5, .5, .5])

        spline_list.append(
            base.Bezier(
                degrees=[2, 2, 2],
                control_points=x_max_y_max_z_max
            )
        )

        return spline_list
