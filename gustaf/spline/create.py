"""gustaf/spline/create.py

Create operations for spline geometries
"""


import itertools
import logging

import numpy as np


def extrude(spline, axis=None):
    """
    Extrudes Splines

    Parameters
    ----------
    spline: GustafSpline
    axis: np.ndarray
    """
    from gustaf.spline.base import GustafSpline
    
    # Check input type
    if not issubclass(type(spline), GustafSpline):
        raise NotImplementedError("Extrude only works for splines")

    # Check axis
    if axis is not None:
        # make flat axis
        axis = np.asarray(axis).ravel()
    else:
        raise ValueError("No extrusion Axis given")

    # Check Axis dimension
    # formulate correct cps
    if spline.dim == axis.shape[0]:
        cps = spline.control_points
    elif int(spline.dim + 1) == axis.shape[0]:
        # one smaller dim is allowed
        # warn that we assume new dim is all zero
        utils.log.warning(
            "Given axis is one dimension bigger than spline's dim.",
            "Assuming 0.0 entries for new dimension."
        )
        cps = np.hstack(
            (
                spline.control_points,
                np.ones((len(spline.control_points), 1)
            )
        )
    else:
        raise ValueError(
            "Dimension Mismatch between extrusion axis and spline."
        )

    # Start Extrusion
    arguments = {}

    arguments["degrees"] = np.concatenate((spline.degrees, [1]))
    arguments["control_points"] = np.vstack((cps, cps + axis))
    if "knot_vectors" in spline._required_properties:
        arguments["knot_vectors"] = spline.knot_vectors + [[0,0,1,1]]
    if "weights" in spline._required_properties:
        arguments["weights"] = np.concatenate((spline.weights, spline.weights))

    return type(spline)(**arguments)


def revolve(spline, axis=None, center=None, angle=None, n_knot_spans=None):
    """
    Revolve spline around an axis and extend its parametric dimension

    Parameters
    ----------
    spline : GustafSpline
      Basis-Spline to be revolved
    axis : np.ndarray
      Axis of revolution
    center : np.ndarray
      Center of revolution
    angle : float
    n_knot_spans : int
      number of non-zero knot-elements for result-spline (if applicable)

    Returns
    -------
    spline : GustafSpline         
    """
    from gustaf.spline.base import GustafSpline

    # Check input type
    if not issubclass(type(spline), GustafSpline):
        raise NotImplementedError("Revolutions only works for splines")

    # The parametric dimension is independent of the revolution but the
    # rotation-matrix is only implemented for 2D and 3D problems
    if not (spline.dim == 2 or spline.dim == 3):
        raise NotImplementedError(
            "Sorry,"
            "revolutions only implemented for 2D and 3D splines"
        )

    # Check axis
    if axis is not None:
        if spline.dim == 2:
            logging.warning("For 2D revolutions axis will be ignored")
            axis=None
        else:
            axis = np.asarray(axis)
            # Check Axis dimension
            if not (spline.control_points.shape[1] == axis.shape[0]):
                raise ValueError(
                    "Dimension Mismatch between extrusion axis and spline."
                )
            # Make sure axis is normalized
            axis = axis / np.linalg.norm(axis)
    else:
        if spline.control_points.shape[1] == 3:
            raise ValueError("No rotation axis given")

    # Init center
    if center is not None:
        center = np.asarray(axis)
        # Check Axis dimension
        if not (spline.control_points.shape[1] == center.shape[0]):
            raise ValueError(
                "Dimension Mismatch between extrusion center and spline."
            )
        spline.control_points -= center

    # Angle must be (0, pi) non including
    # Rotation is always performed in half steps
    PI = np.pi
    if n_knot_spans is None:
        n_knot_spans = int(np.ceil(np.abs((angle + 1e-13) / PI)))
    else:
        if n_knot_spans < int(np.ceil(np.abs((angle + 1e-13) / PI))):
            logging.warning(
                f"Number of segments was insufficient "
                "- raised to {n_knot_spans}"
            )
            n_knot_spans = int(np.ceil(np.abs((angle + 1e-13) / PI)))

    if "Bezier" in spline.whatami:
        if n_knot_spans > 1:
            raise ValueError(
                "Revolutions are only supported for angles up to 180 degrees"
                "for Bezier type splines as they consist of only one knot span"
            )

    # Determine auxiliary values
    rot_a = angle / (2 * n_knot_spans)
    half_counter_angle = PI / 2 - rot_a
    weight = np.sin(half_counter_angle)
    factor = 1 / weight


    # Assemble rotation matrix
    if spline.dim == 2:
        rotation_matrix = np.array([
            [np.cos(rot_a), -np.sin(rot_a)],
            [np.sin(rot_a), np.cos(rot_a)]
        ]).T
    else:
        # See rodrigues' formula
        rotation_matrix = np.array(
            [[0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [axis[1], axis[0], 0]]
        )
        rotation_matrix = (np.eye(3) + np.sin(rot_a) * rotation_matrix + (
            (1 - np.cos(rot_a)) * np.matmul(rotation_matrix, rotation_matrix))
            ).T

    # Start Extrusion
    arguments = {}

    arguments["degrees"] = np.concatenate((spline.degrees, [2]))

    arguments["control_points"] = spline.control_points
    end_points = spline.control_points
    for i_segment in range(n_knot_spans):
        # Rotate around axis
        mid_points = np.matmul(end_points, rotation_matrix)
        end_points = np.matmul(mid_points, rotation_matrix)
        # Move away from axis using dot-product tricks
        if spline.dim == 3:
            mp_scale = axis * np.dot(mid_points, axis).reshape(-1,1)
            mid_points = (mid_points - mp_scale) * factor + mp_scale
        else:
            mid_points *= factor
        arguments["control_points"] = np.concatenate((
            arguments["control_points"],
            mid_points,
            end_points
        ))

    if "knot_vectors" in spline._required_properties:
        kv = [0,0,0]
        [kv.extend([i + 1, i + 1]) for i in range(n_knot_spans - 1)]
        arguments["knot_vectors"] = spline.knot_vectors + [
            kv + [n_knot_spans+1] * 3
        ]
    if "weights" in spline._required_properties:
        mid_weights = spline.weights * weight
        arguments["weights"] = spline.weights
        for i_segment in range(n_knot_spans):
          arguments["weights"] = np.concatenate((
              arguments["weights"],
              mid_weights,
              spline.weights
          ))
    else:
        logging.warning(
            "True revolutions are only possible for rational spline types.\n"
            "Creating Approximation"
        )

    if center is not None:
        spline.control_points += center
        arguments["control_points"] += center

    return type(spline)(**arguments)


class _Creator:
    """
    Helper class to build new splines from existing geometries

    Examples
    ---------
    >>> myspline = <your-spline>
    >>> spline_faces = myspline.create.extrude(vector=[3,1,3])
    """

    def __init__(self, spl):
        self.spline = spl

    def extrude(self, *args, **kwargs):
        return extrude(self.spline, *args, **kwargs)

    def revolve(self, *args, **kwargs):
        return revolve(self.spline, *args, **kwargs)
