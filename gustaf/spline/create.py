"""gustaf/spline/create.py

Create operations for spline geometries
"""


import itertools

import numpy as np
from splinepy._spline import _RequiredProperties

from gustaf import utils
from gustaf import settings


def extrude(spline, extrusion_vector=None):
    """
    Extrudes Splines

    Parameters
    ----------
    spline: GustafSpline
    extrusion_vector: np.ndarray
    """
    from gustaf.spline.base import GustafSpline

    # Check input type
    if not issubclass(type(spline), GustafSpline):
        raise NotImplementedError("Extrude only works for splines")

    # Check extrusion_vector
    if extrusion_vector is not None:
        # make flat extrusion_vector
        extrusion_vector = np.asarray(extrusion_vector).ravel()
    else:
        raise ValueError("No extrusion extrusion_vector given")

    # Check extrusion_vector dimension
    # formulate correct cps
    if spline.dim == extrusion_vector.shape[0]:
        cps = spline.control_points
    elif spline.dim < extrusion_vector.shape[0]:
        expansion_dimension = extrusion_vector.shape[0] - spline.dim
        # one smaller dim is allowed
        # warn that we assume new dim is all zero
        utils.log.warning(
            f"Given extrusion vector is {expansion_dimension} dimension bigger than "
            "spline's dim. Assuming 0.0 entries for new dimension.",
        )
        cps = np.hstack(
            (
                spline.control_points,
                np.zeros((len(spline.control_points), expansion_dimension))
            )
        )
    else:
        raise ValueError(
            "Dimension Mismatch between extrusion extrusion vector and spline."
        )

    # Start Extrusion
    spline_dict = dict()

    spline_dict["degrees"] = np.concatenate((spline.degrees, [1]))
    spline_dict["control_points"] = np.vstack((cps, cps + extrusion_vector))
    if "knot_vectors" in _RequiredProperties.of(spline):
        spline_dict["knot_vectors"] = spline.knot_vectors + [[0, 0, 1, 1]]
    if "weights" in _RequiredProperties.of(spline):
        spline_dict["weights"] = np.concatenate(
            (spline.weights, spline.weights))

    return type(spline)(**spline_dict)


def revolve(spline,
            axis=None,
            center=None,
            angle=None,
            n_knot_spans=None,
            degree=True
            ):
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
    degree : bool
      use degrees instead of radiant

    Returns
    -------
    spline : GustafSpline         
    """
    from gustaf.spline.base import GustafSpline

    # Check input type
    if not issubclass(type(spline), GustafSpline):
        raise NotImplementedError("Revolutions only works for splines")

    # Check axis
    if axis is not None:
        # Transform into numpy array
        axis = np.asarray(axis).ravel()
        # Check Axis dimension
        if (spline.control_points.shape[1] > axis.shape[0]):
            raise ValueError(
                "Dimension Mismatch between extrusion axis and spline."
            )
        elif (spline.control_points.shape[1] < axis.shape[0]):
            utils.log.warning(
                "Control Point dimension is smaller than axis dimension,"
                " filling with zeros"
            )
            expansion_dimension = axis.shape[0] - spline.dim
            cps = np.hstack(
                (
                    spline.control_points,
                    np.zeros((len(spline.control_points), expansion_dimension))
                )
            )
        else:
            cps = np.copy(spline.control_points)

        # Make sure axis is normalized

        axis_norm = np.linalg.norm(axis)
        if not np.isclose(axis_norm, 0, atol=settings.TOLERANCE):
            axis = axis / axis_norm
        else:
            raise ValueError("Axis-norm is too close to zero.")
    else:
        cps = np.copy(spline.control_points)
        if spline.control_points.shape[1] == 3:
            raise ValueError("No rotation axis given")

    # Set Problem dimension
    problem_dimension = cps.shape[1]

    # Make sure axis is ignored for 2D
    if problem_dimension == 2:
        axis = None

    # Update angle
    if degree:
        angle = np.radians(angle)

    # Init center
    if center is not None:
        center = np.asarray(center).ravel()
        # Check Axis dimension
        if not (problem_dimension == center.shape[0]):
            raise ValueError(
                "Dimension Mismatch between axis and center of rotation."
            )
        cps -= center

    # The parametric dimension is independent of the revolution but the
    # rotation-matrix is only implemented for 2D and 3D problems
    if not (cps.shape[1] == 2 or cps.shape[1] == 3):
        raise NotImplementedError(
            "Sorry,"
            "revolutions only implemented for 2D and 3D splines"
        )

    # Angle must be (0, pi) non including
    # Rotation is always performed in half steps
    PI = np.pi
    minimum_n_knot_spans = int(
        np.ceil(np.abs((angle + settings.TOLERANCE) / PI)))
    if (n_knot_spans) is None or (n_knot_spans < minimum_n_knot_spans):
        n_knot_spans = minimum_n_knot_spans

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

    # Determine rotation matrix
    rotation_matrix = utils.arr.rotation_matrix_around_axis(
                          axis=axis,
                          rotation=rot_a,
                          degree=False
                      ).T

    # Start Extrusion
    spline_dict = dict()

    spline_dict["degrees"] = np.concatenate((spline.degrees, [2]))

    spline_dict["control_points"] = cps
    end_points = cps
    for i_segment in range(n_knot_spans):
        # Rotate around axis
        mid_points = np.matmul(end_points, rotation_matrix)
        end_points = np.matmul(mid_points, rotation_matrix)
        # Move away from axis using dot-product tricks
        if problem_dimension == 3:
            mp_scale = axis * np.dot(mid_points, axis).reshape(-1, 1)
            mid_points = (mid_points - mp_scale) * factor + mp_scale
        else:
            mid_points *= factor
        spline_dict["control_points"] = np.concatenate((
            spline_dict["control_points"],
            mid_points,
            end_points
        ))

    if "knot_vectors" in _RequiredProperties.of(spline):
        kv = [0, 0, 0]
        [kv.extend([i + 1, i + 1]) for i in range(n_knot_spans - 1)]
        spline_dict["knot_vectors"] = spline.knot_vectors + [
            kv + [n_knot_spans+1] * 3
        ]
    if "weights" in _RequiredProperties.of(spline):
        mid_weights = spline.weights * weight
        spline_dict["weights"] = spline.weights
        for i_segment in range(n_knot_spans):
            spline_dict["weights"] = np.concatenate((
                spline_dict["weights"],
                mid_weights,
                spline.weights
            ))
    else:
        utils.log.warning(
            "True revolutions are only possible for rational spline types.\n"
            "Creating Approximation"
        )

    if center is not None:
        spline_dict["control_points"] += center

    return type(spline)(**spline_dict)


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
