"""gustaf/spline/create.py.

Create operations for spline geometries
"""

import numpy as np
from splinepy._spline import _RequiredProperties

from gustaf import utils
from gustaf import settings


def extruded(spline, extrusion_vector=None):
    """Extrudes Splines.

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
            f"Given extrusion vector is {expansion_dimension} dimension "
            "bigger than spline's dim. Assuming 0.0 entries for "
            "new dimension.",
        )
        cps = np.hstack(
            (
                spline.control_points,
                np.zeros(
                        (
                            len(spline.control_points),
                            expansion_dimension
                        )
                )
            )
        )
    else:
        raise ValueError(
            "Dimension Mismatch between extrusion extrusion vector "
            "and spline."
        )

    # Start Extrusion
    spline_dict = dict()

    spline_dict["degrees"] = np.concatenate((spline.degrees, [1]))
    spline_dict["control_points"] = np.vstack((cps, cps + extrusion_vector))
    if "knot_vectors" in _RequiredProperties.of(spline):
        spline_dict["knot_vectors"] = spline.knot_vectors + [[0, 0, 1, 1]]
    if "weights" in _RequiredProperties.of(spline):
        spline_dict["weights"] = np.concatenate(
            (spline.weights, spline.weights)
        )

    return type(spline)(**spline_dict)


def revolved(
        spline,
        axis=None,
        center=None,
        angle=None,
        n_knot_spans=None,
        degree=True
):
    """Revolve spline around an axis and extend its parametric dimension.

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
                    np.zeros(
                            (
                                len(spline.control_points),
                                expansion_dimension
                            )
                    )
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
        np.ceil(np.abs((angle + settings.TOLERANCE) / PI))
    )
    if (n_knot_spans) is None or (n_knot_spans < minimum_n_knot_spans):
        n_knot_spans = minimum_n_knot_spans

    if "Bezier" in spline.whatami:
        if n_knot_spans > 1:
            raise ValueError(
                "Revolutions are only supported for angles up to 180 "
                "degrees for Bezier type splines as they consist of only "
                "one knot span"
            )

    # Determine auxiliary values
    rot_a = angle / (2 * n_knot_spans)
    half_counter_angle = PI / 2 - rot_a
    weight = np.sin(half_counter_angle)
    factor = 1 / weight

    # Determine rotation matrix
    rotation_matrix = utils.arr.rotation_matrix_around_axis(
        axis=axis, rotation=rot_a, degree=False
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
        spline_dict["control_points"] = np.concatenate(
            (spline_dict["control_points"], mid_points, end_points)
        )

    if "knot_vectors" in _RequiredProperties.of(spline):
        kv = [0, 0, 0]
        [kv.extend([i + 1, i + 1]) for i in range(n_knot_spans - 1)]
        spline_dict["knot_vectors"] = spline.knot_vectors + [
            kv + [n_knot_spans + 1] * 3
        ]
    if "weights" in _RequiredProperties.of(spline):
        mid_weights = spline.weights * weight
        spline_dict["weights"] = spline.weights
        for i_segment in range(n_knot_spans):
            spline_dict["weights"] = np.concatenate(
                (spline_dict["weights"], mid_weights, spline.weights)
            )
    else:
        utils.log.warning(
            "True revolutions are only possible for rational spline types."
            "\nCreating Approximation."
        )

    if center is not None:
        spline_dict["control_points"] += center

    return type(spline)(**spline_dict)


def line(points, degree=1):
    """Create a spline with the provided points as control points with given 
    degree. 

    Parameters
    ----------
    points : numpy.ndarray
        npoints x ndims array of control points
    degree : int, optional
        Desired spline degree, by default 1

    Returns
    -------
    gustaf.Bspline
        Bspline with choosen control points and if necessary internal knots.
    """
    from gustaf import BSpline
    cps = points
    nknots = cps.shape[0] + degree + 1

    knots = np.concatenate((np.full(degree, 0.),
                           np.linspace(0., 1., nknots - 2 * degree),
                           np.full(degree, 1.0)))

    return BSpline(control_points=cps, knot_vectors=[knots],
                   degrees=[degree])


def circle(r):
    """Line circle with radius r in the x-y plane around the origin. 
    The spline has an open knot vector and degree 2. 

    Parameters
    ----------
    r : float
        radius

    Returns
    -------
    gustaf.NURBS
        Line circle NURBS
    """

    from gustaf.spline import NURBS

    cps = np.array([
        [r, 0.],
        [r, -r],
        [0., -r],
        [-r, -r],
        [-r, 0.],
        [-r, r],
        [0., r],
        [r, r],
        [r, 0.]])

    weights = np.tile([1.0, 1/np.sqrt(2)], 5)[:-1]
    knots = np.repeat([0.0, 0.25, 0.5, 0.75, 1.0], [3, 2, 2, 2, 3])
    degree = 2

    return NURBS(control_points=cps, knot_vectors=[knots],
                 degrees=[degree], weights=weights)


def rectangle(a, b):
    """ Surface spline in rectangluar shape with linear degree. 
    The rectangle lies in the x-y plane with one corner in the origin. 


    Parameters
    ----------
    a : float
        dimension of the rectangle in x-direction
    b : float
        dimension of the rectangle in y-direction
    Returns
    -------
    gustaf.Bezier
        Rectangluar Bezier spline with lengths of the sides a and b. 
    """
    from gustaf.spline import Bezier

    cps = np.zeros([4, 2])
    cps[1, :] = a, 0,
    cps[2, :] = 0., b
    cps[3, :] = a, b

    return Bezier(control_points=cps, degrees=[1, 1])


def box(a, b, h):
    """Volumetric spline with linear degree and dimensions a,b,h in 
    x,y,z-direction

    Parameters
    ----------
    a : float
        dimension of the box in x-direction
    b : float
        dimension of the box in y-direction
    h : float
        dimension of the box in z-direction

    Returns
    -------
    gustaf.Bezier
        Volumetric linear Bezier spline
    """
    return rectangle(a, b).create.extruded([0., 0., h])


def disk(R, r0=0., angle=360., n_knot_spans=4):
    """Surface spline describing a potentially hollow disk with quadratic 
    degree along curved dimension and linear along thickness. 
    The angle describes the returned part of the disk. 

    Parameters
    ----------
    R : float
        Outer radius of the disk
    r0 : float, optional
        Inner radius of the disk, in case of hollow disk, by default 0.
    angle : float, optional
        Rotational angle, by default 360. describing a complete revolution
    n_knot_spans : int, optional
        Number of knot spans, by default 4

    Returns
    -------
    gustaf.NURBS
        Surface NURBS of degrees (1,2)
    """
    from gustaf.spline import NURBS

    cps = np.array([[r0, 0.], [R, 0.]])
    weights = np.ones([cps.shape[0]])
    knots = np.repeat([0., 1.], 2)

    return NURBS(control_points=cps, knot_vectors=[knots], degrees=[1],
                 weights=weights).create.revolved(axis=[0., 0., 1.],
                                                  center=np.zeros(3),
                                                  angle=angle,
                                                  n_knot_spans=n_knot_spans)


def torus(R, r, r0=0., angle=[360., 360.], n_knot_spans=[4, 4]):
    """Creates a volumetric NURBS spline describing a torus with radius R
    and cross-sectional radius r. 

    Parameters
    ----------
    R : float
        Radius of the torus
    r : float 
        Radius of the section of the torus
    r0 : float, optional
        Inner radius in case of hollow torus, by default 0.
    angle : int, optional
        Rotational angle, by default 360. describing a complete revolution
    n_knot_spans : list, optional
        Number of knot spans in 0: circular direction, 1: cross-sectional
        direction, by default both equal 4

    Returns
    -------
    gustaf.NURBS
        Volumetric spline in the shape of a torus with degrees (1,2,2)
    """

    # Create a surface spline representing a disk and move it from the origin
    d = disk(r, r0, n_knot_spans=n_knot_spans[1], angle=angle[1])
    d.control_points[:, 1] += R

    return d.create.revolved(axis=[1., 0, 0],
                             center=np.zeros(3), angle=angle[0],
                             n_knot_spans=n_knot_spans[0])


def sphere(R, angle=[360., 360.], n_knot_spans=[4, 4]):
    """Creates a volumetric spline describing a sphere with radius R.

    Parameters
    ----------
    R : float
        Radius of the sphere
    angle : list
        Rotational angle around 0: x-axis and 1: y-axis, by default each 360 
        (describing a complete revolution)
    n_knot_spans : list
        Number of knot spans in 0: x-axis and 1: y-axis

    Returns
    -------
    gustaf.NURBS
        Volumetric NURBS with degrees (1,2,2)
    """

    from gustaf.spline import NURBS

    cps = np.array([[0., 0, 0.], [0., R, R]])
    weights = np.ones([cps.shape[0]])
    knots = np.repeat([0., 1.], 2)

    return NURBS(control_points=cps, knot_vectors=[knots], degrees=1,
                 weights=weights).create.revolved(
        axis=[1, 0, 0],
        center=[0, 0, 0],
        angle=angle[0],
        n_knot_spans=n_knot_spans[0]).create.revolved(
        axis=[0, 0, 1],
        center=[0, 0, 0],
        angle=angle[1],
        n_knot_spans=n_knot_spans[1])


def cone(R, h, volumetric=True, angle=360.):
    """Creates a cone with base radius R and height h. 

    Parameters
    ----------
    R : float
        Radius of the base
    h : float
        Height of the cone
    volumetric : bool, optional
        Parameter whether surface or volume spline, by default True
    angle : float
        Rotation angle in degrees, only used for solid model

    Returns
    -------
    gustaf.NURBS
        Volumetric or surface NURBS descibing a cone
    """

    if volumetric:
        ground = disk(R, angle=angle)
    else:
        ground = circle(R)

    # Extrude in z
    con = ground.create.extruded([0, 0, h])
    # Move all upper control points to one
    con.control_points[np.isclose(con.control_points[:, -1], h)] = [0, 0, h]

    return con


def pyramid(a, b, h):
    """Creates a volumetric spline in the shape of a pyramid with linear 
    degree in every direction. 

    Parameters
    ----------
    a : float
        Dimension of base in x-axis
    b : float
        Dimension of base in y-axis
    h : float
        Height in z-direction 

    Returns
    -------
    gustaf.Bspline
        Volumetric linear spline in the shape of a pyramid
    """

    # Create box
    p = box(a, b, h)

    # Collapse all upper points on one control point
    p.control_points[np.isclose(p.control_points[:, -1], h)] = [a/2, b/2, h]

    return p


class Creator:
    """Helper class to build new splines from existing geometries.

    Examples
    ---------
    >>> myspline = <your-spline>
    >>> spline_faces = myspline.create.extrude(vector=[3,1,3])
    """

    def __init__(self, spl):
        self.spline = spl

    def extruded(self, *args, **kwargs):
        return extruded(self.spline, *args, **kwargs)

    def revolved(self, *args, **kwargs):
        return revolved(self.spline, *args, **kwargs)
