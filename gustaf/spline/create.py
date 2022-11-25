"""gustaf/spline/create.py.

Creates splines.
"""

import numpy as np
from splinepy.spline import RequiredProperties

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
    if "knot_vectors" in RequiredProperties.of(spline):
        spline_dict["knot_vectors"] = spline.knot_vectors + [[0, 0, 1, 1]]
    if "weights" in RequiredProperties.of(spline):
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

    if "knot_vectors" in RequiredProperties.of(spline):
        kv = [0, 0, 0]
        [kv.extend([i + 1, i + 1]) for i in range(n_knot_spans - 1)]
        spline_dict["knot_vectors"] = spline.knot_vectors + [
                kv + [n_knot_spans + 1] * 3
        ]
    if "weights" in RequiredProperties.of(spline):
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


def line(points):
    """Create a spline with the provided points as control points.

    Parameters
    ----------
    points: (n, d) numpy.ndarray
      npoints x ndims array of control points

    Returns
    -------
    line: BSpline
      Spline degree [1].
    """
    from gustaf import BSpline

    # lines have degree 1
    degree = 1

    cps = np.array(points)
    nknots = cps.shape[0] + degree + 1

    knots = np.concatenate(
            (
                    np.full(degree, 0.),
                    np.linspace(0., 1., nknots - 2 * degree),
                    np.full(degree, 1.0),
            )
    )

    spline = BSpline(
            control_points=cps, knot_vectors=[knots], degrees=[degree]
    )

    return spline


def arc(radius=1., angle=90., n_knot_spans=None, start_angle=0., degree=True):
    """Creates a 1-D arc as Rational Bezier or NURBS with given radius and
    angle. The arc lies in the x-y plane and rotates around the z-axis.

    Parameters
    ----------
    radius : float, optional
      radius of the arc, defaults to 1
    angle : float, optional
      angle of the section of the arc, defaults to 90 degrees
    n_knot_spans : int
      Number of knot spans, by default minimum number for angle is used.
    start_angle : float, optional
      starting point of the angle, by default 0.
    degree: bool, optional
      degrees for angle used, by default True

    Returns
    -------
    arc: NURBS or RationalBezier
    """
    from gustaf import RationalBezier

    # Define point spline of degree 0 at starting point of the arc
    if degree:
        start_angle = np.radians(start_angle)
        angle = np.radians(angle)
    start_point = [radius * np.cos(start_angle), radius * np.sin(start_angle)]
    point_spline = RationalBezier(
            degrees=[0], control_points=[start_point], weights=[1.]
    )
    # Bezier splines only support angles lower than 180 degrees
    if abs(angle) >= np.pi or n_knot_spans > 1:
        point_spline = point_spline.nurbs

    # Revolve
    arc_attribs = point_spline.create.revolved(
            angle=angle, n_knot_spans=n_knot_spans, degree=degree
    ).todict()
    # Remove the first parametric dimenions, which is only a point and
    # only used for the revolution
    arc_attribs['degrees'] = list(arc_attribs['degrees'])[1:]
    if "knot_vectors" in RequiredProperties.of(point_spline):
        arc_attribs['knot_vectors'] = list(arc_attribs['knot_vectors'])[1:]

    return type(point_spline)(**arc_attribs)


def circle(radius=1., n_knot_spans=3):
    """Circle (parametric dim = 1) with radius r in the x-y plane around the
    origin. The spline has an open knot vector and degree 2.

    Parameters
    ----------
    radius : float, optional
      radius, defaults to one
    n_knots_spans : int, optional
      number of knot spans, defaults to 3

    Returns
    -------
    circle: NURBS
    """
    return arc(radius=radius, angle=360, n_knot_spans=n_knot_spans)


def box(*lengths):
    """ND box (hyperrectangle).

    Parameters
    ----------
    *lengths: list(float)

    Returns
    -------
    ndbox: Bezier
    """
    from gustaf import Bezier

    # may dim check here?

    # starting point
    ndbox = Bezier(degrees=[1], control_points=[[0], [lengths[0]]])
    # use extrude
    for i, l in enumerate(lengths[1:]):
        ndbox = ndbox.create.extruded([0] * int(i + 1) + [l])

    return ndbox


def plate(radius=1.):
    """Creates a biquadratic 2-D spline in the shape of a plate with given
    radius.

    Parameters
    ----------
    radius : float, optional
      Radius of the plate, defaults to one

    Returns
    -------
    plate: RationalBezier
    """
    from gustaf.spline import RationalBezier

    degrees = [2, 2]
    control_points = np.array(
            [
                    [-0.5, -0.5],
                    [0., -1.],
                    [0.5, -0.5],
                    [-1., 0.],
                    [0., 0.],
                    [1., 0.],
                    [-0.5, 0.5],
                    [0., 1.],
                    [0.5, 0.5],
            ]
    ) * radius
    weights = np.tile([1., 1 / np.sqrt(2)], 5)[:-1]

    return RationalBezier(
            degrees=degrees, control_points=control_points, weights=weights
    )


def disk(
        outer_radius,
        inner_radius=None,
        angle=360.,
        n_knot_spans=None,
        degree=True
):
    """Surface spline describing a potentially hollow disk with quadratic
    degree along curved dimension and linear along thickness. The angle
    describes the returned part of the disk.

    Parameters
    ----------
    outer_radius : float
      Outer radius of the disk
    inner_radius : float, optional
      Inner radius of the disk, in case of hollow disk, by default 0.
    angle : float, optional
      Rotational angle, by default 360. describing a complete revolution
    n_knot_spans : int, optional
      Number of knot spans, by default 4

    Returns
    -------
    disk: NURBS
      Surface NURBS of degrees (1,2)
    """

    from gustaf.spline import NURBS

    if inner_radius is None:
        inner_radius = 0.

    cps = np.array([[inner_radius, 0.], [outer_radius, 0.]])
    weights = np.ones([cps.shape[0]])
    knots = np.repeat([0., 1.], 2)

    return NURBS(
            control_points=cps,
            knot_vectors=[knots],
            degrees=[1],
            weights=weights
    ).create.revolved(angle=angle, n_knot_spans=n_knot_spans, degree=degree)


def torus(
        torus_radius,
        section_outer_radius,
        section_inner_radius=None,
        torus_angle=None,
        section_angle=None,
        section_n_knot_spans=4,
        torus_n_knot_spans=4,
        degree=True,
):
    """Creates a volumetric NURBS spline describing a torus revolved around the
    x-axis. Possible cross-sections are plate, disk (yielding a tube) and
    section of a disk.

    Parameters
    ----------
    torus_radius : float
      Radius of the torus
    section_outer_radius : float
      Radius of the section of the torus
    section_inner_radius : float, optional
      Inner radius in case of hollow torus, by default 0.
    torus_angle : float, optional
      Rotational anlge of the torus, by default None, giving a complete
      revolution
    section_angle : float, optional
      Rotational angle, by default None, yielding a complete revolution
    section_n_knot_spans : float, optional
      Number of knot spans along the cross-section, by default 4
    torus_n_knot_spans : float, optional
      Number of knot spans along the torus, by default 4

    Returns
    -------
    torus: NURBS
      Volumetric spline in the shape of a torus with degrees (1,2,2)
    """

    if torus_angle is None:
        torus_angle = 2 * np.pi
        degree = False

    if section_angle is None:
        section_angle = 2 * np.pi
        section_angle_flag = False
        degree = False
    else:
        section_angle_flag = True

    if section_inner_radius is None:
        section_inner_radius = 0
        section_inner_radius_flag = False
    else:
        section_inner_radius_flag = True

    # Create the cross-section
    if not section_angle_flag and not section_inner_radius_flag:
        cross_section = plate(section_outer_radius)
        # For more than 180 degree only NURBS can be used
        if abs(torus_angle) >= np.pi:
            cross_section = cross_section.nurbs
    else:
        cross_section = disk(
                outer_radius=section_outer_radius,
                inner_radius=section_inner_radius,
                n_knot_spans=section_n_knot_spans,
                angle=section_angle,
                degree=degree
        )

    # Create a surface spline representing a disk and move it from the origin
    cross_section.control_points[:, 1] += torus_radius

    return cross_section.create.revolved(
            axis=[1., 0, 0],
            center=np.zeros(3),
            angle=torus_angle,
            n_knot_spans=torus_n_knot_spans,
            degree=degree
    )


def sphere(
        outer_radius,
        inner_radius=None,
        angle=360.,
        n_knot_spans=None,
        degree=True
):
    """Creates a volumetric spline describing a sphere with radius R.

    Parameters
    ----------
    outer_radius : float
      Outer radius of the sphere
    inner_radius : float, optional
      Inner radius of the potentially hollow sphere.
    angle : float
      Rotational angle around x-axis, by default each 360
      (describing a complete revolution)
    n_knot_spans : int
      Number of knot spans

    Returns
    -------
    sphere: NURBS
      Volumetric NURBS with degrees (1,2,2)
    """

    if inner_radius is None:
        sphere = plate(outer_radius).nurbs.create.revolved(
                axis=[1, 0, 0],
                center=[0, 0, 0],
                angle=angle,
                n_knot_spans=n_knot_spans,
                degree=degree
        )
    else:
        inner_radius = float(inner_radius)
        sphere = disk(outer_radius, inner_radius).nurbs.create.revolved(
                angle=angle, n_knot_spans=n_knot_spans, degree=degree
        )
    return sphere


def cone(
        outer_radius,
        height,
        inner_radius=None,
        volumetric=True,
        angle=360.,
        degree=True
):
    """Creates a cone with circular base.

    Parameters
    ----------
    radius : float
      Radius of the base
    height : float
      Height of the cone
    volumetric : bool, optional
      Parameter whether surface or volume spline, by default True
    angle : float
      Rotation angle in degrees, only used for solid model

    Returns
    -------
    cone: NURBS
      Volumetric or surface NURBS descibing a cone
    """

    if volumetric:
        ground = disk(
                outer_radius,
                inner_radius=inner_radius,
                angle=angle,
                degree=degree
        )
    else:
        ground = circle(outer_radius)

    # Extrude in z
    cone = ground.create.extruded([0, 0, height])
    # Move all upper control points to one
    cone.control_points[np.isclose(cone.control_points[:, -1],
                                   height)] = [0, 0, height]

    return cone


def pyramid(width, length, height):
    """Creates a volumetric spline in the shape of a pyramid with linear degree
    in every direction.

    Parameters
    ----------
    width : float
      Dimension of base in x-axis
    lenght : float
      Dimension of base in y-axis
    height : float
      Height in z-direction

    Returns
    -------
    pyramid: BSpline
      Volumetric linear spline in the shape of a pyramid
    """

    # Create box
    p = box(width, length, height)

    # Collapse all upper points on one control point
    p.control_points[4:, :] = [width / 2, length / 2, height]

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
