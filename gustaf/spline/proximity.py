"""
gustaf/spline/proximity.py

Closest what?
"""


def closest_control_points(
        spline,
        query_points,
        return_distances=False,
):
    """
    Returns indices of closest control points

    Parameters
    -----------
    spline: BSpline or NURBS
    query_points: (n, spline.dim) np.ndarray
      float
    return_distances: bool
      
    tolerance: float
      Default is settings.Tolerance. Only relevant iff nearest_only==False

    Returns
    --------
    indices: (n,) np.ndarray
    distances: (n, spline.dim) np.ndarray
      iff return_distances==True
    """
    from scipy.spatial import cKDTree as KDTree

    kdt = KDTree(spline.control_points)

    dist, ids = kdt.query(query_points)

    if return_distances:
        return ids, dist

    else:
        return ids


def closest_parametric_coordinate(
        spline,
        query_points,
):
    """
    Finds closest points using 
    """
    pass

class _Proximity:
    """
    Helper class to allow direct proximity queries for spline obj (BSpline or
    NURBS).
    Internal use only.

    Examples
    ---------
    >>> myspline = <your-spline>
    >>> closest_cp_ids = myspline.proximity.closest_control_points(queries)
    """

    def __init__(self, spl):
        self.spline = spl

    def closest_control_points(self, *args, **kwargs):
        return closest_control_points(self.spline, *args, **kwargs)
