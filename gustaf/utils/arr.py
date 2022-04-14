"""gustaf/gustaf/utils/arr.py

Useful functions for array / point operations.
Named `arr`, since `array` is python library and it sounds funny.
"""

import numpy as np

from gustaf import settings


def make_c_contiguous(array, dtype=None):
    """
    Make given array like object a c contiguous np.ndarray.
    dtype is optional. If None is given, just returns None.

    Parameters
    -----------
    array: array-like
    dtype: type or str
      (Optional) `numpy` interpretable type or str, describing type.
      Difference is that type will always return a copy and str will only copy
      if types doesn't match.

    Returns
    --------
    c_contiguous_array: np.ndarray
    """
    if array is None:
        return None

    if isinstance(array, np.ndarray):
        if array.flags.c_contiguous:
            if dtype is not None:
                if isinstance(dtype, type):
                    return array.astype(dtype)

                elif isinstance(dtype, str):
                    if array.dtype.name != dtype:
                        return array.astype(dtype)

            return array

    if dtype:
        return np.ascontiguousarray(array, dtype=dtype)

    else:
        return np.ascontiguousarray(array)


def unique_rows(
        in_arr,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        dtype_name=None,
):
    """
    Find unique rows using np.unique, but apply tricks. Adapted from
    `skimage.util.unique_rows`.
    url: github.com/scikit-image/scikit-image/blob/main/skimage/util/unique.py

    Parameters
    -----------
    in_arr: (n, m) 2D array-like
    return_index: bool
    return_inverse: bool
    return_counts: bool
    dtype_name: str

    Returns
    --------
    unique_arr: (p, q) np.ndarray
    unique_ind: (w,) np.ndarray
    unique_inv: (t,) np.ndarray
    """
    if dtype_name is None:
        dtype_name = settings.INT_DTYPE

    in_arr = make_c_contiguous(in_arr, dtype_name)

    if len(in_arr.shape) != 2:
        raise ValueError(
            "unique_rows can be only applied for 2D arrays"
        )

    in_arr_row_view = in_arr.view(f"|S{in_arr.itemsize * in_arr.shape[1]}")

    unique_stuff = np.unique(
        in_arr_row_view,
        return_index=True,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    unique_stuff = list(unique_stuff) # list, to allow item assignment

    # switch view to original
    unique_stuff[0] = in_arr[unique_stuff[1]]
    if not return_index:
        # pop return index
        unique_stuff.pop(1)

    return unique_stuff


def bounds(arr):
    """
    Return bounds.

    Parameters
    -----------
    arr: (n, d) array-like

    Returns
    --------
    bounds: (2, d) np.ndarray
    """
    return np.vstack(
        (
            np.min(arr, axis=0).reshape(1, -1),
            np.max(arr, axis=0).reshape(1, -1),
        )
    )


def bounds_diagonal(arr):
    """
    Returns diagonal vector of the bounds.
    bounds[1] - bounds[0]

    Parameters
    -----------
    arr: (n, d) array-like

    Returns
    --------
    bounds_diagonal: (n,) np.ndarray
    """
    b = bounds(arr)
    return b[1] - b[0]


def bounds_norm(arr):
    """
    Returns norm of the bounds.

    Parameters
    -----------
    arr: (n, d) array-like

    Returns
    --------
    bounds_norm: float
    """
    return np.linalg.norm(bounds_diagonal(arr))


def bounds_mean(arr):
    """
    Returns mean of the bounds

    Parameters
    -----------
    arr: (n, d) array-like

    Returns
    --------
    bounds_mean: (n,) array-like
    """
    return np.mean(bounds(arr), axis=0)


def select_with_ranges(arr, ranges):
    """
    Select array with ranges of each column.
    Always parsed as:
    [[greater_than, less_than], [....], ...]

    Parameters
    -----------
    ranges: (d, 2) array-like
      Takes None.

    Returns
    --------
    ids: (n,) np.ndarray
    """
    masks = []
    for i, r in enumerate(ranges):
        if r is None:
            continue

        else:
            lower = arr[:, i] > r[0]
            upper = arr[:, i] < r[1]
            if r[1] > r[0]:
                masks.append(np.logical_and(lower, upper))
            else:
                masks.append(np.logical_or(lower, upper))

    if len(masks) > 1:
        mask = np.zeros(arr.shape[0], dtype=bool)
        for i, m in enumerate(masks):
            if i == 0:
                mask = np.logical_or(mask, m)
            else:
                mask = np.logical_and(mask, m)

    else:
        mask = masks[0]

    return np.arange(arr.shape[0])[mask]


def rotation_matrix(rotation, degree=True):
    '''
    Compute rotation matrix.
    Works for both 2D and 3D point sets.
    In 2D, it can rotate along the (virtual) z-axis.
    In 3D, it can rotate along [x, y, z]-axis.
    Uses `scipy.spatial.transform.Rotation`.

    Parameters
    -----------
    rotation: list or float
      Amount of rotation along [x,y,z] axis. Default is in degrees.
      In 2D, it can be float.
    degree: bool
      (Optional) rotation given in degrees.
      Default is `True`. If `False`, in radian.

    Returns
    --------
    rotation_matrix: np.ndarray (3,3)
    '''
    from scipy.spatial.transform import Rotation as R

    rotation = np.asarray(rotation).flatten()

    if degree == True:
        rotation = np.radians(rotation)

    # 2D
    if len(rotation) == 1:
        return R.from_rotvec([0, 0, rotation]).as_matrix()[:2, :2]

    # 3D
    elif len(rotation) == 3:
        return R.from_rotvec(rotation).as_matrix()


def rotate(arr, rotation, rotation_axis=None, degree=True):
    """
    Rotates given arrays.
    Arrays shape[1] should equal to either 2 or 3
    For more information, see `rotation_matrix()`.

    Parameters
    -----------
    arr: (n, (2 or 3)) list-like
    rotation: list or float
    rotation_axis: (n, (2 or 3)) list-like

    Returns
    --------
    rotated_points: (n, d) np.ndarray
    """
    arr = make_c_contiguous(arr, settings.FLOAT_DTYPE)
    if rotation_axis is not None:
        rotation_axis = make_c_contiguous(
            rotation_axis,
            settings.FLOAT_DTYPE
        ).ravel()

    if rotation_axis is None:
        return np.matmul(
            arr,
            rotation_matrix(rotation, degree)
        )

    else:
        rarr = arr - rotation_axis
        rarr = np.matmul(rarr, rotation_matrix(rotation, degree))
        rarr += rotation_axis

        return rarr
