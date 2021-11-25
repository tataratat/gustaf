"""gustav/gustav/utils/arr.py

Useful functions for array / point operations.
Named `arr`, since `array` is python library and it sounds funny.
"""

import numpy as np

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
        dtype_name="int32",
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
    in_arr = make_c_contiguous(in_arr, dtype_name)

    if len(in_arr.shape) != 2:
        raise ValueError(
            "unique_rows can be only applied for 2D arrays"
        )

    in_arr_row_view = in_arr.view(f"|S{in_arr.itemsize * in_arr.shape[1]}")

    return np.unique(
        in_arr_row_view,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )


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
            np.min(a, axis=0).reshape(1, -1),
            np.max(a, axis=0).reshape(1, -1),
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
    bounds = bounds(arr)
    return bounds[1] - bounds[0]


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
