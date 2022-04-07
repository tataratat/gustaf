"""gustaf/spline/_utils.py

Utils for spline. Internal use only.
"""

def to_res_list(res, length):
    """
    Given int or list, returns a list of resolutions of correct length.
    If list is given, checks if it is a correct length.

    Parameters
    -----------
    res: int or list
    length: int
      length of desired res_list. Usually corresponds to spline.para_dim

    Returns
    --------
    res_list: list
    """
    if isinstance(res, list):
        if len(res) != length:
            raise ValueError(
                "Invalid resolution length! "
                + "It should match length."
            )

        return res

    elif isinstance(res, (int, float)):
        res_list = [res for _ in range(length)]
        return res_list

    elif isinstance(res, tuple):
        return to_res_list(list(res), length)

    else:
        raise TypeError(
            "Invalid resolutions input. It should be int, tuple, or list."
        )
