"""gustaf/gustaf/io/ioutils.py.

utils for lord load and his expected expertise, export.
"""

import os


def abs_fname(fname):
    """Checks if fname is abs. If not, returns abs. Tilde safe.

    Parameters
    -----------
    fname: str

    Returns
    --------
    abs_fname: str
      Maybe same as fname, maybe not.
    """
    if os.path.isabs(fname):
        pass
    # elif fname.startswith("~"):
    elif "~" in fname:
        fname = os.path.expanduser(fname)
    else:
        fname = os.path.abspath(fname)

    return fname


def check_and_makedirs(fname):
    """Checks if the directories of the path exists. If not, makedirs!

    Parameters
    -----------
    fname: str

    Returns
    --------
    None
    """
    dirs = os.path.dirname(fname)

    if dirs == "":
        return None

    if not os.path.isdir(dirs):
        os.makedirs(dirs)

    return None
