"""gustaf/gustaf/io/mfem.py

io functions for mixd.
"""

import numpy as np

from gustaf.vertices import Vertices
from gustaf.faces import Faces
from gustaf.volumes import Volumes
from gustaf.io.ioutils import abs_fname
from gustaf.utils import log


def load(
        simplex=True,
        volume=False,
        fname=None,
        mxyz=None,
        mien=None,
        mrng=None
):
    """
    mixd load.
    To avoid reading minf, all the crucial info can be given as params.
    Default input will try to import `mxyz`, `mien`, `mrng` from current
    location and assumes mesh is 2D triangle.

    Parameters
    -----------
    simplex: bool
      Default is True. Is it triangle based?
    volume: bool
      Default is False. Is it 3D?
    fname: str
      Default is None. Specify your mixd file names with ".xns" postfix.
      Ex) "gustaf.xns" will try to load "gustaf.mxyz", "gustaf.mien",
          and "gustaf.mrng"
    mxyz: str
      Default is None.
    mien: str
      Default is None.
    mrng: str
      Default is None. This is optional.
    """
    # figure out input type
    specified_input = mxyz != None # bare minimum input
    fname_input = (fname != None) and not specified_input
    default_input = not (fname_input or specified_input)

    if default_input:
        mxyz = "mxyz"
        mien = "mien"
        mrng = "mrng"

    elif fname_input:
        absfname = abs_fname(fname)
        base, ext = os.path.splitext(fname)

        mxyz = base + ".mxyz"
        mien = base + ".mien"
        mrng = base + ".mrng"

    # vertices
    vertices = np.fromfile(mxyz, dtype=">d").astype(np.float64)

    # connec
    connec = None
    try:
        connec = (np.fromfile(mien, dtype=">i") - int(1)).astype(np.int32)
    except:
        log.debug(f"mien file, `{mien}`, does not exist. Skipping.")

    # boundary conditions
    bcs = dict()
    try:
        bcs_in = np.fromfile(mrng, dtype=">i").astype(np.int32) # flattened
        for i in range(bcs_in.max()):
            if i < 1: # ignore bc numbers below 0
                continue

            globinds = np.where(bcs == i)[0] # returns tuple
            bcs.update({str(i) : globinds})
            
    except:
        log.debug(f"mien file, `{mien}`, does not exist. Skipping.")

    # reshape vertices
    vertices = vertices.reshape(-1, 3) if volume else vertices.reshape(-1, 2)

    # reshape connec
    if connec is not None:
        ncol = int(3) if simplex and not volume else int(4)
        ncol = int(8) if ncol == int(4) and volume and not simplex else ncol

        connec = connec.reshape(-1, ncol)

        mesh = Volumes(vertices, connec) if volume else Faces(vertices, connec)

    # bc
    if len(bcs) != 0:
        mesh.BC = bcs

    return mesh
