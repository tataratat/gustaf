"""gustaf/gustaf/io/mixd.py.

io functions for mixd.
"""

import os
import struct

import numpy as np

from gustaf.faces import Faces
from gustaf.io.ioutils import abs_fname, check_and_makedirs
from gustaf.utils import log
from gustaf.volumes import Volumes


def load(
    simplex=True, volume=False, fname=None, mxyz=None, mien=None, mrng=None
):
    """mixd load. To avoid reading minf, all the crucial info can be given as
    params. Default input will try to import `mxyz`, `mien`, `mrng` from
    current location and assumes mesh is 2D triangle.

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

    Returns
    --------
    mesh: Faces or Volumes
    """
    # figure out input type
    specified_input = mxyz is not None  # bare minimum input
    fname_input = (fname is not None) and not specified_input
    default_input = not (fname_input or specified_input)

    if default_input:
        mxyz = "mxyz"
        mien = "mien"
        mrng = "mrng"

    elif fname_input:
        fname = abs_fname(fname)
        fbase, ext = os.path.splitext(fname)

        if os.path.basename(fbase) == "_":
            fbase = fbase[:-1]
        else:
            fbase += "."

        mxyz = fbase + "mxyz"
        mien = fbase + "mien"
        mrng = fbase + "mrng"

    # vertices
    vertices = np.fromfile(mxyz, dtype=">d").astype(np.float64)

    # connec
    connec = None
    try:
        connec = (np.fromfile(mien, dtype=">i") - int(1)).astype(np.int32)
    except BaseException:
        log.debug(f"mien file, `{mien}`, does not exist. Skipping.")

    # boundary conditions
    bcs = dict()
    try:
        bcs_in = np.fromfile(mrng, dtype=">i").astype(np.int32)  # flattened
        uniq_bcs_in = np.unique(bcs_in)
        uniq_bcs_in = uniq_bcs_in[uniq_bcs_in > 0]  # keep only natural nums
        sub_elem_ids = np.arange(bcs_in.size)

        for ubci in uniq_bcs_in:
            bcs.update({str(ubci): sub_elem_ids[bcs_in == ubci]})

    except BaseException:
        log.debug(f"mrng file, `{mrng}`, does not exist. Skipping.")

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


def export(
    mesh,
    fname,
    space_time=False,
):
    """Export in mixd format. Supports triangle, quadrilateral, tetrahedron,
    and hexahedron semi-discrete and (flat) space-time mesh output.

    Parameters
    -----------
    mesh: Faces or Volumes
    fname: str
    space_time : bool
      Export Mesh as Space-Time Slab for discontinuous space-time

    Returns
    --------
    None
    """
    # did you give us an acceptable mesh?
    acceptable_shapes = ["tri", "quad", "tet", "hexa"]
    whatami = mesh.whatami

    if whatami not in acceptable_shapes:
        raise NotImplementedError(
            f"Sorry, we can't export {whatami}-shape in mixd format."
        )

    # prepare export location
    fname = abs_fname(fname)
    check_and_makedirs(fname)

    # basic infos
    dim = mesh.vertices.shape[1]
    big_endian_int = ">i"
    big_endian_double = ">d"

    # prep files
    fbase, ext = os.path.splitext(fname)

    if ext.startswith(".xns"):
        # frequently used case in practice. no base export
        if os.path.basename(fbase) == "_":
            fdir = os.path.dirname(fbase)
            vert_file = os.path.join(fdir, "mxyz")
            connec_file = os.path.join(fdir, "mien")
            bc_file = os.path.join(fdir, "mrng")
            info_file = os.path.join(fdir, "minf")

        else:
            fbase += "."
            vert_file = fbase + "mxyz"
            connec_file = fbase + "mien"
            bc_file = fbase + "mrng"
            info_file = fbase + "minf"

    else:
        raise NotImplementedError("`mixd` format only supports xns.")

    # write v
    with open(vert_file, "wb") as vf:
        for v in mesh.vertices.ravel():
            vf.write(struct.pack(big_endian_double, v))

        if space_time:
            for v in mesh.vertices.ravel():
                vf.write(struct.pack(big_endian_double, v))

    # write connec
    with open(connec_file, "wb") as cf:
        for c in mesh.elements.ravel() + 1:
            cf.write(struct.pack(big_endian_int, c))

    # write bc
    with open(bc_file, "wb") as bf:
        boundaries = make_mrng(mesh)

        for b in boundaries:
            bf.write(struct.pack(big_endian_int, b))

    # write info
    with open(info_file, "w") as infof:  # if and inf... just can't
        infof.write(f"# dim: {dim}\n")
        infof.write(f"# mesh type: {whatami}\n\n")

        st_factor = 2 if space_time else 1
        infof.write(f"nn {int(mesh.vertices.shape[0] * st_factor)}\n")
        infof.write(f"ne {int(mesh.elements.shape[0])}\n")
        infof.write(f"nsd {dim}\n")
        infof.write(f"nen {int(mesh.elements.shape[1] * st_factor)}\n")

        if space_time:
            infof.write("space-time on\n\n\n")
        else:
            infof.write("semi-discrete on\n\n\n")

        # BC guide
        infof.write("# boundary name : referenced number.\n")
        for i, bname in enumerate(mesh.BC.keys()):
            infof.write(f"# {bname} : {i + 1}\n")

        # signature
        infof.write("\n\n\n# MIXD generated using `gustaf`.\n")


def make_mrng(mesh):
    """
    Builds and return mrng array based on `mesh.BC`
    Supports `Faces` and `Volumes`.

    Parameters
    -----------
    mesh: Faces or Volumes
      Number of participating elements

    Returns
    --------
    boundaries : ndarray
      The mrng-array.
    """

    # determine number of sub elements
    whatami = mesh.whatami
    nbelem = 3

    if whatami.startswith("quad") or whatami.startswith("tet"):
        nbelem += 1
    elif whatami.startswith("hexa"):
        nbelem += 3

    # init boundaries with -1, as it is the value for non-boundary.
    # alternatively, they could be (-1 * neighbor_elem_id).
    # But they aren't.
    boundaries = np.empty(mesh.elements.shape[0] * nbelem, dtype=int)
    boundaries[:] = -1

    for i, belem_ids in enumerate(mesh.BC.values()):
        boundaries[belem_ids] = i + 1  # bid starts at 1

    return boundaries
