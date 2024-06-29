"""gustaf/gustaf/io/mixd.py.

io functions for mixd.
"""

import os
import struct

import numpy as np

from gustaf import settings
from gustaf.faces import Faces
from gustaf.io.ioutils import abs_fname, check_and_makedirs
from gustaf.utils import log
from gustaf.utils.arr import close_rows
from gustaf.volumes import Volumes

_big_endian_int = ">i"
_big_endian_double = ">d"


def load(
    simplex=True,
    volume=False,
    fname=None,
    mxyz=None,
    mien=None,
    mrng=None,
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
    vertices = np.fromfile(mxyz, dtype=_big_endian_double).astype(np.float64)

    # connec
    connec = None
    try:
        connec = (np.fromfile(mien, dtype=_big_endian_int) - 1).astype(
            np.int32
        )
    except BaseException:
        log.debug(f"mien file, `{mien}`, does not exist. Skipping.")

    # boundary conditions
    bcs = {}
    try:
        bcs_in = np.fromfile(mrng, dtype=_big_endian_int).astype(np.int32)
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
        ncol = 3 if simplex and not volume else 4
        ncol = 8 if ncol == 4 and volume and not simplex else ncol

        connec = connec.reshape(-1, ncol)

        mesh = Volumes(vertices, connec) if volume else Faces(vertices, connec)

    # bc
    if len(bcs) != 0:
        mesh.BC = bcs

    return mesh


def _ravel(array):
    """
    Ravel if it is an array. Else, just return
    """
    if isinstance(array, np.ndarray):
        return array.ravel()
    return array


def _write_raveled_int(file, array):
    """
    Writes raveled array to the file as big endian int.

    Parameters
    ----------
    file: _io.TextIOWrapper
      Objects for calling `open(fname)`
    array: np.ndarray
    """
    for v in _ravel(array):
        file.write(struct.pack(_big_endian_int, v))


def _write_raveled_double(file, array):
    """
    Writes raveled array to the file as big endian double.

    Parameters
    ----------
    file: _io.TextIOWrapper
      Objects for calling `open(fname)`
    array: np.ndarray
    """
    for v in _ravel(array):
        file.write(struct.pack(_big_endian_double, v))


def export(
    fname,
    mesh,
    space_time=False,
    dual=False,
):
    """Export in mixd format. Supports triangle, quadrilateral, tetrahedron,
    and hexahedron semi-discrete and (flat) space-time mesh output.

    Parameters
    -----------
    mesh: Faces or Volumes
    fname: str
    space_time: bool
      Export Mesh as Space-Time Slab for discontinuous space-time
    dual: bool
      Includes dual-subelement information.

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
            dual_file = os.path.join(fdir, "dual")

        else:
            fbase += "."
            vert_file = fbase + "mxyz"
            connec_file = fbase + "mien"
            bc_file = fbase + "mrng"
            info_file = fbase + "minf"
            dual_file = fbase + "dual"

    else:
        raise NotImplementedError("`mixd` format only supports xns.")

    # write v
    with open(vert_file, "wb") as vf:
        _write_raveled_double(vf, mesh.vertices)

        # write it one more time for Discontinuous Prismic Space Time meshes
        # used to be called Flat Space Time mesh.
        if space_time:
            _write_raveled_double(vf, mesh.vertices)

    # write connec
    with open(connec_file, "wb") as cf:
        # let's not forget fortran numbering (+ 1)
        _write_raveled_int(cf, mesh.elements.ravel() + 1)

    # get boundaries of each element - subelement interface array
    sub_interface = make_mrng(mesh)

    # write bc first - after writing it, we can modify inplace for dual.
    with open(bc_file, "wb") as bf:
        _write_raveled_int(bf, sub_interface)

    # if dual is True, we fill dual infos.
    if dual:
        sub_elements = mesh.to_subelements(False)
        # this should be always dividable without remnants
        n_subelem_per_elem, rem = divmod(
            len(sub_elements.elements), len(mesh.elements)
        )
        if rem != 0:
            raise ValueError(
                "something went wrong with subelement creation."
                "Please report this issue, thank you!"
            )

        # get intersection - can use this info to determine duals
        _, _, _, intersections = close_rows(
            sub_elements.centers(), settings.TOLERANCE, True
        )

        # loop intersections and look for 2 intersections
        for i, intersection in enumerate(intersections):
            n_inter = len(intersection)

            # we modify interface only if there're 2 intersections.
            if n_inter == 2:
                # intersection is always sorted.
                # we don't want dual to point to itself
                dual_id = 0 if i != intersection[0] else 1

                # get element number and apply fortran's offset, 1
                sub_interface[i] = -int(
                    intersection[dual_id] // n_subelem_per_elem + 1
                )
                continue

            # intersection should be at most 2. Otherwise, it either means
            # that you have a bad mesh or to big tolerance
            if n_inter > 2:
                raise ValueError(
                    f"{i}-th subelement overlaps more than once. "
                    "Please check your elements or decrease "
                    "gustaf.settings.TOLERANCE."
                )

        # write dual
        with open(dual_file, "wb") as df:
            _write_raveled_int(df, sub_interface)

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

    # init boundaries with 0, as boundaries are marked with positive numbers
    # staring from 1 and dual infos with negative values, starting with -1
    boundaries = np.empty(mesh.elements.shape[0] * nbelem, dtype=int)
    boundaries[:] = 0

    for i, belem_ids in enumerate(mesh.BC.values()):
        boundaries[belem_ids] = i + 1  # bid starts at 1

    return boundaries
