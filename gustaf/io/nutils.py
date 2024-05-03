"""gustaf/gustaf/io/nutils.py.

io functions for nutils.
"""

import numpy as np

from gustaf.faces import Faces
from gustaf.io import mixd
from gustaf.io.ioutils import abs_fname, check_and_makedirs
from gustaf.volumes import Volumes


def load(fname):
    """nutils load.
    Loads a nutils (np.savez) file and returns a Gustaf Mesh.

    Parameters
    -----------
    fname: str
      The npz file needs the following keys:
      nodes, cnodes, coords, tags, btags, ptags.

    Returns
    --------
    mesh: Faces or Volumes
    """
    npz_file = np.load(fname, allow_pickle=True)
    nodes = npz_file["nodes"]
    _ = npz_file["cnodes"]
    coords = npz_file["coords"]
    _ = npz_file["tags"].item()
    btags = npz_file["btags"].item()
    _ = npz_file["ptags"].item()

    if nodes.shape[0] == 0:
        raise TypeError("Can not find nodes. Check nutils mesh description.")
    if coords.shape[0] == 0:
        raise TypeError("Can not find coords. Check nutils mesh description.")

    vertices = coords

    # connec
    simplex = True
    connec = nodes

    volume = vertices.shape[1] != 2

    # reshape connec
    try:
        ncol = 3 if simplex and not volume else 4
        connec = connec.reshape(-1, ncol)
        mesh = Volumes(vertices, connec) if volume else Faces(vertices, connec)
    except BaseException:
        raise RuntimeError(
            "Can not generate a mesh from the nutils input."
            "Check nutils mesh description."
        )

    mesh.BC = btags
    return mesh


def export(fname, mesh):
    """Export in Nutils format. Files are saved as np.savez().
    Supports triangle,and tetrahedron Meshes.

    Parameters
    -----------
    fname: str
    mesh: Faces or Volumes

    Returns
    --------
    None
    """

    dic = to_nutils_simplex(mesh)

    # prepare export location
    fname = abs_fname(fname)
    check_and_makedirs(fname)

    np.savez(fname, **dic)


def to_nutils_simplex(mesh):
    """Converts a Gustaf_Mesh to a Dictionary, which can be interpreted
    by ``nutils.mesh.simplex(**to_nutils_simplex(mesh))``. Only works for
    Triangles and Tetrahedrons!

    Parameters
    -----------
    mesh: Faces or Volumes

    Returns
    --------
    dic_to_nutils: dict
    """

    vertices = mesh.vertices
    faces = mesh.faces
    whatami = mesh.whatami

    # In 2D, element = face. In 3D, element = volume.
    if whatami.startswith("tri"):
        dimension = 2
        permutation = [1, 2, 0]
        elements = faces
    elif whatami.startswith("tet"):
        dimension = 3
        permutation = [2, 3, 1, 0]
        volumes = mesh.volumes
        elements = volumes
    else:
        raise TypeError("Only Triangle and Tetrahedrons are accepted.")

    dic_to_nutils = {}

    # Sort the Node IDs for each Element.
    sort_array = np.argsort(elements, axis=1)
    elements_sorted = np.take_along_axis(elements, sort_array, axis=1)

    # Let`s get the Boundaries
    bcs = {}
    bcs_in = mixd.make_mrng(mesh)
    bcs_in = np.ndarray.reshape(
        bcs_in, (int(len(bcs_in) / (dimension + 1)), (dimension + 1))
    )

    bound_id = np.unique(bcs_in)
    bound_id = bound_id[bound_id > 0]

    # Reorder the mrng according to nutils permutation: swap columns
    bcs_in[:, :] = bcs_in[:, permutation]

    # Let's reorder the bcs file with the sort_array
    bcs_sorted = np.take_along_axis(bcs_in, sort_array, axis=1)

    for bi in bound_id:
        bcs[str(bi)] = np.argwhere(bcs_sorted == bi)

    dic_to_nutils.update(
        {
            "nodes": elements_sorted,
            "cnodes": elements_sorted,
            "coords": vertices,
            "tags": {},
            "btags": bcs,
            "ptags": {},
        }
    )

    return dic_to_nutils
