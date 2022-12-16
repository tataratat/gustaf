"""Import a meshio based mesh,

Export can only happen after it is possible to save and define boundaries in
gustaf.
"""
import pathlib
from typing import Union

import numpy as np

from gustaf._typing import MESH_TYPES
from gustaf.helpers.raise_if import ModuleImportRaiser
from gustaf.faces import Faces
from gustaf.volumes import Volumes

try:
    import meshio
except ModuleNotFoundError as err:
    meshio = ModuleImportRaiser("meshio", err)
    # from meshio import Mesh as MeshioMesh


def load(fname: Union[str, pathlib.Path]) -> MESH_TYPES:
    """Load mesh in meshio format. Loads vertices and their connectivity.
    Currently cannot process boundary.

    Note
    -----
    This is more or less a direct copy from the original gustav implementation.
    A lot of the meshio information are lost. When boundaries and multi-patch
    definitions are added this needs to be revisited and extended.

    Parameters
    ------------
    fname: str | pathlib.Path

    Returns
    --------
    MESH_TYPES
    """
    mesh_type = Faces

    fname = pathlib.Path(fname)
    if not (fname.exists() and fname.is_file()):
        raise ValueError(
                "The given file does not point to file. The given path is: "
                f"{fname.resolve()}"
        )

    meshio_mesh: meshio.Mesh = meshio.read(fname)
    vertices = meshio_mesh.points

    # check for 2D mesh
    # Try for triangle grid
    cells = meshio_mesh.get_cells_type('triangle')
    # If  no triangle elements, try for square
    if len(cells) == 0:
        cells = meshio_mesh.get_cells_type('quad')
    if not len(cells) > 0:
        # 3D mesh
        mesh_type = Volumes
        cells = meshio_mesh.get_cells_type('tetra')
        if len(cells) == 0:
            cells = meshio_mesh.get_cells_type('hexahedron')

    for i, cell in enumerate(cells):
        if i == 0:
            elements = cell.data
        else:
            elements = np.vstack((elements, cell.data))

    mesh = mesh_type(vertices=vertices, elements=elements)

    return mesh


def export(mesh: MESH_TYPES, fname: Union[str, pathlib.Path]):
    """Currently not implemented function.

    Parameters
    ------------
    mesh: MESH_TYPES
      Mesh to be exported.
    fname: Union[str, pathlib.Path]
      File to save the mesh in.

    Raises
    -------
    NotImplementedError: This method is currently not implemented.
    """
    raise NotImplementedError
