"""Import a meshio based mesh,

Export can only happen after it is possible to save and define boundaries in
gustaf.
"""
import pathlib

import numpy as np

from gustaf.faces import Faces
from gustaf.helpers.raise_if import ModuleImportRaiser
from gustaf.volumes import Volumes

try:
    import meshio
except ModuleNotFoundError as err:
    meshio = ModuleImportRaiser("meshio", err)
    # from meshio import Mesh as MeshioMesh


def load(fname):
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
    cells = meshio_mesh.get_cells_type("triangle")
    # If  no triangle elements, try for square
    if len(cells) == 0:
        cells = meshio_mesh.get_cells_type("quad")
    if not len(cells) > 0:
        # 3D mesh
        mesh_type = Volumes
        cells = meshio_mesh.get_cells_type("tetra")
        if len(cells) == 0:
            cells = meshio_mesh.get_cells_type("hexahedron")

    for i, cell in enumerate(cells):
        if i == 0:
            elements = cell.data
        else:
            elements = np.vstack((elements, cell.data))

    mesh = mesh_type(vertices=vertices, elements=elements)

    return mesh


def export(mesh, fname):
    """Export mesh elements and vertex data into meshio and use its write
    function.

    Parameters
    ------------
    mesh: Edges, Faces or Volumes
      Mesh to be exported.
    fname: Union[str, pathlib.Path]
      File to save the mesh in.

    Raises
    -------
    NotImplementedError: For mesh types that are not implemented.
    """

    # Mapping between meshio cell types and gustaf cell types
    meshio_dict = {
        "edges": "line",
        "tri": "triangle",
        "quad": "quad",
        "tet": "tetra",
        "hexa": "hexahedron",
    }
    whatami = mesh.whatami

    if whatami not in meshio_dict.keys():
        raise NotImplementedError(
            f"Sorry, we can't export {whatami}-shape with meshio."
        )
    else:
        cell_type = meshio_dict[whatami]

    mesh = meshio.Mesh(
        points=mesh.vertices,
        cells=[(cell_type, mesh.elements)],
        point_data=mesh.vertex_data,
    ).write(fname)
