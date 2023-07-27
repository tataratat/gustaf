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


def export(mesh, fname, submeshes=None, **kwargs):
    """Export mesh elements and vertex data into meshio and use its write
    function. The definition of submeshes with identical vertex coordinates
    is possible. In that case vertex numbering and data from the main mesh
    are used. For more export options, refer to meshio's documentation
    https://github.com/nschloe/meshio .

    .. code-block:: python

        import gustaf
        # define coordinates
        v = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        # define triangle connectivity
        tf = np.array(
            [
                [1, 0, 2],
                [0, 1, 5],
                [3, 2, 6],
                [2, 0, 4],
                [4, 5, 7],
                [2, 3, 1],
                [7, 5, 1],
                [6, 7, 3],
                [4, 6, 2],
                [7, 6, 4],
            ]
        )
        # init tri faces
        mesh = gus.Faces(
            vertices=v,
            faces=tf,
        )
        gustaf.io.meshio.export(mesh, 'tri-mesh.stl')

    Parameters
    ------------
    mesh: Edges, Faces or Volumes
      Input mesh
    fname: Union[str, pathlib.Path]
      File to save the mesh in.
    submeshes: Iterable
      Submeshes where the vertices are identical to the main mesh. The element
      type can be identical to mesh.elements or lower-dimensional (e.g.
      boundary elements).
    **kwargs : Any
      Any additional argument will be passed to the respective meshio `write`
      function. See meshio docs for more information

    Raises
    -------
    NotImplementedError:
       For mesh types that are not implemented.
    ValueError:
       Raises a value error, if the vertices indexed in a subset are not
       present in the main mesh.
    """

    # Mapping between meshio cell types and gustaf cell types
    meshio_dict = {
        "edges": "line",
        "tri": "triangle",
        "quad": "quad",
        "tet": "tetra",
        "hexa": "hexahedron",
    }

    cells = []
    # Merge main mesh and submeshes in one list
    meshes = [mesh]
    if submeshes is not None:
        meshes.extend(submeshes)

    # Iterate the meshes and extract the element information in meshio format
    for m in meshes:
        whatami = m.whatami
        if whatami not in meshio_dict.keys():
            raise NotImplementedError(
                f"{whatami}-type meshes not supported (yet)."
            )
        else:
            if np.any(m.elements > len(m.vertices) - 1):
                raise ValueError("Invalid vertex IDs in submesh connectivity.")
            else:
                cells.append((meshio_dict[whatami], m.elements))

    # Export data to meshio and write file
    meshio.Mesh(
        points=mesh.vertices,
        cells=cells,
        point_data=mesh.vertex_data,
    ).write(fname, **kwargs)
