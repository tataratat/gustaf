"""Import a meshio based mesh,

Export can only happen after it is possible to save and define boundaries in
gustaf.
"""
import pathlib

import numpy as np

from gustaf.edges import Edges
from gustaf.faces import Faces
from gustaf.helpers.raise_if import ModuleImportRaiser
from gustaf.volumes import Volumes

try:
    import meshio
except ModuleNotFoundError as err:
    meshio = ModuleImportRaiser("meshio", err)
    # from meshio import Mesh as MeshioMesh


def load(fname, set_boundary=False, return_only_one_mesh=True):
    """Load mesh in meshio format. Loads vertices and their connectivity.

    The function reads all gustaf-processable element types from meshio, which
    are sorted by the dimensionality of their element types. Different
    elements can occur at the same dimensionality, e.g., in mixed meshes,
    or as sub-topologies, e.g., as boundaries or interfaces.

    If set_boundary is set true, all (d-1)-dimensional submesh node identifiers
    are assigned to MESH_TYPES.BC without check.

    Return_only_one_mesh ensures compatibility with previous gustaf versions.

    Loading physical groups is not supported yet.

    Parameters
    ------------
    fname: str | pathlib.Path
       Input filename
    set_boundary : bool, optional, default is False
       Assign nodes of lower-dimensional meshes mesh.BC
    return_only_one_mesh : bool, optional, default is True
        Return only the highest-dimensional mesh, ensures compartibility.

    Returns,
    --------
    MESH_TYPES or list(MESH_TYPES)
    """
    fname = pathlib.Path(fname)
    if not (fname.exists() and fname.is_file()):
        raise ValueError(
            "The given file does not point to file. The given path is: "
            f"{fname.resolve()}"
        )

    # Read mesh
    meshio_mesh: meshio.Mesh = meshio.read(fname)
    vertices = meshio_mesh.points

    # Maps meshio element types to gustaf types and dimensionality
    # Meshio vertex elements are ommited, as they do not follow gustaf logics
    meshio_mapping = {
        "hexahedron": [Volumes, 3],
        "tetra": [Volumes, 3],
        "quad": [Faces, 2],
        "triangle": [Faces, 2],
        "line": [Edges, 1],
    }
    meshes = [
        [
            meshio_mapping[key][0](
                vertices=vertices, elements=meshio_mesh.cells_dict[key]
            ),
            meshio_mapping[key][1],
        ]
        for key in meshio_mapping.keys()
        if key in meshio_mesh.cells_dict.keys()
    ]
    # Sort by decreasing dimensionality
    meshes.sort(key=lambda x: -x[1])

    # Raises exception if the mesh file contains gustaf-incompartible types
    for elem_type in meshio_mesh.cells_dict.keys():
        if elem_type not in list(meshio_mapping.keys()) + ["vertex"]:
            # Be aware that gustaf currently only supports linear element
            # types.
            raise NotImplementedError(
                f"Sorry, {elem_type} elements currently are not supported."
            )

    if set_boundary is True:
        # mesh.BC is only defined for volumes and faces
        for i in range(len(meshes)):
            # Indicator for boundary
            if meshes[i][1] in [3, 2]:
                for j in range(i, len(meshes)):
                    # Considers only (d-1)-dimensional subspaces
                    if meshes[j][1] == meshes[i][1] - 1:
                        meshes[i][0].BC[
                            f"{meshes[j][0].whatami}-nodes"
                        ] = np.unique(meshes[j][0].elements)

    # Ensures backwards-compartibility
    if return_only_one_mesh:
        # Return highest-dimensional mesh
        return_value = meshes[0][0]
    else:
        return_value = [mesh[0] for mesh in meshes]

    return return_value


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
