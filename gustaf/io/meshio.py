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

    set_boundary = True
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


def export(mesh, fname):
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
