"""Example showing the import of a gmsh file via meshio.

This example shows how a gmsh file can be imported into gustaf with the meshio
io module.
"""
import pathlib

from gustaf import io

if __name__ == "__main__":

    base_samples_path = pathlib.Path("samples/faces")
    if not base_samples_path.exists():
        raise RuntimeError(
                "The geometries could not be found. Please initialize the "
                "samples submodule, instructions can be found in the "
                "README.md."
        )
    mesh_file_tri = pathlib.Path("samples/faces/tri/2DChannelTria.msh")
    mesh_file_quad = pathlib.Path("samples/faces/quad/2DChannelQuad.msh")

    # load the .msh file directly with the correct io module (meshio)
    loaded_mesh_tri = io.meshio.load(mesh_file_tri)

    loaded_mesh_tri.show()

    # load the .msh file directly with the correct io module (meshio)
    loaded_mesh_quad = io.meshio.load(mesh_file_quad)

    loaded_mesh_quad.show()

    # load the .msh file with the default load function which needs to find out
    # it self which module is the correct one.
    loaded_mesh_default = io.load(mesh_file_tri)

    loaded_mesh_default.show()
