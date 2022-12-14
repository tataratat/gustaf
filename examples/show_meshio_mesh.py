"""Example showing the import of a gmsh file via meshio.

This example shows how a gmsh file can be imported into gustaf with the meshio
io module.
"""
from gustaf import io

if __name__ == "__main__":

    mesh_file = "data/2DChannelTria.msh"

    # load the .msh file directly with the correct io module (meshio)
    loaded_mesh = io.meshio.load(mesh_file)

    loaded_mesh.show()

    # load the .msh file with the default load function which needs to find out
    # it self which module is the correct one.
    loaded_mesh_default = io.load(mesh_file)

    loaded_mesh_default.show()
