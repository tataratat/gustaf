"""Example showing the import of a gmsh file via meshio.

This example shows how a gmsh file can be imported into gustaf with the meshio
io module.
"""

import pathlib

import load_sample_file

import gustaf
from gustaf import io


def example():
    mesh_file_tri = pathlib.Path("faces/tri/2DChannelTria.msh")
    mesh_file_quad = pathlib.Path("faces/quad/2DChannelQuad.msh")
    mesh_file_tetra = pathlib.Path("volumes/tet/3DBrickTet.msh")

    base_samples_path = pathlib.Path(__file__).parent / "samples"
    load_sample_file.load_sample_file(str(mesh_file_tri))
    load_sample_file.load_sample_file(str(mesh_file_quad))
    load_sample_file.load_sample_file(str(mesh_file_tetra))

    # load the .msh file directly with the correct io module (meshio)
    loaded_mesh_tri = io.meshio.load(base_samples_path / mesh_file_tri)

    gustaf.show(loaded_mesh_tri)

    # load the .msh file directly with the correct io module (meshio)
    loaded_mesh_quad = io.meshio.load(base_samples_path / mesh_file_quad)

    gustaf.show(loaded_mesh_quad)

    # load the .msh file with the default load function which needs to find out
    # it self which module is the correct one.
    loaded_mesh_default = io.load(base_samples_path / mesh_file_tetra)

    gustaf.show(
        *[[msh.__class__.__name__, msh] for msh in loaded_mesh_default],
        title="3D mesh with tetrahedrons",
    )


if __name__ == "__main__":
    example()
