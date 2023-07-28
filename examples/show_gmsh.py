"""Example showing the import of a gmsh file via meshio.

This example shows how a gmsh file can be imported into gustaf with the meshio
io module.
"""
import pathlib

import load_sample_file

from gustaf import Vertices, io, show

if __name__ == "__main__":
    mesh_file_tri = pathlib.Path("faces/tri/2DChannelTria.msh")
    mesh_file_quad = pathlib.Path("faces/quad/2DChannelQuad.msh")
    mesh_file_tetra = pathlib.Path("volumes/tet/3DBrickTet.msh")

    base_samples_path = pathlib.Path("samples")
    load_sample_file.load_sample_file(str(mesh_file_tri))
    load_sample_file.load_sample_file(str(mesh_file_quad))
    load_sample_file.load_sample_file(str(mesh_file_tetra))

    # load the .msh file directly with the correct io module (meshio)
    loaded_mesh_tri = io.meshio.load(base_samples_path / mesh_file_tri)

    loaded_mesh_tri.show()

    # load the .msh file directly with the correct io module (meshio)
    loaded_mesh_quad = io.meshio.load(base_samples_path / mesh_file_quad)

    loaded_mesh_quad.show()

    # load the .msh file with the default load function which needs to find out
    # it self which module is the correct one.
    loaded_mesh_default = io.load(base_samples_path / mesh_file_tri)

    loaded_mesh_default.show()

    loaded_meshes = io.load(
        base_samples_path / mesh_file_tri,
        return_only_one_mesh=False,
        set_boundary=True,
    )
    boundary_node_indices = loaded_meshes[0].BC["edges-nodes"]
    boundary_vertices = Vertices(
        loaded_meshes[0].vertices[boundary_node_indices]
    )

    show.show_vedo(
        ["faces", loaded_meshes[0]],
        ["boundaries: edges", loaded_meshes[1]],
        ["vertices", boundary_vertices],
    )

    rect_meshes = io.load(
        base_samples_path / mesh_file_tetra,
        return_only_one_mesh=False,
        set_boundary=True,
    )

    show.show_vedo(
        *[
            [name, mesh]
            for name, mesh in zip(["volume", "face", "line"], rect_meshes)
        ]
    )

    volume_nodes = Vertices(
        rect_meshes[0].vertices[rect_meshes[0].BC["tri-nodes"]]
    )
    faces_nodes = Vertices(
        rect_meshes[0].vertices[rect_meshes[1].BC["edges-nodes"]]
    )

    show.show_vedo(
        ["volumes: boundary-nodes", rect_meshes[0], volume_nodes],
        ["faces: boundary-nodes", rect_meshes[1], faces_nodes],
    )
