import gustaf as gus
import numpy as np

if __name__ == "__main__":

    # hexahedron mesh
    hex_mesh = gus.io.hmascii.load("input/hypermesh_hex.hmascii")

    hex_mesh.shrink().show()
    hex_mesh.tofaces().shrink().show()

    # demonstrate face group display
    for face_group in hex_mesh.face_groups:
        hex_mesh.extract_face_group(face_group).show()

    # demonstrate face group to vertex group conversion
    hex_mesh.face_groups.export_all_vertex_groups()
    for vertex_group in hex_mesh.vertex_groups:
        hex_mesh.extract_vertex_group(vertex_group).show()

    # demonstrate vertex group to face group conversion
    del hex_mesh.face_groups["hull"]
    hex_mesh.face_groups.import_vertex_group("hull")
    hex_mesh.extract_face_group("hull").show()

    # tetrahedron mesh
    tet_mesh = gus.io.hmascii.load("input/hypermesh_tet.hmascii")
    tet_mesh.shrink().show()

    tet_mesh.extract_face_group("hull").show()

