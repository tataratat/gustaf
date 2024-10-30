from pathlib import Path

import numpy as np

import gustaf as gus

if __name__ == "__main__":
    export_path = Path("export")

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
    # define tet connectivity
    tv = np.array(
        [
            [0, 2, 7, 3],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [5, 0, 4, 7],
            [5, 0, 7, 1],
            [7, 0, 3, 1],
        ]
    )

    # init tet
    tet = gus.Volumes(
        vertices=v,
        volumes=tv,
    )

    fac = gus.Faces(vertices=v, faces=tv[:4, :3])

    tet.vertex_data["arange"] = np.arange(len(tet.vertices))

    # Export with subgroup triangles
    gus.io.meshio.export(
        export_path / "export_meshio_subgroups.vtu", tet, submeshes=[fac]
    )
    # The mesh still has to conform the needed format, e.g. stl will discard
    # tetrahedra
    gus.io.meshio.export(
        export_path / "export_meshio_subgroups.stl", tet, submeshes=[fac]
    )

    # Export only tetrahedra
    gus.io.meshio.export(export_path / "export_meshio.vtu", tet)
    gus.io.meshio.export(export_path / "export_meshio.stl", tet.to_faces())
