import numpy as np

import gustaf as gus


def main():
    # Creates a gustaf volume.
    mesh = create_mesh()

    # Export it as .mixd-file and .npz-file
    gus.io.mixd.export("export/export_mixd.xns", mesh)
    gus.io.nutils.export("export/export_npz.npz", mesh)

    # Load the mixd-file and the .npz-file
    mesh_mixd = gus.io.mixd.load(
        volume=True,
        mxyz="export/export_mixd.mxyz",
        mien="export/export_mixd.mien",
        mrng="export/export_mixd.mrng",
    )
    mesh_npz = gus.io.nutils.load("export/export_npz.npz")

    gus.show(
        ["gustaf-mesh", mesh],
        ["mixd-mesh", mesh_mixd],
        ["npz-mesh", mesh_npz],
    )


def create_mesh():
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

    return tet


if __name__ == "__main__":
    main()
