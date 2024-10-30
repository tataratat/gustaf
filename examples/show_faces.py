import numpy as np

import gustaf as gus

if __name__ == "__main__":
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
            # [1, 3, 7],
            [3, 2, 6],
            [2, 0, 4],
            [4, 5, 7],
            [2, 3, 1],
            # [5, 4, 0], # make some holes to see triangles
            [7, 5, 1],
            [6, 7, 3],
            [4, 6, 2],
            [7, 6, 4],
        ]
    )
    # define quad connectivity
    qf = np.array(
        [
            [1, 0, 2, 3],
            # [0, 1, 5, 4],
            [1, 3, 7, 5],
            # [3, 2, 6, 7], # show quad hole
            [2, 0, 4, 6],
            [4, 5, 7, 6],
        ]
    )

    # init tri faces
    tri = gus.Faces(
        vertices=v,
        faces=tf,
    )

    # init quad faces
    quad = gus.Faces(
        vertices=v,
        faces=qf,
    )

    # show
    gus.show(["triangles", tri], ["quads", quad])

    # plot data - plots vector data as arrows
    for mesh in [tri, quad]:
        mesh.vertex_data["coords"] = np.random.default_rng().random(
            tri.vertices.shape
        )
        mesh.show_options(arrow_data="coords")
    gus.show(["triangles with arrows", tri], ["quads with arrows", quad])

    # point data to origin
    for mesh in [tri, quad]:
        mesh.show_options(arrow_data_to_origin=True)
    gus.show(
        ["triangles arrows to origin", tri], ["quads arrows to origin", quad]
    )
