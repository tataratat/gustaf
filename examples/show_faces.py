import numpy as np

import gustaf as gus


def example():
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
    tri.show()
    quad.show()


if __name__ == "__main__":
    example()
