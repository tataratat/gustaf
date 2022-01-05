import gustav as gus
import numpy as np


if __name__ == "__main__":
    # define coordinates
    v = np.array(
        [
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 1., 0.],
            [0., 0., 1.],
            [1., 0., 1.],
            [0., 1., 1.],
            [1., 1., 1.],
        ]
    )
    # define edge connectivity
    e = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7]
        ]
    )


    # init edges
    edges = gus.Edges(
        vertices=v,
        edges=e,
    )

    # show
    edges.show()
