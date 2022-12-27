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

    # init vertices
    vert = gus.Vertices(vertices=v)

    # show
    vert.show()
