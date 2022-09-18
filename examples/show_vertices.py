import gustaf as gus
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

    # init vertices
    vert = gus.Vertices(vertices=v)

    # show
    vert.show()
