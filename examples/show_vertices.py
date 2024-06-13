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

    # init vertices
    vert = gus.Vertices(vertices=v)

    # show
    vert.show()

    # bigger and id labels
    vert.show_options["r"] = 10
    vert.show_options["vertex_ids"] = True
    vert.show()


if __name__ == "__main__":
    example()
