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
    # define hexa connectivity
    hv = np.array([[0, 1, 3, 2, 4, 5, 7, 6]])

    # init tet
    tet = gus.Volumes(
            vertices=v,
            volumes=tv,
    )

    # init hexa
    hexa = gus.Volumes(
            vertices=v,
            volumes=hv,
    )

    # show
    tet.show()
    tet.show(shrink=False)  # Default is True
    hexa.show(c="blue")
