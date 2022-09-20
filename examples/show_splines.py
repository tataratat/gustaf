import gustaf as gus
import numpy as np

if __name__ == "__main__":
    # turn on debug logs
    gus.utils.log.configure(debug=True)

    # curve
    # define degrees
    ds1 = [1]

    # define knot vectors
    kvs1 = [[0., 0., 1., 1.]]

    # define control points
    cps1 = np.array([
            [0., 0.],
            [1., 3.],
    ])

    # init bspline
    bspline1 = gus.BSpline(
            degrees=ds1,
            knot_vectors=kvs1,
            control_points=cps1,
    )

    # show
    bspline1.show()

    # surface
    # define degrees
    ds2 = [2, 2]

    # define knot vectors
    kvs2 = [
            [0., 0., 0., 0.5, 1., 1., 1.],
            [0., 0., 0., 1., 1., 1.],
    ]

    # define control points
    cps2 = np.array(
            [
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 1.5, 0],
                    [3, 1.5, 0],
                    [-1, 0, 0],
                    [-1, 2, 0],
                    [1, 4, 0],
                    [3, 4, 0],
                    [-2, 0, 0],
                    [-2, 2, 0],
                    [1, 5, 0],
                    [3, 5, 2],
            ]
    )

    # init bspline
    bspline2 = gus.BSpline(
            degrees=ds2,
            knot_vectors=kvs2,
            control_points=cps2,
    )

    # show
    bspline2.show()

    # volume
    # define degrees
    ds3 = [1, 1, 1]

    # define knot vectors
    kvs3 = [
            [0., 0., 1., 1.],
            [0., 0., 1., 1.],
            [0., 0., 1., 1.],
    ]

    # define control points
    cps3 = np.array(
            [
                    [0., 0., 0.],
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [1., 1., 0.],
                    [0., -1., 1.],
                    [1., 0., 1.],
                    [-1., 1., 2.],
                    [2., 2., 2.],
            ]
    )

    # init bspline
    bspline3 = gus.BSpline(
            degrees=ds3,
            knot_vectors=kvs3,
            control_points=cps3,
    )

    bspline3.show()
