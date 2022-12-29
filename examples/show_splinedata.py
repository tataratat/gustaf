import numpy as np

import gustaf as gus

if __name__ == "__main__":
    # turn on debug logs
    # gus.utils.log.configure(debug=True)
    # surface
    # define degrees
    ds2 = [2, 2]

    # define knot vectors
    kvs2 = [
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
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
    b = gus.BSpline(
        degrees=ds2,
        knot_vectors=kvs2,
        control_points=cps2,
    )

    # define splinedata
    # 1. see coordinates's norm
    b.splinedata["me"] = b
    b.show_options["dataname"] = "me"
    b.show()

    # 2. see coordinate's norm and as arrow
    b.show_options["arrowdata"] = "me"
    b.show()

    # 3. see coordinates norm and as arrows only on specified places
    b.show_options["arrowdata_on"] = np.random.random((100,2)) # para_coords
    b.show()

    # 4. see 3. with parametric_view
    b.show(parametric_space=True)
