"""Create gus"""
import gustaf as gus

if __name__ == "__main__":

    l_eye = gus.spline.create.disk(0.15)
    l_eye.cps += [0.6, 0.2]
    l_eye.show_options["c"] = "black"

    r_eye = gus.spline.create.disk(0.15)
    r_eye.cps += [-0.6, 0.2]
    r_eye.show_options["c"] = "black"

    upperlip = gus.Bezier(
        [5, 1],
        [
            [-0.8, -0.1],
            [-0.2, -0.4],
            [-0.5, 0.5],
            [0.5, 0.5],
            [0.2, -0.4],
            [0.8, -0.1],
            [-0.75, -0.15],
            [-0.2, -0.55],
            [-0.5, 0.05],
            [0.5, 0.05],
            [0.2, -0.55],
            [0.75, -0.15],
        ],
    )
    upperlip.show_options["c"] = "orange7"

    innermouth = gus.Bezier(
        [5, 1],
        [
            *upperlip.cps[6:],
            [-0.75, -0.15],
            [-0.6, -0.325],
            [-0.4, -0.525],
            [0.4, -0.525],
            [0.6, -0.325],
            [0.75, -0.15],
        ],
    )
    innermouth.show_options["c"] = "orange5"

    lowerlip = gus.Bezier(
        [5, 1],
        [
            *innermouth.cps[6:],
            [-0.8, -0.1],
            [-0.6, -0.4],
            [-0.4, -0.6],
            [0.4, -0.6],
            [0.6, -0.4],
            [0.8, -0.1],
        ],
    )
    lowerlip.show_options["c"] = "orange7"

    plt = gus.show(
        [l_eye, r_eye, upperlip, innermouth, lowerlip],
        control_points=False,
        knots=False,
        lighting="off",
        close=False,
    )
