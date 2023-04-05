import numpy as np

import gustaf as gus
import vedo

colors = list(vedo.colors.colors.keys())

def random_color():
    return np.random.choice(colors)

TOL = 1e-10
if __name__ == "__main__":
    # curve
    # define degrees
    ds1 = [1]

    # define knot vectors
    kvs1 = [[0.0, 0.0, 1.0, 1.0]]

    # define control points
    cps1 = np.array(
        [
            [0.0, 0.0],
            [1.0, 3.0],
        ]
    )

    # init bspline
    b = gus.BSpline(
        degrees=ds1,
        knot_vectors=kvs1,
        control_points=cps1,
    )

    # make richer
    b.elevate_degrees([0,0])
    b.insert_knots(0, [.1,.2,.4,.7,.8,.8,.8])

    # query points
    q = np.linspace(0,1, 1001).reshape(-1,1)

    # compute 
    basis, support = b.basis_and_support(q)

    # build a basis matrix - not necessary, just easy to extract
    mat = np.zeros((len(q), len(b.control_points)))
    np.put_along_axis(mat, support, basis, axis=1)

    # prepare vertices to plot
    vertices = [
        np.hstack((q, mat[:, i].reshape(-1,1))) for i in range(mat.shape[1])
    ]

    # you are looking at the right part
    edges = [
        gus.Edges(ev, gus.utils.connec.range_to_edges(len(ev))) for ev in vertices
    ]

    # assign some color
    for e in edges:
        e.show_options["c"] = random_color()
        e.show_options["lw"] = 5

    # press `+` key to plot axes behind
    gus.show(edges)
