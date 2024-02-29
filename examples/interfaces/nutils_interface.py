"""This example shows the functionality of gustaf.io.nutils.

A 2-dimensional plate with fixed edges on the left and right edges is
exposed to Gravitation.
The plate consists of 20 triangular elements and is created with gustaf.
One can either add the displacements directly to gustaf.vertices or evaluate
the new coordinates and create a new gustaf.mesh.
"""

import numpy as np
from nutils import function, mesh, solver
from nutils.expression_v2 import Namespace

import gustaf as gus


def example():
    # Simulation parameters
    degree = 1
    btype = "std"

    # Physical Parameters
    g = 9.81
    poisson = 0.4
    rho = 1.2
    e_mod = 1 * pow(10, 2)

    m = define_mesh()

    # Define Boundaries by Edges
    m.BC.update({"1": np.array([4, 10]), "2": np.array([49, 55])})

    m_in = m.copy()

    to_nutils = gus.io.nutils.to_nutils_simplex(m)
    domain, geom = mesh.simplex(**to_nutils)

    # Define the Namespace for Nutils Simulation
    ns = Namespace()
    ns.g = g
    ns.rho = rho
    ns.x = geom

    ns.define_for("x", gradient="∇", normal="n", jacobians=("dV", "dS"))
    ns.ubasis = domain.basis(btype, degree=degree).vector(2)
    ns.u = function.dotarg("lhs", ns.ubasis)
    ns.X_i = "x_i + u_i"

    ns.lmbda = (e_mod * poisson) / (1 + poisson) / (1 - 2 * poisson)
    ns.mu = e_mod / (1 + poisson)

    ns.strain_ij = "(∇_j(u_i) + ∇_i(u_j)) / 2"
    ns.energy = "lmbda strain_ii strain_jj + 2 mu strain_ij strain_ij"
    ns.energy = "energy + g rho u_1"

    # Set the boundaries. Note, that bound-keys start with 1,2,3,..
    sqr = domain.boundary["1"].integral("u_k u_k dS" @ ns, degree=degree * 2)
    sqr += domain.boundary["2"].integral("u_k u_k dS" @ ns, degree=degree * 2)

    cons = solver.optimize("lhs", sqr, droptol=1e-15)

    energy = domain.integral("energy dV" @ ns, degree=degree * 2)

    lhs = solver.optimize("lhs", energy, constrain=cons)  # displacements

    # With bezier
    bezier = domain.sample("vertex", 0)
    coordinates = bezier.eval("X_i" @ ns, lhs=lhs)
    element_nodes = bezier.tri

    # how many axes?
    # (we get simplices, so take number of simplex nodes minus 1)
    n_axes = element_nodes.shape[1] - 1

    # generate gustaf mesh using information given by nutils
    mesh_type = {1: gus.Edges, 2: gus.Faces, 3: gus.Volumes}[n_axes]
    gustaf_mesh = mesh_type(elements=element_nodes, vertices=coordinates)

    # with lhs
    deformation = lhs.reshape(m.vertices.shape)
    m.vertices += deformation

    gus.show.show(
        [m_in, "gustaf_input-mesh"],
        [m, "gustaf_mesh-with-lhs"],
        [gustaf_mesh, "gustaf_mesh-bezier"],
        c="blue",
    )


def define_mesh():
    v = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [4.0, 2.0],
            [5.0, 0.0],
            [5.0, 1.0],
            [5.0, 2.0],
        ]
    )

    tf = np.array(
        [
            [0, 3, 4],
            [4, 1, 0],
            [1, 4, 5],
            [5, 2, 1],
            [3, 6, 7],
            [7, 4, 3],
            [4, 7, 8],
            [8, 5, 4],
            [6, 9, 10],
            [10, 7, 6],
            [7, 10, 11],
            [11, 8, 7],
            [9, 12, 13],
            [13, 10, 9],
            [10, 13, 14],
            [14, 11, 10],
            [12, 15, 16],
            [16, 13, 12],
            [13, 16, 17],
            [17, 14, 13],
        ]
    )

    mesh = gus.Faces(vertices=v, faces=tf)

    return mesh


if __name__ == "__main__":
    example()
