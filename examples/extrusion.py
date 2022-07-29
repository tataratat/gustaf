import gustaf as gus
import numpy as np

if __name__ == "__main__":
    # create 4x5 element test mesh
    v_res = [5, 6]
    vertices = gus.create.vertices.raster(bounds=[[0, 0], [1, 1]], resolutions=v_res)
    connec = gus.utils.connec.make_quad_faces(v_res)
    quad = gus.Faces(vertices.vertices, connec)
    tri = gus.create.faces.quad_to_tri(quad)

    # show
    tri.shrink().show()

    # 1 layer
    tet1 = gus.create.volumes.extrude_tri_to_tet(tri, thickness = .2,
            layers = 1, randomize = True)
    tet1.show()

    # 3 layers
    tet3 = gus.create.volumes.extrude_tri_to_tet(tri, thickness = .2,
            layers = 3, randomize = True)
    tet3.shrink().show()
