import gustaf as gus
import numpy as np


if __name__ == "__main__":
    # create 4x5 element test mesh
    v_res = [5, 6]
    vertices = gus.create.vertices.raster(bounds=[[0, 0], [1, 1]], resolutions=v_res)
    connec = gus.utils.connec.make_quad_faces(v_res)

    quad = gus.Faces(vertices.vertices, connec)

    # show
    quad.show()
    quad.shrink().show()

    e = quad.shrink(ratio=0.7).toedges(False)
    e.vis_dict["arrows"] = True
    e.show()

