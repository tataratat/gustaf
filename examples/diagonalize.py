import gustaf as gus
import numpy as np


if __name__ == "__main__":
    # create 4x5 element test mesh
    v_res = [5, 6]
    vertices = gus.create.vertices.raster(bounds=[[0, 0], [1, 1]], resolutions=v_res)
    connec = gus.utils.connec.make_quad_faces(v_res)

    quad = gus.Faces(vertices.vertices, connec)

    # we create vertex and face groups for testing purposes
    quad.vertex_groups["odd_vertices"] = np.arange(vertices.vertices.shape[0],
            step=2)
    quad.face_groups["odd_faces"] = np.arange(connec.shape[0], step=2)

    # show
    quad.shrink().show()

    tri = gus.create.faces.simplexify(quad)

    show_options = [
            ["no backslash, no alternate",
            gus.create.faces.simplexify(quad, alternate=False).shrink()],
            ["backslash, no alternate",
            gus.create.faces.simplexify(quad, backslash=True,
                alternate=False).shrink()],
            ["no backslash, alternate",
            tri.shrink()],
            ["backslash, alternate",
            gus.create.faces.simplexify(quad, backslash=True).shrink()]
            ]

    try:
        import vedo # if nothing's raised, following should be usable

        gus.show.show_vedo(*show_options)

        gus.show.show_vedo(*(
                [[vertex_group, quad.extract_vertex_group(vertex_group)] for
                    vertex_group in quad.vertex_groups] +
                [[vertex_group, tri.extract_vertex_group(vertex_group)] for
                    vertex_group in tri.vertex_groups] +
                [[face_group, quad.extract_face_group(face_group).shrink()] for
                    face_group in quad.face_groups] +
                [[face_group, tri.extract_face_group(face_group).shrink()] for
                    face_group in tri.face_groups]
                ))
    except:
        for show_option in show_options:
            print(f"Showing: {show_option[0]}")
            show_option[1].show()

        # show groups
        for vertex_group in tri.vertex_groups:
            print(f"Showing vertex group {vertex_group}.")
            quad.extract_vertex_group(vertex_group).show()
            tri.extract_vertex_group(vertex_group).show()
        for face_group in tri.face_groups:
            print(f"Showing face group {face_group}.")
            quad.extract_face_group(face_group).shrink().show()
            tri.extract_face_group(face_group).shrink().show()

    # show edges
    e = tri.shrink(ratio=0.7).toedges(False)
    e.vis_dict["arrows"] = True
    e.show()
