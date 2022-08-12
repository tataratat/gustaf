import gustaf as gus
import numpy as np


if __name__ == "__main__":
    # create 4x5 element test mesh
    quad = gus.create.faces.quad_block_mesh(
            bounds = [[0, 0], [1, 1]],
            resolutions = [5, 6]
            )

    # we create vertex and face groups for testing purposes
    quad.vertex_groups["odd_vertices"] = np.arange(quad.vertices.shape[0],
            step=2)
    quad.face_groups["odd_faces"] = np.arange(quad.faces.shape[0], step=2)

    # show
    quad.shrink().show()

    tri = gus.create.faces.simplexify(quad)

    option_plots = [
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

    vertex_group_plots = (
                [[vertex_group, quad.extract_vertex_group(vertex_group)] for
                    vertex_group in quad.vertex_groups] +
                [[vertex_group, tri.extract_vertex_group(vertex_group)] for
                    vertex_group in tri.vertex_groups]
                )

    edge_group_plots = (
                [[edge_group, quad.extract_edge_group(edge_group).shrink()]
                    for edge_group in quad.edge_groups] +
                [[edge_group, tri.extract_edge_group(edge_group).shrink()]
                    for edge_group in tri.edge_groups]
                )

    face_group_plots = (
                [[face_group, quad.extract_face_group(face_group).shrink()]
                    for face_group in quad.face_groups] +
                [[face_group, tri.extract_face_group(face_group).shrink()]
                    for face_group in tri.face_groups]
                )

    try:
        import vedo
        gus.show.show_vedo(*option_plots)
        gus.show.show_vedo(*vertex_group_plots)
        gus.show.show_vedo(*edge_group_plots)
        gus.show.show_vedo(*face_group_plots)
    except:
        for item in (option_plots + vertex_group_plots +
                edge_group_plots + face_group_plots):
            print(f"Showing {item[0]}.")
            item[1].show()

    # show edges
    e = tri.shrink(ratio=0.7).toedges(False)
    e.vis_dict["arrows"] = True
    e.show()

