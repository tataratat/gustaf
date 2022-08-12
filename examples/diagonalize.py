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
    quad.face_groups["even_faces"] = np.arange(start=1,
            stop=quad.faces.shape[0], step=2)

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

    quad_vertex_plot = gus.show.group_plot(quad.extract_all_vertex_groups())
    tri_vertex_plot = gus.show.group_plot(tri.extract_all_vertex_groups())

    quad_edge_plot = gus.show.group_plot(quad.extract_all_edge_groups(),
            shrink=0.95)
    tri_edge_plot = gus.show.group_plot(tri.extract_all_edge_groups(),
            shrink=0.95)

    quad_face_plot = gus.show.group_plot(quad.extract_all_face_groups(),
            shrink=0.95)
    tri_face_plot = gus.show.group_plot(tri.extract_all_face_groups(),
            shrink=0.95)

    try:
        import vedo
        gus.show.show_vedo(*option_plots)
        gus.show.show_vedo(quad_vertex_plot, tri_vertex_plot)
        gus.show.show_vedo(quad_edge_plot, tri_edge_plot)
        gus.show.show_vedo(quad_face_plot, tri_face_plot)
    except:
        for item in option_plots:
            print(f"Showing {item[0]}.")
            item[1].show()
        for item in (quad_vertex_plot + tri_vertex_plot
                + quad_edge_plot + tri_edge_plot
                + quad_face_plot + tri_face_plot):
            item.show()

    # show edges
    e = tri.shrink(ratio=0.7).toedges(False)
    e.vis_dict["arrows"] = True
    e.show()

