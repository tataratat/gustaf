import gustaf as gus
import numpy as np

if __name__ == "__main__":
    # create 4x5 element test mesh
    quad = gus.create.faces.quad_block_mesh(
            bounds = [[0, 0], [1, 1]],
            resolutions = [5, 6]
            )

    # we create vertex and face groups for testing purposes
    quad.face_groups["odd_faces"] = np.arange(quad.faces.shape[0], step=2)
    quad.face_groups["even_faces"] = np.arange(start=1,
            stop=quad.faces.shape[0], step=2)

    extrusion_plots = []

    # show
    extrusion_plots += [["quad", quad.shrink()]]

    # 1 layer
    tet1 = gus.create.volumes.extrude_to_tet(quad, thickness = .2,
            layers = 1, randomize = True)
    extrusion_plots += [["1 layer", tet1]]

    # 3 layers
    tet3 = gus.create.volumes.extrude_to_tet(quad, thickness = .2,
            layers = 3, randomize = True)
    extrusion_plots += [["3 layers", tet3.shrink()]]

    quad_vertex_plot = gus.show.group_plot(quad.extract_all_vertex_groups())
    tet_vertex_plot = gus.show.group_plot(tet3.extract_all_vertex_groups())

    quad_boundary_plot = gus.show.group_plot(
            quad.extract_all_subelement_groups(),
            shrink=0.98)
    tet_boundary_plot = gus.show.group_plot(
            tet3.extract_all_subelement_groups(),
            shrink=0.98)

    quad_element_plot = gus.show.group_plot(
            quad.extract_all_element_groups(),
            shrink=0.8)
    tet_element_plot = gus.show.group_plot(
            tet3.extract_all_element_groups(),
            shrink=0.8)

    try:
        import vedo
        gus.show.show_vedo(*extrusion_plots)
        gus.show.show_vedo(quad_vertex_plot, tet_vertex_plot)
        gus.show.show_vedo(quad_boundary_plot, tet_boundary_plot)
        gus.show.show_vedo(quad_element_plot, tet_element_plot)
    except:
        for item in extrusion_plots:
            print(f"Showing {item[0]}.")
            item[1].show()
        for item in (quad_vertex_plot + tet_vertex_plot
                + quad_boundary_plot + tet_boundary_plot
                + quad_element_plot + tet_element_plot):
            item.show()

