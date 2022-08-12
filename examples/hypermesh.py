import gustaf as gus
import numpy as np

if __name__ == "__main__":

    hex_mesh = gus.io.hmascii.load("input/hypermesh_hex.hmascii")
    tet_mesh = gus.io.hmascii.load("input/hypermesh_tet.hmascii")

    # display volumes
    plots = [
            ["hex", hex_mesh.shrink()],
            ["tet", tet_mesh.shrink()]
            ]

    # display vertex groups
    hex_mesh.face_groups.export_all_vertex_groups()
    hex_vertex_plot = gus.show.group_plot(
            hex_mesh.extract_all_vertex_groups())
    tet_mesh.face_groups.export_all_vertex_groups()
    tet_vertex_plot = gus.show.group_plot(
            tet_mesh.extract_all_vertex_groups())

    # display bondaries
    hex_boundary_plot = gus.show.group_plot(
            hex_mesh.extract_all_subelement_groups(),
            shrink=0.98)
    tet_boundary_plot = gus.show.group_plot(
            tet_mesh.extract_all_subelement_groups(),
            shrink=0.98)

    try:
        import vedo
        gus.show.show_vedo(*plots)
        gus.show.show_vedo(hex_vertex_plot, tet_vertex_plot)
        gus.show.show_vedo(hex_boundary_plot, tet_boundary_plot)
    except:
        for item in plots:
            print(f"Showing {item[0]}.")
            item[1].show()
        for item in (hex_vertex_plot + tet_vertex_plot
                + hex_boundary_plot + tet_boundary_plot):
            item.show()

