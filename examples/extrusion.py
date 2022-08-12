import gustaf as gus
import numpy as np

if __name__ == "__main__":
    # create 4x5 element test mesh
    quad = gus.create.faces.quad_block_mesh(
            bounds = [[0, 0], [1, 1]],
            resolutions = [5, 6]
            )

    # we create vertex and face groups for testing purposes
    #quad.vertex_groups["odd_vertices"] = np.arange(vertices.vertices.shape[0],
    #        step=2)
    quad.face_groups["odd_faces"] = np.arange(quad.faces.shape[0], step=2)

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

    vertex_group_plots = []
    for vertex_group in quad.vertex_groups:
        vertex_group_plots += [
                [f"quad.{vertex_group}",
                    quad.extract_vertex_group(vertex_group)],
                [f"tet3.{vertex_group}",
                    tet3.extract_vertex_group(vertex_group)]
                ]

    face_group_plots = []
    for edge_group in quad.edge_groups:
        face_group_plots += [[f"quad.{edge_group}",
            quad.extract_edge_group(edge_group).shrink()]]
    for face_group in tet3.face_groups:
        face_group_plots += [[f"tet3.{face_group}",
            tet3.extract_face_group(face_group).shrink()]]

    volume_group_plots = []
    for face_group in quad.face_groups:
        volume_group_plots += [[f"quad.{face_group}",
            quad.extract_face_group(face_group).shrink()]]
    for volume_group in tet3.volume_groups:
        volume_group_plots += [[f"tet3.{volume_group}",
            tet3.extract_volume_group(volume_group).shrink()]]

    try:
        import vedo
        gus.show.show_vedo(*extrusion_plots)
        gus.show.show_vedo(*vertex_group_plots)
        gus.show.show_vedo(*face_group_plots)
        gus.show.show_vedo(*volume_group_plots)
    except:
        for item in (extrusion_plots + vertex_group_plots + face_group_plots
                + volume_group_plots):
            print(f"Showing {item[0]}.")
            item[1].show()

