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
    #quad.face_groups["odd_faces"] = quad.get_surfaces()[::2]

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

    group_plots = []
    for vertex_group in quad.vertex_groups:
        group_plots += [
                [vertex_group, quad.extract_vertex_group(vertex_group)],
                [vertex_group, tet3.extract_vertex_group(vertex_group)]
                ]
    for face_group in tet3.face_groups:
        group_plots += [[face_group,
            tet3.extract_face_group(face_group).shrink()]]

    try:
        import vedo
        gus.show.show_vedo(*extrusion_plots)
        gus.show.show_vedo(*group_plots)
    except:
        for item in extrusion_plots + group_plots:
            print(f"Showing {item[0]}.")
            item[1].show()

