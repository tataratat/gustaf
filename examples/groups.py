import gustaf as gus
import numpy as np

if __name__ == "__main__":
    # create 2x3x4 test hexa element
    v_res = [2, 3, 4]
    vertices = gus.create.vertices.raster(
        bounds=[[0, 0, 0], [1, 1, 1]],
        resolutions=v_res
    )
    connec = gus.utils.connec.make_hexa_volumes(v_res)
    v = gus.Volumes(vertices.vertices, connec)

    plots = []

    # show original mesh
    plots.append(["v", v.shrink()])

    # create a face group
    v.face_groups["odd_faces"] = v.get_surfaces()[::2]
    plots.append(["odd_faces", v.extract_face_group("odd_faces").shrink()])

    # create vertex group
    v.face_groups.export_all_vertex_groups()
    plots.append(["odd_faces", v.extract_vertex_group("odd_faces")])

    try:
        import vedo
        gus.show.show_vedo(*plots)
    except:
        for item in plots:
            print(f"Showing {item[0]}.")
            item[1].show()

