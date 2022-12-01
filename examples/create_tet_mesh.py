import gustaf as gus


def main():

    mesh_faces_triangle = gus.create.faces.triangle(        
        bounds = [[0, 0], [2, 2]],
        resolutions = [2, 3],
        alternate_diagonals = False
        )

    mesh_volumes_tet = gus.create.volumes.tet_block_mesh(
        bounds = [[0., 0., 0.], [1., 1., 1.]],
        resolutions = [2, 3, 4],
        alternate_diagonals = True,
        randomize_extrusion = False
    )

    gus.show.show_vedo(
        ["faces-triangle", mesh_faces_triangle], 
        ["volumes-tet", mesh_volumes_tet]
        )


if __name__ == "__main__":
    main()
