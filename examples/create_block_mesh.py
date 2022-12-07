import gustaf as gus


def main():

    mesh_faces_box = gus.create.faces.box(
            bounds=[[0, 0], [2, 2]], resolutions=[2, 3]
    )

    mesh_volumes_box = gus.create.volumes.box(
            bounds=[[0., 0., 0.], [1., 1., 1.]], resolutions=[2, 3, 4]
    )

    gus.show.show_vedo(
            ["faces-box", mesh_faces_box],
            ["volumes-box", mesh_volumes_box],
    )


if __name__ == "__main__":
    main()
