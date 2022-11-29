import gustaf as gus


def main():

    mesh_faces = gus.create.faces.box(
            bounds=[[0, 0], [2, 2]], resolutions=[2, 3]
    )

    mesh_volumes = gus.create.volumes.box(
            bounds=[[0., 0., 0.], [1., 1., 1.]], resolutions=[2, 3, 4]
    )

    mesh_faces.show()
    mesh_volumes.show()


if __name__ == "__main__":
    main()
