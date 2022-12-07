import gustaf as gus


def main():

    mesh_faces_triangle_bs = gus.create.faces.box(
            bounds=[[0, 0], [2, 2]],
            resolutions=[4, 4],
            simplex=True,
            backslash=True
    )

    mesh_faces_triangle = gus.create.faces.box(
            bounds=[[0, 0], [2, 2]],
            resolutions=[4, 4],
            simplex=True,
            backslash=False
    )

    gus.show.show_vedo(
            ["faces-triangle-backslash", mesh_faces_triangle_bs],
            ["faces-triangle", mesh_faces_triangle]
    )


if __name__ == "__main__":
    main()
