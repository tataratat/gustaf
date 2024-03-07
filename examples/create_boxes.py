"""In this example, four separate meshes are created.

mesh_faces_box: creates a quadrilateral 2D-face-mesh with quadrilateral
elements. The bounds of the mesh are two diagonal corners in the
x-y-directions, in this example [x=0, y=0] (origin) and [2, 2].
The number of vertices is defined by the resolution.
Here, two vertices along the x-axis and three vertices along the
y-axis are defined. In other words, a mesh of size 2x2 with 2 (1x2) elements
is created.
For a uniform grid, x- and y-value of resolutions must be equal.

mesh_volumes_box: creates a hexahedron 3D-block-mesh with hexahedron
elements. The bounds of the mesh are two diagonal corners in the
x-y-z-directions, in this example [x=0, y=0, z=0] (origin) and [1, 1, 1].
The number of vertices is defined by the resolution.
Here, two vertices along the x-axis and three vertices along
the y-axis and 4 vertices align the z-axis are defined.In other words, a mesh
of size 1x1x1 with 6 (1x2x3) elements is created.
For a uniform grid, x-, y- and z-value of resolutions must be equal.

mesh_faces_triangle: creates a quadrilateral 2D-face-mesh with triangular
elements. The bounds of the mesh are two diagonal corners in the
x-y-directions, in this example [x=0, y=0] (origin) and [2, 2].
The number of vertices is defined by the resolution. Here, three vertices
along the x-axis and three vertices along the y-axis are defined.
In other words, a mesh of size 2x2 with 8 triangular elements
is created.

mesh_faces_triangle_bs: Similar to mesh_faces_triangle, but with
diagonalization in the other direction.
"""

import gustaf as gus


def example():
    mesh_faces_box = gus.create.faces.box(
        bounds=[[0, 0], [2, 2]], resolutions=[2, 3]
    )

    mesh_volumes_box = gus.create.volumes.box(
        bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], resolutions=[2, 3, 4]
    )

    mesh_faces_triangle = gus.create.faces.box(
        bounds=[[0, 0], [2, 2]],
        resolutions=[3, 3],
        simplex=True,
        backslash=False,
    )

    mesh_faces_triangle_bs = gus.create.faces.box(
        bounds=[[0, 0], [2, 2]],
        resolutions=[3, 3],
        simplex=True,
        backslash=True,
    )

    gus.show(
        ["faces-box", mesh_faces_box],
        ["volumes-box", mesh_volumes_box],
        ["faces-triangle", mesh_faces_triangle],
        ["faces-triangle-backslash", mesh_faces_triangle_bs],
    )


if __name__ == "__main__":
    example()
