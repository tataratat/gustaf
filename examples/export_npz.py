import numpy as np

import gustaf

if __name__ == "__main__":
    # Define coordinates
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    # Define hexa connectivity
    hv = np.array([[0, 1, 3, 2, 4, 5, 7, 6]])

    # Init hexa elements
    hexa = gustaf.Volumes(
        vertices=vertices,
        volumes=hv,
    )

    # Define tet connectivity
    tv = np.array(
        [
            [0, 2, 7, 3],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [5, 0, 4, 7],
            [5, 0, 7, 1],
            [7, 0, 3, 1],
        ]
    )

    # Create tetra elements and set vertex data
    tet = gustaf.Volumes(vertices=vertices, volumes=tv)
    tet.vertex_data["arange"] = -np.arange(len(vertices))
    tet.BC = {"min_z": np.array([0, 1, 2, 3]), "max_z": np.array([4, 5, 6, 7])}

    # Create hexa elements and set vertex data
    hexa.vertex_data["arange"] = np.arange(len(vertices))
    hexa.vertex_data["quad_x"] = [v[0] ** 2 for v in vertices]
    hexa.BC = {
        "min_z": np.array([0, 1, 2, 3]),
        "max_z": np.array([4, 5, 6, 7]),
    }

    hexa.show_options["data_name"] = "arange"

    # Export hexa, edges and vertices
    gustaf.io.npz.export(hexa, "export/hexa.npz")
    gustaf.io.npz.export(hexa.to_edges(), "export/hexa_edges.npz")
    gustaf.io.npz.export(hexa.to_vertices(), "export/hexa_vertices.npz")

    # Load hexa files
    load_hexa = gustaf.io.npz.load("export/hexa.npz")
    load_edges = gustaf.io.npz.load("export/hexa_edges.npz")
    load_verts = gustaf.io.npz.load("export/hexa_vertices.npz")

    # Show original and loaded file
    # Show options are not exported and have to be reset
    gustaf.show(
        ["Original hexa", hexa],
        ["Loaded hexa (show options must be set again)", load_hexa],
    )

    # Show the expored edges and vertices
    gustaf.show(
        ["Exported edges", load_edges],
        ["Exported vertices", load_verts],
        [
            "Exported hexa with indicated boundary nodes",
            load_hexa,
            gustaf.Vertices(load_hexa.vertices[load_hexa.BC["min_z"]]),
            gustaf.Vertices(load_hexa.vertices[load_hexa.BC["max_z"]]),
        ],
    )

    # Vertex data has to be set for edges and vertices as to_edges() and
    # to_vertices() does not copy it
    load_hexa.show_options["data_name"] = "arange"

    load_edges.vertex_data["quad_x"] = hexa.vertex_data["quad_x"]
    load_edges.show_options["data_name"] = "quad_x"

    load_verts.vertex_data["arange"] = hexa.vertex_data["arange"]
    load_verts.show_options["data_name"] = "arange"

    # Show colored vertices
    gustaf.show(
        ["Show entities with copied data", load_hexa], load_edges, load_verts
    )

    # Export tetra files
    gustaf.io.npz.export(tet, "export/tet.npz")
    gustaf.io.npz.export(tet.to_edges(), "export/tet_edges.npz")
    gustaf.io.npz.export(tet.to_vertices(), "export/tet_vertices.npz")

    # Load tetra files with default load function
    load_tet = gustaf.io.load("export/tet.npz")
    load_tet_edges = gustaf.io.load("export/tet_edges.npz")
    load_tet_verts = gustaf.io.load("export/tet_vertices.npz")

    # Show tetras
    gustaf.show(
        ["Tetra", load_tet],
        ["Tetra edges", load_tet_edges],
        ["Tetra vertices", load_verts],
    )
