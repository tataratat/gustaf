import gustaf as gus


def example():
    # create 2x3x4 test hexa element
    v_res = [2, 3, 4]
    vertices = gus.create.vertices.raster(
        bounds=[[0, 0, 0], [1, 1, 1]], resolutions=v_res
    )
    connec = gus.utils.connec.make_hexa_volumes(v_res)
    v = gus.Volumes(vertices.vertices, connec)

    # v shrink - f shrink - e shrink
    e = v.shrink().to_faces().shrink().to_edges().shrink()
    e.show_options["as_arrows"] = True

    direct_toedges = v.to_edges(unique=False).shrink()
    direct_toedges.show_options["as_arrows"] = True

    # not the most efficient way, but it is possible.
    gus.show(
        ["v, Volumes", v],
        ["v.shrink()", v.shrink()],
        [
            "v.shrink().to_faces().shrink()",
            v.shrink().to_faces().shrink(),
        ],
        [
            "v.shrink().to_faces().shrink().to_edges().shrink()",
            v.shrink().to_faces().shrink().to_edges().shrink(),
        ],
        [
            "as arrows - useful for orientation check!",
            e,
        ],
        [
            "v.to_edges(unique=False).shrink()\nas arrows",
            direct_toedges,
        ],
    )


if __name__ == "__main__":
    example()
