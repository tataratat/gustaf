import numpy as np

from gustaf.edges import Edges
from gustaf.faces import Faces
from gustaf.vertices import Vertices
from gustaf.volumes import Volumes


def load(fname, **kwargs):
    """
    Read gus_object from `.npz` file.

    .. code-block:: python

        tet = gustaf.io.npz.load("export/tet.npz")
        tet.show()

    Returns
    --------
    gus_object: gustaf entity
        Can be vertices, Edges, Faces or Volumes
    """

    # Load the file and check entity type
    loaded = np.load(fname, **kwargs)
    dict_keys = list(loaded.keys())
    whatami = str(loaded["whatami"])

    gus_object_types = {
        "edges": Edges,
        "tri": Faces,
        "quad": Faces,
        "tet": Volumes,
        "hexa": Volumes,
    }

    # Vertices
    if whatami == "vertices":
        gus_object = Vertices(vertices=loaded["vertices"])
    # Meshes
    elif whatami in gus_object_types:
        gus_object = gus_object_types[whatami](
            vertices=loaded["vertices"], elements=loaded["elements"]
        )
    # Unknown types
    else:
        raise ValueError(f"Type {whatami} is not supported in gustaf.")

    for key in dict_keys:
        # Load vertex data
        if key.startswith("vertex_data-"):
            gus_object.vertex_data[key.removeprefix("vertex_data-")] = loaded[
                key
            ]
        # Load boundaries
        elif key.startswith("BC-"):
            gus_object.BC[key.removeprefix("BC-")] = loaded[key]

    return gus_object


def export(gus_object, fname):
    """
    Save a gustaf object (Vertices, Edges, Faces or Volumes) as a `npz`-file.
    The export file contains (if applicable): vertices, elements, vertex data,
    BC.
    The `npz`-format is a compressed numpy format
    https://numpy.org/doc/stable/reference/generated/numpy.savez.html .

    .. code-block:: python

        import gustaf
        import numpy as np

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

        # Create hexa elements and set vertex data
        hexa.vertex_data["arange"] = np.arange(len(vertices))
        hexa.vertex_data["quad_x"] = [v[0] ** 2 for v in vertices]
        hexa.BC = {
            "min_z": np.array([0, 1, 2, 3]),
            "max_z": np.array([4, 5, 6, 7]),
        }

        # Export hexa
        gustaf.io.npz.export(hexa, "export/hexa.npz")

    Parameters
    -----------
    gus_object: gustaf entity
    fname: str
       Export filename with `npz` suffix

    Returns
    --------
    None
    """

    if not fname.endswith(".npz"):
        raise ValueError("The filename must end with .npz.")

    property_dicts = {
        "whatami": gus_object.whatami,
        "vertices": gus_object.vertices,
    }

    # Include elements for meshes
    if type(gus_object) in [Edges, Faces, Volumes]:
        property_dicts["elements"] = gus_object.elements

    # Export vertex data
    property_dicts.update(
        {
            f"vertex_data-{key}": item
            for key, item in gus_object.vertex_data.items()
        }
    )

    # In case of faces and volumes, export the boundaries
    if type(gus_object) in [Faces, Volumes]:
        property_dicts.update(
            {f"BC-{key}": item for key, item in gus_object.BC.items()}
        )

    np.savez(
        fname,
        **property_dicts,
    )
