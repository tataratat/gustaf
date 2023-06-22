import pathlib

from gustaf.io import meshio, mfem, mixd


def load(fname):
    """Load function for all supported file formats.

    This function tries to guess the correct io module for the given file.

    Parameters
    -----------
    fname: Union[str, pathlib.Path]
        Name of the file to be loaded.

    Returns
    --------
    MESH_TYPES:
        Loaded mesh.
    """
    extensions_to_load_functions = {
        ".mixd": mixd.load,
        ".mfem": mfem.load,
        ".msh": meshio.load,
    }

    fname = pathlib.Path(fname).resolve()

    if fname.suffix in extensions_to_load_functions:
        return extensions_to_load_functions[fname.suffix](fname)

    else:
        raise ValueError(
            f"Failed to load given file with '{fname.suffix}' extension. "
            "Valid extensions are: "
            f"{tuple(extensions_to_load_functions.keys())}."
        )
