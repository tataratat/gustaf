from typing import Union

import pathlib

from gustaf._typing import MESH_TYPES
from gustaf.io import meshio, mfem, mixd


def load(fname: Union[str, pathlib.Path]) -> MESH_TYPES:
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
    extensions = {
            "mixd": {
                    "extensions": [".mixd"],
                    "load_function": mixd.load
            },
            "mfem": {
                    "extensions": [".mfem"],
                    "load_function": mfem.load
            },
            "meshio_extensions": {
                    "extensions": [".msh"],
                    "load_function": meshio.load
            }
    }
    all_extensions = [
            ext for load_type in extensions.values()
            for ext in load_type["extensions"]
    ]
    fname = pathlib.Path(fname).resolve()
    if fname.suffix in all_extensions:
        for load_type in extensions.values():
            if fname.suffix in load_type["extensions"]:
                return load_type["load_function"](fname)
    else:
        ValueError(
                f"The given file with extension {fname.suffix} can not be "
                f"loaded. Only files of the following types can be currently "
                f"loaded {all_extensions}"
        )
