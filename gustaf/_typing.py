import sys
from typing import Any, Union

from gustaf.edges import Edges
from gustaf.faces import Faces
from gustaf.vertices import Vertices
from gustaf.volumes import Volumes

MESH_TYPES = Union[Vertices, Edges, Faces, Volumes]


def is_mesh(candidate: Any) -> bool:
    """This function checks if the candidate is a mesh.

    The complicated version check is so that even python 3.6 is supported.

    Parameters
    -----------
    candidate: Any
      object to check for being a mesh.

    Returns
    --------
    is_mesh: bool
      Is the given object a mesh.
    """
    if float(sys.version.split(".")[1]) > 6:
        return isinstance(candidate, MESH_TYPES.__args__)
    else:
        return issubclass(type(candidate), MESH_TYPES)
