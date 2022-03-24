from gustaf import _version
from gustaf import settings
from gustaf import vertices
from gustaf import edges
from gustaf import faces
from gustaf import volumes
from gustaf import show
from gustaf import utils
from gustaf import create
from gustaf import io

try:
    from gustaf import spline
    from gustaf.spline.base import BSpline, NURBS
except:
    utils.log.debug("can't import spline related modules.")

# import try/catch for triangle and gustaf-tetgen

from gustaf.vertices import Vertices
from gustaf.edges import Edges
from gustaf.faces import Faces
from gustaf.volumes import Volumes

__version__ = _version.version

__all__ = [
    "__version__",
    "Vertices",
    "Edges",
    "Faces",
    "Volumes",
]
