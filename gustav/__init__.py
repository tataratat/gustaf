from gustav import _version
from gustav import settings
from gustav import vertices
from gustav import edges
from gustav import faces
from gustav import volumes
from gustav import show
from gustav import utils
from gustav import create
from gustav import io

try:
    from gustav import spline
    from gustav.spline.base import BSpline, NURBS
except:
    utils.log.debug("can't import spline related modules.")

# import try/catch for triangle and gustav-tetgen

from gustav.vertices import Vertices
from gustav.edges import Edges
from gustav.faces import Faces
from gustav.volumes import Volumes

__version__ = _version.version

__all__ = [
    "__version__",
    "Vertices",
    "Edges",
    "Faces",
    "Volumes",
]
