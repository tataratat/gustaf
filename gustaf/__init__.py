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
from gustaf import helpers
from gustaf.vertices import Vertices
from gustaf.edges import Edges
from gustaf.faces import Faces
from gustaf.volumes import Volumes

has_spline = False
try:
    from gustaf import spline
    from gustaf.spline.base import BSpline, NURBS, Bezier, RationalBezier
    from gustaf.spline.ffd import FFD
    has_spline = True
except ImportError as err:
    # overwrites the all modules which depend on the `splinepy` library
    # with an object which will throw an error
    # as soon as it is used the first time. This means that any non spline
    # functionality works as before, but as soon as these are used a
    # comprehensive exception will be raised which is understandable in
    # contrast to the possible multitude of errors previously possible
    from gustaf.helpers.raise_if import ModuleImportRaiser
    spline = ModuleImportRaiser("splinepy", err)
    BSpline = spline
    NURBS = spline
    Bezier = spline
    RationalBezier = spline
    FFD = spline

# import try/catch for triangle and gustaf-tetgen

__version__ = _version.version

__all__ = [
        "__version__",
        "settings",
        "vertices",
        "edges",
        "faces",
        "volumes",
        "show",
        "utils",
        "create",
        "io",
        "helpers",
        "Vertices",
        "Edges",
        "Faces",
        "Volumes",
        "spline",
        "has_spline",
        "BSpline",
        "NURBS",
        "Bezier",
        "RationalBezier",
        "FFD",
]
