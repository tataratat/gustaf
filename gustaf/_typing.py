from typing import Union

from gustaf.edges import Edges
from gustaf.faces import Faces
from gustaf.vertices import Vertices
from gustaf.volumes import Volumes

MESH_TYPES = Union[Vertices, Edges, Faces, Volumes]

try:
    from gustaf.spline.base import BSpline, NURBS, Bezier, RationalBezier
    SPLINE_TYPES = Union[Bezier, RationalBezier, BSpline, NURBS]
except ImportError:
    SPLINE_TYPES = None
