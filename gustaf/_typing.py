from typing import Union

from gustaf.edges import Edges
from gustaf.faces import Faces
from gustaf.spline.base import NURBS, BSpline, Bezier
from gustaf.vertices import Vertices
from gustaf.volumes import Volumes



SPLINE_TYPES = Union[Bezier, BSpline, NURBS]
MESH_TYPES = Union[Vertices, Edges, Faces, Volumes]