"""gustaf/spline/__init__.py.

Interface for `splinepy` library and additional operations. Everything
related to `splinepy` is here, which includes its load function.
However, since create.spline does not rely on `splinepy`, it is not
here.
"""

import splinepy
from splinepy import io

from gustaf.spline import base, create, extract, ffd, microstructure
from gustaf.spline.base import (
    NURBS,
    Bezier,
    BSpline,
    RationalBezier,
    from_mfem,
    load_splines,
    show,
)

# overwrite name to type map
splinepy.settings.NAME_TO_TYPE = dict(
    Bezier=Bezier,
    RationalBezier=RationalBezier,
    BSpline=BSpline,
    NURBS=NURBS,
)

# a shortcut
NAME_TO_TYPE = splinepy.settings.NAME_TO_TYPE

__all__ = [
    "base",
    "create",
    "extract",
    "Bezier",
    "RationalBezier",
    "BSpline",
    "NURBS",
    "show",
    "from_mfem",
    "load_splines",
    "ffd",
    "microstructure",
    "io",
]
