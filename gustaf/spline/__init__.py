"""gustaf/spline/__init__.py.

Interface for `splinepy` library and additional operations. Everything
related to `splinepy` is here, which includes its load function.
However, since create.spline does not rely on `splinepy`, it is not
here.
"""

from gustaf.spline import base
from gustaf.spline import create
from gustaf.spline import extract
from gustaf.spline.base import (
        Bezier, RationalBezier, BSpline, NURBS, show, from_mfem, load_splines
)
from gustaf.spline import ffd
from gustaf.spline import microstructure

import splinepy
from splinepy import io

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
