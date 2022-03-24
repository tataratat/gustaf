"""gustaf/spline/__init__.py

Interface for `splinepy` library and additional operations.
Everything related to `splinepy` is here, which includes its load function.
However, since create.spline does not rely on `splinepy`, it is not here.
"""


from gustaf.spline import base
from gustaf.spline import extract
from gustaf.spline.base import show, from_mfem, load_splines
