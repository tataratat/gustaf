"""gustaf/gustaf/settings.py.

Global variables/constants that's used throughout `gustaf`.
"""

from typing import Literal

TOLERANCE: float = 1e-10

FLOAT_DTYPE: Literal["float64"] = "float64"
INT_DTYPE: Literal["int32"] = "int32"

# OPTIONS are <"vedo" | "trimesh" | "matplotlib">
VISUALIZATION_BACKEND: Literal["vedo"] = "vedo"

VEDO_DEFAULT_OPTIONS: dict[str, dict] = {
    "vertex": {},
    "edges": {},
    "faces": {},
    "volumes": {},
}

NTHREADS: int = 1
