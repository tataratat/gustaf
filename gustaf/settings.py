"""gustaf/gustaf/settings.py.

Global variables/constants that's used throughout `gustaf`.
"""

TOLERANCE = 1e-10

FLOAT_DTYPE = "float64"
INT_DTYPE = "int32"

# OPTIONS are <"vedo" | "trimesh" | "matplotlib">
VISUALIZATION_BACKEND = "vedo"

VEDO_DEFAULT_OPTIONS = {
    "vertex": {},
    "edges": {},
    "faces": {},
    "volumes": {},
}

NTHREADS = 1

DEFAULT_OFFSCREEN = False
#: If True, the visualization will default to offscreen. This is useful for
#: running tests on CI/CD servers.
