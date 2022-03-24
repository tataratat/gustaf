"""gustaf/gustaf/settings.py

Global variables/constants that's used throughout `gustaf`.
"""

TOLERANCE = 1e-10

FLOAT_DTYPE = "float64"
INT_DTYPE = "int32"

VISUALIZATION_BACKEND = "vedo" # OPTIONS are <"vedo" | "trimesh" | "matplotlib">

VEDO_DEFAULT_OPTIONS = dict(
    vertex=dict(
    ),
    edges=dict(
    ),
    faces=dict(
    ),
    volumes=dict(
    ),
)

NTHREADS = 1
