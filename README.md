![gustaf](https://github.com/tataratat/gustaf/raw/gustaf/docs/source/gustaf-logo.png)

__gustaf__ is a python library to process and visualize numerical-analysis-geometries; especially for Finite Element Methods (FEM) and Isogemetric Analysis (IGA).
gustaf currently supports linear elements:
- triangle,
- quadrilateral,
- tetrahedron, and
- hexahedron,
as well as both single and multi-patch splines (with `splinepy` extension):
- Bezier,
- Rational Bezier,
- BSpline, and
- NURBS.


## Installation
`gustaf` only has `numpy` for its strict dependency. The minimal version can be installed using `pip`.
```
pip install gustaf
```
To install all the [optional dependencies](#dependencies) at the same time, you can use:
```
pip install gustaf[all]
```
For the latest develop version of gustaf:
```
pip install git+https://github.com/tataratat/gustaf.git@main
```

## Quick Start
This example shows how to visualize and extract properties of tetrahedrons and NURBS using gustaf. 
For visualization, gustaf uses [vedo](https://vedo.embl.es) as main backend.
```python
import gustaf as gus
import numpy as np


# create tet mesh using Volumes
# it requires vertices and connectivity info, volumes
tet = gus.Volumes(
    vertices=[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    volumes=[
        [0, 2, 7, 3],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [5, 0, 4, 7],
        [5, 0, 7, 1],
        [7, 0, 3, 1],
    ],
)
tet.show()


# elements can transform to their subelement types
# set unique=True, if you don't want duplicating internal subelements
as_faces = tet.tofaces(unique=False)
as_edges = tet.toedges(unique=False)


# as geometry classes inherit from its subelement class, we can
# extract subelement connectivity directly.
# Volumes' subelements are faces and subsubelements are edges
face_connectivity = tet.faces()
edge_connectivity = tet.edges()


# this holds
assert np.allclose(face_connectivity, as_faces.faces)
assert np.allclose(edge_connectivity, as_edges.edges)

# the uniqueness of subelement connectivity is useful for finding
# boundary elements, especially ones that appear only once.

```

**But until then check out the `examples` folder, where some functionality is already shown.**

For some examples a submodule is necessary, this can be initialized via the commandline:

```
git submodule update --init
```

This submodule provides sample geometries.


### Dependencies
- []
|Package|Optional|pip|conda|Description|
|-------|:---:|---|-----|-----------|
|numpy|no|`pip install numpy`|`conda install numpy`|Necessary for computation|
|splinepy|yes|`pip install splinepy`|-|Necessary for any spline based functionality|
|vedo|yes|`pip install vedo`|`conda install -c conda-forge vedo`|Default renderer of gustaf, only needed if visualization is performed|
|scipy|yes|`pip install scipy`|`conda install scipy`|Necessary for vertex operations|
|meshio|yes|`pip install meshio`|`conda install -c conda-forge meshio`|Necessary for meshio mesh imports|
|pytest|yes|`pip install pytest`|`conda install pytest`|Necessary for testing of the package. Not needed for normal usage.|

If you install `gustaf` from source we recommend to also install `splinepy` from source, see the install instructions for this in the [splinepy docs](https://tataratat.github.io/splinepy).



Test version of documentations are available [here](https://tataratat.github.io/gustaf/)
