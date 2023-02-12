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
To install all the [optional dependencies](https://github.com/tataratat/gustaf#dependencies) at the same time, you can use:
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
# first, general information about connectivity uniqueness
unique_face_infos = tet.unique_faces()  # returns namedtuple
print(unique_face_infos.values)
print(unique_face_infos.ids)
print(unique_face_infos.inverse)
print(unique_face_infos.counts)

# there's a shortcut - single_volumes(), single_faces(), single_edges()
assert np.allclose(
    tet.single_faces(),
    unique_face_infos.ids[unique_face_infos.counts == 1]
)

# let's visualize some scalar data and vector data defined on vertices
tet.vertexdata["arange"] = np.arange(len(tet.vertices))  # scalar
tet.show_options["dataname"] = "arange"
tet.vertexdata["random"] = np.random.random((len(tet.vertices), 3))  # vector
tet.show_options["arrowdata"] = "random"
tet.show()


# create a 2D NURBS disc and visualize
# all the spline types inherits from splinepy's splines and equipped with
# additional functionalities
nurbs = gus.NURBS(
    degrees=[1, 2],
    knot_vectors=[
        [0, 0, 1, 1],
        [0, 0, 0, 1, 1, 2, 2, 2],
    ],
    control_points=[
        [ 1.        ,  0.        ],
        [ 0.5       ,  0.        ],
        [ 1.        ,  0.59493748],
        [ 0.5       ,  0.29746874],
        [ 0.47715876,  0.87881711],
        [ 0.23857938,  0.43940856],
        [-0.04568248,  1.16269674],
        [-0.02284124,  0.58134837],
        [-0.54463904,  0.83867057],
        [-0.27231952,  0.41933528],
    ],
    weights=[
        [1.        ],
        [1.        ],
        [0.85940641],
        [0.85940641],
        [1.        ],
        [1.        ],
        [0.85940641],
        [0.85940641],
        [1.        ],
        [1.        ]
    ]
)
nurbs.show()

# extract / sample using Extractor helper class
# they are all "show()"-able
nurbs_as_faces = nurbs.extract.faces(resolutions=[100, 50])
bezier_patches = nurbs.extract.beziers()  # returns list
boundaries = nurbs.extract.boundaries()  # list of boundary splines
subspline = nurbs.extract.spline(
    {0: [.4, .8], 1: .7}  # define range dimension-wise
)

# create derived spline using Creator helper class
extruded = nurbs.create.extruded(extrusion_vector=[0, 0, 1])
revolved = nurbs.create.revolved(axis=[1, 0, 0], angle=70)
parametric_view = nurbs.create.parametric_view()

# just like vertexdata, you can define splinedata
# for more options, checkout `gus.spline.SplineDataAdaptor`
# following will plot the norm of nurbs' physical coordinates
nurbs.splinedata["coords"] = nurbs
nurbs.show_options["dataname"] = "coords"

# show them all together. each arg is plotted on a separate subplot
# translate tet a bit to avoid overlapping
tet.vertices += [2, 0, 0]
gus.show(
    ["NURBS and translated tet together", nurbs, tet],
    ["Extruded NURBS", extruded],
    ["Revolved NURBS", revolved],
    ["NURBS parametric view", parametric_view],
)
```
Check out [documentations](https://tataratat.github.io/gustaf/) and [examples](https://github.com/tataratat/gustaf/tree/main/examples) for more!


### Dependencies
- [numpy](https://numpy.org)
- [splinepy](https://github.com/tataratat/splinepy)
- [vedo](https://vedo.embl.es)
- [scipy](https://scipy.org)
- [meshio](https://github.com/nschloe/meshio)
- [pytest](https://pytest.org)
