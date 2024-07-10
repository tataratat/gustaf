import os
import tempfile

import numpy as np

import gustaf as gus

mesh = gus.Volumes(
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

tets = mesh.volumes
verts = mesh.vertices


def test_mfem_export(to_tmpf, are_stripped_lines_same):
    faces = mesh.to_faces(False)
    boundary_faces = faces.single_faces()

    BC = {1: [], 2: [], 3: []}
    for i in boundary_faces:
        # mark boundaries at x = 0 with 1
        if np.max(verts[faces.const_faces[i], 0]) < 0.1:
            BC[1].append(i)
        # mark boundaries at x = 1 with 2
        elif np.min(verts[faces.const_faces[i], 0]) > 0.9:
            BC[2].append(i)
        # mark rest of the boundaries with 3
        else:
            BC[3].append(i)

    mesh.BC = BC

    # Test output
    with tempfile.TemporaryDirectory() as tmpd:
        tmpf = to_tmpf(tmpd)
        gus.io.mfem.export(tmpf, mesh)

        with open(tmpf) as tmp_read, open(
            os.path.dirname(__file__) + "/./data/mfem_tetrahedra_3d.mesh"
        ) as base_file:
            assert are_stripped_lines_same(
                base_file.readlines(), tmp_read.readlines(), True
            )
