import os
import tempfile

import numpy as np
import pytest

import gustaf as gus

all_grids = (
    ("volumes_hexa", "mfem_hexahedra_3d.mesh"),
    ("volumes_tet", "mfem_tetrahedra_3d.mesh"),
)


@pytest.mark.parametrize("grid", all_grids)
def test_mfem_export(to_tmpf, are_stripped_lines_same, grid, request):
    mesh = request.getfixturevalue(grid[0])
    ground_truth_filename = grid[1]

    verts = mesh.vertices

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
            os.path.dirname(__file__) + f"/./data/{ground_truth_filename}"
        ) as base_file:
            print(base_file.readlines())
            assert are_stripped_lines_same(
                base_file.readlines(), tmp_read.readlines(), True
            )
