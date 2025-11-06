import os
import tempfile

import numpy as np
import pytest

import gustaf as gus

all_grids = (
    ("faces_tri_2d", "mfem_triangles_2d.mesh"),
    ("faces_quad_2d", "mfem_quadrilaterals_2d.mesh"),
    ("volumes_hexa", "mfem_hexahedra_3d.mesh"),
    ("volumes_tet", "mfem_tetrahedra_3d.mesh"),
    ("volumes_hexa_222", "mfem_hexahedra_3d_222.mesh"),
)


@pytest.mark.parametrize(("grid", "ground_truth_filename"), all_grids)
def test_mfem_export(
    to_tmpf,
    are_stripped_lines_same,
    grid,
    ground_truth_filename,
    request,
):
    mesh = request.getfixturevalue(grid)

    verts = mesh.vertices

    if mesh.whatami in ("tri", "quad"):
        lines = mesh.to_edges(unique=False)
        BC = {1: [], 2: [], 3: []}
        for i in lines.single_edges():
            # mark boundaries at x = 0 with 1
            if np.max(verts[lines.const_edges[i], 0]) < 0.1:
                BC[1].append(i)
            # mark boundaries at x = 1 with 2
            elif np.min(verts[lines.const_edges[i], 0]) > 0.9:
                BC[2].append(i)
            # mark rest of the boundaries with 3
            else:
                BC[3].append(i)
    elif mesh.whatami in ("hexa", "tet"):
        faces = mesh.to_faces(unique=False)

        BC = {1: [], 2: [], 3: []}
        # single faces only produces exterior/boundary faces
        for i in faces.single_faces():
            # mark boundaries at x = 0 with 1
            if np.max(verts[faces.const_faces[i], 0]) < 0.1:
                BC[1].append(i)
            # mark boundaries at x = 1 with 2
            elif np.min(verts[faces.const_faces[i], 0]) > 0.9:
                BC[2].append(i)
            # mark rest of the boundaries with 3 if they do not contain
            # interior vertices
            else:
                BC[3].append(i)

    mesh.BC = BC

    # Test output
    with tempfile.TemporaryDirectory() as tmpd:
        tmpf = to_tmpf(tmpd)
        gus.io.mfem.export(tmpf, mesh)

        with (
            open(tmpf) as tmp_read,
            open(
                os.path.join(
                    os.path.dirname(__file__),
                    f"data/{ground_truth_filename}",
                )
            ) as base_file,
        ):
            assert are_stripped_lines_same(
                base_file.readlines(), tmp_read.readlines(), ignore_order=True
            )
