import numpy as np
import pytest

import gustaf as gus

@pytest.fixture
def sample_vertices(vertices_3d):
    return gus.Vertices(vertices_3d)


@pytest.fixture
def sample_edges(edges):
    return gus.Edges(edges)


@pytest.fixture
def sample_tri(vertices_3d, faces_tri):
    return gus.Faces(vertices_3d, faces_tri)


@pytest.fixture
def sample_quad(vertices_3d, faces_quad):
    return gus.Faces(vertices_3d, faces_quad)


@pytest.fixture
def sample_tet(vertices_3d, volume_tet):
    return gus.Volumes(vertices_3d, volume_tet)


@pytest.fixture
def sample_hexa(vertices_3d, volume_hexa):
    return gus.Volumes(vertices_3d, volume_hexa)


all_grids = (
        "sample_vertices",
        "sample_edges",
        "sample_tri",
        "sample_quad",
        "sample_tet",
        "sample_hexa",
)


@pytest.mark.parametrize('grid', all_grids)
def test_unique_vertices(grid, request):
    """Test unique_vertices. requires scipy."""
    grid = request.getfixturevalue(grid)

    # random vertices
    n_ran = 50
    random_vertices = np.random.random((n_ran, grid.vertices.shape[1]))
    # copy original
    n_original_vertices = len(grid.vertices)
    original_vertices = grid.vertices.copy()
    # assgin new vertices
    #  ==> stacks (original, random, original, random)
    grid.vertices = np.vstack(
            (
                    grid.vertices, random_vertices, original_vertices,
                    random_vertices
            )
    )
    # check if vertices are correctly set
    assert len(grid.vertices) == int((n_original_vertices + n_ran) * 2)

    unique_vs = grid.unique_vertices()

    n_expected_unique = int(n_original_vertices + n_ran)

    # value check
    assert np.allclose(
            grid.vertices[:n_expected_unique],
            unique_vs.values
    )

    # ids check
    assert all(np.arange(n_expected_unique) == unique_vs.ids)

    # inverse check
    assert all(np.tile(np.arange(n_expected_unique), 2) == unique_vs.inverse)

    # intersection check - should include itself as well
    # also, should be sorted, assuming scipy verion is > 1.6
    intersection_list = [*unique_vs.intersection]
    intersection_ref = [
            [i, i + n_expected_unique] for i in range(n_expected_unique)
    ] * 2
    assert intersection_list == intersection_ref 
