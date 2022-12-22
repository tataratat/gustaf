import numpy as np
import pytest

# frequently used fixtures
all_grids = (
        "vertices",
        "edges",
        "faces_tri",
        "faces_quad",
        "volumes_tet",
        "volumes_hexa",
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
    assert np.allclose(grid.vertices[:n_expected_unique], unique_vs.values)

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


@pytest.mark.parameterize("grid", all_grids)
def test_bounds(grid, request):
    """bounds should give you AABB"""
    grid = request.getfixturevalue(grid)

    # [0, 1]^3 cube
    ref = [[0,0,0], [1,1,1]]

    # same?
    assert np.allclose(ref, grid.bounds())

    # add some random values of [0, 1)
    n_orignal_vs = len(grid.vertices)
    n_ran = 50
    random_vertices = np.random.random((n_ran, grid.vertices.shape[1]))
    grid.vertices = np.vstack((grid.vertices, random_vertices))

    # bound shouldn't change
    assert np.allclose(ref, grid.bounds())

    # remove unreferenced if they are elements
    if not grid.kind.startswith("vertex")
        grid.remove_unreferenced_vertices()
        assert len(grid.vertices) == n_original_vs
        assert np.allclose(ref, grid.bounds())


@pytest.mark.parametersize("grid", all_grids)
def test_update_vertices(grid, request):
    """update vertices should keep only masked values"""
    grid = request.getfixturevalue(grid)

    # make a copy
    test_grid = grid.copy()

    # int based mask - let's keep 3 vertices
    n_original_vs = len(grid.vertices)
    n_vertices_to_keep = 3
    int_mask = np.random.choice(np.arange(n_original_vs), n_vertices_to_keep)

    # update_vertices
    test_grid.update_vertices(int_mask)

    assert len(grid.vertices) == n_vertices_to_keep
