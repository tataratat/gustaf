import numpy as np
import pytest

import gustaf as gus

# frequently used fixtures
all_grids = (
    "vertices",
    "edges",
    "faces_tri",
    "faces_quad",
    "volumes_tet",
    "volumes_hexa",
)


@pytest.fixture
def volumes_hexa333():
    v = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [1.0, 0.5, 0.5],
        [0.0, 1.0, 0.5],
        [0.5, 1.0, 0.5],
        [1.0, 1.0, 0.5],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.5, 0.5, 1.0],
        [1.0, 0.5, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]

    vol = [
        [0, 1, 4, 3, 9, 10, 13, 12],
        [1, 2, 5, 4, 10, 11, 14, 13],
        [3, 4, 7, 6, 12, 13, 16, 15],
        [4, 5, 8, 7, 13, 14, 17, 16],
        [9, 10, 13, 12, 18, 19, 22, 21],
        [10, 11, 14, 13, 19, 20, 23, 22],
        [12, 13, 16, 15, 21, 22, 25, 24],
        [13, 14, 17, 16, 22, 23, 26, 25],
    ]

    return gus.Volumes(v, vol)


update_elements_params = (
    *all_grids[1:-1],  # without vertices and simple hexa
    "volumes_hexa333",
)


@pytest.mark.parametrize("grid", all_grids)
def test_unique_vertices(grid, np_rng, request):
    """Test unique_vertices. requires scipy."""
    grid = request.getfixturevalue(grid)

    # random vertices
    n_ran = 50
    random_vertices = np_rng.random((n_ran, grid.vertices.shape[1]))
    # copy original
    n_original_vertices = len(grid.vertices)
    original_vertices = grid.vertices.copy()
    # assign new vertices
    #  ==> stacks (original, random, original, random)
    grid.vertices = np.vstack(
        (grid.vertices, random_vertices, original_vertices, random_vertices)
    )
    # check if vertices are correctly set
    assert len(grid.vertices) == int((n_original_vertices + n_ran) * 2)

    unique_vs = grid.unique_vertices(return_intersection=True)

    n_expected_unique = int(n_original_vertices + n_ran)

    # value check

    assert np.allclose(grid.vertices[:n_expected_unique], unique_vs.values)

    # ids check
    assert all(np.arange(n_expected_unique) == unique_vs.ids)

    # inverse check
    assert all(np.tile(np.arange(n_expected_unique), 2) == unique_vs.inverse)

    # intersection check - should include itself as well
    # also, should be sorted, assuming scipy version is > 1.6
    intersection_list = [*unique_vs.intersection]
    intersection_ref = [
        [i, i + n_expected_unique] for i in range(n_expected_unique)
    ] * 2
    # although these are integers, this is still very nice one-line assert
    assert np.allclose(intersection_list, intersection_ref)


@pytest.mark.parametrize("grid", all_grids)
def test_bounds(grid, np_rng, request):
    """bounds should give you AABB"""
    grid = request.getfixturevalue(grid)

    # [0, 1]^3 cube
    ref = [[0, 0, 0], [1, 1, 1]]

    # same?
    assert np.allclose(ref, grid.bounds())

    # add some random values of [0, 1)
    n_original_vs = len(grid.vertices)
    n_ran = 50
    random_vertices = np_rng.random((n_ran, grid.vertices.shape[1]))
    grid.vertices = np.vstack((grid.vertices, random_vertices))

    # bound shouldn't change
    assert np.allclose(ref, grid.bounds())

    # remove unreferenced if they are elements
    if not grid.kind.startswith("vertex"):
        grid.remove_unreferenced_vertices()
        assert len(grid.vertices) == n_original_vs
        assert np.allclose(ref, grid.bounds())


@pytest.mark.parametrize("grid", all_grids)
def test_update_vertices(grid, np_rng, request):
    """update vertices should keep only masked values"""
    grid = request.getfixturevalue(grid)

    # make a copy
    test_grid = grid.copy()

    # int based mask - let's keep 3 vertices
    n_original_vs = len(grid.vertices)
    n_vertices_to_keep = 3
    int_mask = np_rng.choice(
        np.arange(n_original_vs),
        n_vertices_to_keep,
        replace=False,
    )

    # save ref values for updated vertices
    updated_vs_ref = test_grid.vertices[int_mask]

    # update_vertices
    test_grid.update_vertices(int_mask)

    assert len(test_grid.vertices) == n_vertices_to_keep
    assert np.allclose(updated_vs_ref, test_grid.vertices)

    # make a copy
    test_grid = grid.copy()
    assert len(test_grid.vertices) == n_original_vs

    # bool based mask - same masking as int_mask
    bool_mask = np.zeros(n_original_vs, dtype=bool)
    bool_mask[int_mask] = True

    # bool mask will keep values in sorted way
    reorganize_ref = np.argsort(int_mask)

    # update_vertices
    test_grid.update_vertices(bool_mask)

    assert len(test_grid.vertices) == n_vertices_to_keep
    assert np.allclose(updated_vs_ref[reorganize_ref], test_grid.vertices)

    # check if elements are updated accordingly
    if not grid.kind.startswith("vertex"):
        # a bit more expensive, but should give
        # same result as inverse
        original_elements = grid.elements.copy()
        kept_vertices = original_elements == int_mask[0]
        for v_id in int_mask[1:]:
            kept_vertices |= original_elements == v_id

        assert len(test_grid.elements) == sum(kept_vertices.all(axis=1))


@pytest.mark.parametrize("grid", update_elements_params)
def test_update_elements(grid, np_rng, request):
    """keep masked elements"""
    grid = request.getfixturevalue(grid)

    n_original_es = len(grid.elements)
    n_elements_to_keep = 3
    int_mask = np_rng.choice(
        np.arange(n_original_es),
        n_elements_to_keep,
        replace=False,
    )

    test_grid = grid.copy()
    # save ref values for updated vertices
    updated_es_ref = test_grid.elements[int_mask].copy()

    # update_vertices
    test_grid.update_elements(int_mask)

    assert len(test_grid.elements) == n_elements_to_keep
    assert np.allclose(
        grid.vertices[updated_es_ref], test_grid.vertices[test_grid.elements]
    )

    # make a copy
    test_grid = grid.copy()
    assert len(test_grid.elements) == n_original_es

    # bool based mask - same masking as int_mask
    bool_mask = np.zeros(n_original_es, dtype=bool)
    bool_mask[int_mask] = True

    # bool mask will keep values in sorted way
    reorganize_ref = np.argsort(int_mask)

    # update_vertices
    test_grid.update_elements(bool_mask)

    assert len(test_grid.elements) == n_elements_to_keep
    assert np.allclose(
        grid.vertices[updated_es_ref[reorganize_ref]],
        test_grid.vertices[test_grid.elements],
    )

    # see if vertices are removed.
    # this uses remove_unreferenced_vertices
    leftover_vertex_ids = np.unique(test_grid.elements)
    leftover_vertex_ids_ref = np.unique(grid.elements[int_mask])
    assert len(leftover_vertex_ids) == len(leftover_vertex_ids_ref)

    assert np.allclose(
        test_grid.vertices[leftover_vertex_ids],
        grid.vertices[leftover_vertex_ids_ref],
    )
