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


@pytest.mark.parametrize("grid", all_grids)
def test_concat(grid, request):
    """Test concat. uses merge_vertices to double check."""
    grid = request.getfixturevalue(grid)

    n_grids = 5

    n_vertices = len(grid.vertices)

    if grid.kind != "vertex":
        n_elements = len(grid.elements)

    grids = [grid for _ in range(n_grids)]

    concated = type(grid).concat(*grids)

    # check concatenated vertices
    assert np.allclose(np.tile(grid.vertices, (n_grids, 1)), concated.vertices)

    # check number of concatenated elements
    if grid.kind != "vertex":
        assert int(n_elements * n_grids) == len(concated.elements)
        assert int(n_vertices * n_grids - 1) == concated.elements.max()

    # merge vertices, then this should only have n_vertices left
    concated.merge_vertices()
    assert np.allclose(grid.vertices, concated.vertices)

    # as well as n_elements * n_grids, but elements just repeats n_grids times
    if grid.kind != "vertex":
        assert (
            np.tile(grid.elements, (n_grids, 1)) - concated.elements
        ).sum() == 0
