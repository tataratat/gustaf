import numpy as np
import pytest

import gustaf


def new_tracked_array(dtype=float):
    """
    create new tracked array and checks if default flags are set correctly.
    Then sets modified to False, to give an easy start for testing
    """
    ta = gustaf.helpers.data.make_tracked_array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ],
        dtype=dtype,
    )

    assert ta.modified
    assert ta._super_arr

    ta.modified = False

    return ta


def test_TrackedArray():
    """test if modified flag is well set"""
    # 1. set item
    ta = new_tracked_array()
    ta[0] = 1
    assert ta.modified

    ta = new_tracked_array()
    ta[1, 1] = 2
    assert ta.modified

    # in place
    ta = new_tracked_array()
    ta += 5
    assert ta.modified

    ta = new_tracked_array()
    ta -= 3
    assert ta.modified

    ta = new_tracked_array()
    ta *= 1
    assert ta.modified

    ta = new_tracked_array()
    ta /= 1.5
    assert ta.modified

    ta = new_tracked_array()
    ta @= ta
    assert ta.modified

    ta = new_tracked_array()
    ta **= 2
    assert ta.modified

    ta = new_tracked_array()
    ta %= 3
    assert ta.modified

    ta = new_tracked_array()
    ta //= 2
    assert ta.modified

    ta = new_tracked_array(int)
    ta <<= 3
    assert ta.modified

    ta = new_tracked_array(int)
    ta >>= 1
    assert ta.modified

    ta = new_tracked_array(int)
    ta |= 3
    assert ta.modified

    ta = new_tracked_array(int)
    ta &= 3
    assert ta.modified

    ta = new_tracked_array(int)
    ta ^= 3
    assert ta.modified

    # child array modification
    ta = new_tracked_array()
    ta_child = ta[0]
    assert ta_child.base is ta
    ta_child += 5
    assert ta.modified
    assert ta_child.modified

    # copy returns normal np.ndarray
    assert isinstance(new_tracked_array().copy(), np.ndarray)


def test_DataHolder():
    """Base class of dataholder types"""

    class Helpee:
        pass

    helpee = Helpee()

    dataholder = gustaf.helpers.data.DataHolder(helpee)

    # setitem is pure abstract
    with pytest.raises(NotImplementedError):
        dataholder["somedata"] = []

    # test other functions by injecting some keys and values directly to the
    # member
    dataholder._saved.update(a=1, b=2, c=3)

    # getitem
    assert dataholder["a"] == 1
    assert dataholder["b"] == 2
    assert dataholder["c"] == 3
    with pytest.raises(KeyError):
        dataholder["d"]

    # contains
    assert "a" in dataholder
    assert "b" in dataholder
    assert "c" in dataholder
    assert "d" not in dataholder
    assert "e" not in dataholder

    # len
    assert len(dataholder) == 3

    # pop
    assert dataholder.pop("c") == 3
    assert "c" not in dataholder

    # get
    # 1. key
    assert dataholder.get("a") == 1
    # 2. key and default
    assert dataholder.get("b", 2) == 2
    # 3. key and wrong default
    assert dataholder.get("b", 3) == 2
    # 4. empty key - always None
    assert dataholder.get("c") is None
    # 5. empty key and default
    assert dataholder.get("c", "123") == "123"

    # keys
    assert len(set(dataholder.keys()).difference({"a", "b"})) == 0

    # values
    assert len(set(dataholder.values()).difference({1, 2})) == 0

    # items
    for k, v in dataholder.items():
        assert k in dataholder.keys()  # noqa SIM118
        assert v in dataholder.values()

    # update
    dataholder.update(b=22, c=33, d=44)
    assert dataholder["a"] == 1
    assert dataholder["b"] == 22
    assert dataholder["c"] == 33
    assert dataholder["d"] == 44

    # clear
    dataholder.clear()
    assert "a" not in dataholder
    assert "b" not in dataholder
    assert "c" not in dataholder
    assert "d" not in dataholder
    assert len(dataholder) == 0


@pytest.mark.parametrize(
    "grid", ("edges", "faces_tri", "faces_quad", "volumes_tet", "volumes_hexa")
)
def test_ComputedData(grid, request):
    grid = request.getfixturevalue(grid)

    # vertex related data
    v_data = (
        "unique_vertices",
        "bounds",
        "bounds_diagonal",
        "bounds_diagonal_norm",
    )

    # element related data
    e_data = (
        "sorted_edges",
        "unique_edges",
        "single_edges",
        "edges",
        "sorted_faces",
        "unique_faces",
        "single_faces",
        "faces",
        "sorted_volumes",
        "unique_volumes",
    )

    # for both
    both_data = ("centers", "referenced_vertices")

    # entities before modification
    data_dependency = {"vertex": v_data, "element": e_data, "both": both_data}
    before = {}
    for dependency, attributes in data_dependency.items():
        # init
        before[dependency] = {}
        for attr in attributes:
            func = getattr(grid, attr, None)
            if attr is not None and callable(func):
                before[dependency][attr] = func()

        # ensure that func is called at least once
        assert len(before[dependency]) != 0

    # loop to check if you get the saved data
    for attributes in before.values():
        for attr, value in attributes.items():
            func = getattr(grid, attr, None)
            assert value is func()

    # change vertices - assign new vertices
    grid.vertices = grid.vertices.copy()
    for dependency, attributes in before.items():
        if dependency == "element":
            continue
        for attr, value in attributes.items():
            func = getattr(grid, attr, None)
            assert value is not func()  # should be different object

    # change elements - assign new elements
    grid.elements = grid.elements.copy()
    for dependency, attributes in before.items():
        if dependency == "vertex":
            continue
        for attr, value in attributes.items():
            func = getattr(grid, attr, None)
            assert value is not func()


@pytest.mark.parametrize(
    "grid", ("edges", "faces_tri", "faces_quad", "volumes_tet", "volumes_hexa")
)
def test_VertexData(grid, request):
    grid = request.getfixturevalue(grid)

    key = "vertices"

    # set data
    grid.vertex_data[key] = grid.vertices

    # get_data - data is viewed as TrackedArray, so check against base
    assert grid.vertices is grid.vertex_data[key].base

    # scalar extraction should return a norm
    assert np.allclose(
        grid.vertex_data.as_scalar(key).ravel(),
        np.linalg.norm(grid.vertex_data.get(key), axis=1),
    )

    # norms should be saved, as long as data array isn't changed
    assert grid.vertex_data.as_scalar(key) is grid.vertex_data.as_scalar(key)

    before = grid.vertex_data.as_scalar(key)
    # trigger modified flag on data - either reset or inplace change
    # reset first - with copy, just so that we can try to make inplace changes
    # later
    grid.vertex_data[key] = grid.vertex_data[key].copy()
    assert before is not grid.vertex_data.as_scalar(key)
    assert grid.vertex_data.as_scalar(key) is grid.vertex_data.as_scalar(key)

    grid.vertex_data[key][0] = grid.vertex_data[key][0]
    assert before is not grid.vertex_data.as_scalar(key)
    assert grid.vertex_data.as_scalar(key) is grid.vertex_data.as_scalar(key)

    # check arrow data
    assert grid.vertex_data[key] is grid.vertex_data.as_arrow(key)

    # check wrong length assignment
    with pytest.raises(ValueError):
        grid.vertex_data["bad"] = np.vstack((grid.vertices, grid.vertices))

    # check wrong arrow data request
    with pytest.raises(ValueError):
        grid.vertex_data["norm"] = grid.vertex_data.as_scalar(key)
        grid.vertex_data.as_arrow("norm")
