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


def test_ComputedData():
    pass
