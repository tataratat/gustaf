import pytest

import gustaf


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
    assert dataholder.pop("c") == 33
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
