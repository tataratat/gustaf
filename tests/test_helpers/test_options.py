import pytest

import gustaf as gus

def sample_option():
    return gus.helpers.options.Option(
        "vedo", "color", "color in german is Farbe", (str, tuple, list), "green"
    )

def test_option_init():
    op = gus.helpers.options.Option("vedo", "a", "abc", (str, int), "efg")
    assert "vedo" in op.backends
    assert op.key == "a"
    assert op.description == "abc"
    assert op.allowed_types == set(str, int)
    assert op.default == "efg"

def test_option_type_check():
    pass