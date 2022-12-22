import numpy as np
import pytest

from gustaf.spline.ffd import FFD

@pytest.mark.parametrize(
    "init_values, throw_error",
    [
        ([], False), # test empty input
        (["jallo"], True),
        ([], False)

    ]
)
def test_init_error(init_values, throw_error, request):
    if throw_error:
        with pytest.raises(ValueError):
            FFD(*init_values)
    else:
        FFD(*init_values)


def test_mesh_with_empty_init():
    a = FFD()

