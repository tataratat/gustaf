import numpy as np
import pytest
from typing import List

from gustaf.spline.ffd import FFD


@pytest.mark.parametrize(
        "init_values, throw_error",
        [
                ([], False),  # test empty input
                (["jallo"], True),
                ([1], True),
                (["fixture_bspline_2d"], True),
                (["fixture_faces_tri"], False),
                (["fixture_faces_quad"], False),
                (["fixture_faces_tri", "fixture_bspline_3d"], False),
                (["fixture_faces_tri", "fixture_bspline_3d_deformed"], False),
                (["fixture_faces_tri", 1], True),
                (["fixture_faces_tri", "testing"], True),
        ]
)
def test_init_error(init_values: List, throw_error, request):
    for idx, value in enumerate(init_values):
        if isinstance(value, str):
            if "fixture_" in value:
                init_values[idx] = request.getfixturevalue(value[8:])
    if throw_error:
        with pytest.raises(ValueError):
            FFD(*init_values)
    else:
        FFD(*init_values)


def test_mesh_with_empty_init():
    a = FFD()
