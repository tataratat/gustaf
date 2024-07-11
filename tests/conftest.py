"""gustaf/test/common.py

Common imports/routines needed for testing.
"""

import contextlib
import os
import re

import numpy as np
import pytest

import gustaf as gus


@pytest.fixture
def np_rng():
    return np.random.default_rng()


@pytest.fixture
def vertices_3d():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return V


@pytest.fixture
def vertices(vertices_3d):
    return gus.Vertices(vertices_3d)


@pytest.fixture
def edge_connec():
    E = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ],
        dtype=np.int32,
    )
    return E


@pytest.fixture
def edges(vertices_3d, edge_connec):
    return gus.Edges(vertices_3d, edge_connec)


@pytest.fixture
def tri_connec():
    TF = np.array(
        [
            [1, 0, 2],
            [0, 1, 5],
            [1, 3, 7],
            [3, 2, 6],
            [2, 0, 4],
            [4, 5, 7],
            [2, 3, 1],
            [5, 4, 0],
            [7, 5, 1],
            [6, 7, 3],
            [4, 6, 2],
            [7, 6, 4],
        ],
        dtype=np.int32,
    )
    return TF


@pytest.fixture
def faces_tri(vertices_3d, tri_connec):
    return gus.Faces(vertices_3d, tri_connec)


@pytest.fixture
def quad_connec():
    QF = np.array(
        [
            [1, 0, 2, 3],
            [0, 1, 5, 4],
            [1, 3, 7, 5],
            [3, 2, 6, 7],
            [2, 0, 4, 6],
            [4, 5, 7, 6],
        ],
        dtype=np.int32,
    )
    return QF


@pytest.fixture
def faces_quad(vertices_3d, quad_connec):
    return gus.Faces(vertices_3d, quad_connec)


@pytest.fixture
def tet_connec():
    TV = np.array(
        [
            [0, 2, 7, 3],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [5, 0, 4, 7],
            [5, 0, 7, 1],
            [7, 0, 3, 1],
        ],
        dtype=np.int32,
    )
    return TV


@pytest.fixture
def volumes_tet(vertices_3d, tet_connec):
    return gus.Volumes(vertices_3d, tet_connec)


@pytest.fixture
def hexa_connec():
    HV = np.array([[0, 1, 3, 2, 4, 5, 7, 6]], dtype=np.int32)
    return HV


@pytest.fixture
def volumes_hexa(vertices_3d, hexa_connec):
    return gus.Volumes(vertices_3d, hexa_connec)


@pytest.fixture
def provide_data_to_unittest(
    request,
    vertices_3d,
    edge_connec,
    tri_connec,
    quad_connec,
    tet_connec,
    hexa_connec,
):
    request.cls.V = vertices_3d
    request.cls.E = edge_connec
    request.cls.TF = tri_connec
    request.cls.QF = quad_connec
    request.cls.TV = tet_connec
    request.cls.HV = hexa_connec


@pytest.fixture
def are_stripped_lines_same():
    def _are_stripped_lines_same(a, b, ignore_order=False):
        """returns True if items in a and b same, preceding and tailing
        whitepaces are ignored and strings are joined"""
        all_same = True

        for i, (line_a, line_b) in enumerate(zip(a, b)):
            # check stripped string
            stripped_a, stripped_b = line_a.strip(), line_b.strip()

            # print general info
            if stripped_a != stripped_b:
                print(f"stripped line at index-{i} are not the same")
                print(f"\tfrom first: {line_a}")
                print(f"\tfrom second: {line_b}")

            # give one more chance if ignore_order
            if stripped_a != stripped_b and ignore_order:
                print("\tchecking again, while ignoring word order:")

                # This is meant for attributes
                delimiters = r" |\>|\<|\t|,"
                splitted_a = list(
                    filter(None, re.split(delimiters, stripped_a))
                )
                splitted_b = list(
                    filter(None, re.split(delimiters, stripped_b))
                )
                # first, len check
                len_a, len_b = len(splitted_a), len(splitted_b)
                if len(splitted_a) != len(splitted_b):
                    print(f"\t\tdifferent word counts: a-{len_a}, b-{len_b}")
                    all_same = False
                else:
                    # word order
                    a_to_b = []
                    nums_b = None
                    for word_a in splitted_a:
                        try:
                            a_to_b.append(splitted_b.index(word_a))
                        except BaseException:
                            try:
                                num_a = float(word_a)
                            except ValueError:
                                pass
                            else:
                                if nums_b is None:
                                    nums_b = []
                                    for idx, num_b in enumerate(splitted_b):
                                        with contextlib.suppress(ValueError):
                                            nums_b.append((idx, float(num_b)))
                                for idx, num_b in nums_b:
                                    if np.isclose(num_a, num_b):
                                        a_to_b.append(idx)
                                        break
                                else:
                                    print(
                                        f"\t\tsecond does not contain ({word_a})"
                                    )
                                    all_same = False

        return all_same

    return _are_stripped_lines_same


@pytest.fixture
def to_tmpf():
    def _to_tmpf(tmpd):
        """given tmpd, returns tmpf"""
        return os.path.join(tmpd, "nqv248p90")

    return _to_tmpf
