import numpy as np

import gustaf as gus

shuffle = np.random.default_rng().shuffle


def one_polygon():
    """
    edge connectivity of one polygon
    """
    op = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 0],
        ],
        dtype=int,
    )
    # shuffle before return
    shuffle(op)

    return op


def two_polygons():
    """
    edge connectivity of two polygons
    """
    tp = np.array(
        [
            # first, ascending order
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 0],
            # second, descending.
            [7, 6],
            [8, 7],
            [9, 8],
            [10, 9],
            [6, 10],
        ],
        dtype=int,
    )

    shuffle(tp)

    return tp


def one_line():
    """
    edge connectivity of one line
    """
    ol = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
        ],
        dtype=int,
    )

    shuffle(ol)

    return ol


def two_lines():
    """
    edge connectivity of two lines
    """
    tl = np.array(
        [
            # first, ascending order
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            # second, descending.
            [7, 6],
            [8, 7],
            [9, 8],
            [10, 9],
        ],
        dtype=int,
    )

    shuffle(tl)

    return tl


def test_sequentialize_directed_edges():
    # polygon
    # as default, it starts at minimum
    seq, is_p = gus.utils.connec.sequentialize_edges(
        one_polygon(), directed=True
    )
    assert is_p[0]
    assert seq[0] == [0, 1, 2, 3, 4, 5]

    seq, is_p = gus.utils.connec.sequentialize_edges(
        two_polygons(), directed=True
    )
    for ip in is_p:
        assert ip
    assert seq[0] == [0, 1, 2, 3, 4, 5]
    assert seq[1] == [6, 10, 9, 8, 7]

    seq, is_p = gus.utils.connec.sequentialize_edges(one_line(), directed=True)
    assert not is_p[0]
    assert seq[0] == [0, 1, 2, 3, 4, 5]

    # this one include descending indices.
    # it should be able to eliminate 6, as a starting point.
    # directed query keeps the direction
    seq, is_p = gus.utils.connec.sequentialize_edges(
        two_lines(), directed=True
    )
    for ip in is_p:
        assert not ip
    assert seq[0] == [0, 1, 2, 3, 4, 5]
    assert seq[1] == [10, 9, 8, 7, 6]


def test_sequentialize_edges():
    # polygon
    # as default, it starts at minimum
    seq, is_p = gus.utils.connec.sequentialize_edges(
        one_polygon(), directed=False
    )
    assert is_p[0]
    assert seq[0] == [0, 1, 2, 3, 4, 5]

    seq, is_p = gus.utils.connec.sequentialize_edges(
        two_polygons(), directed=False
    )
    for ip in is_p:
        assert ip
    assert seq[0] == [0, 1, 2, 3, 4, 5]
    assert seq[1] == [6, 10, 9, 8, 7]

    seq, is_p = gus.utils.connec.sequentialize_edges(
        one_line(), directed=False
    )
    assert not is_p[0]
    assert seq[0] == [0, 1, 2, 3, 4, 5]

    # this one include descending indices.
    # non-directed query doesn't keep the direction.
    seq, is_p = gus.utils.connec.sequentialize_edges(
        two_lines(), directed=False
    )
    for ip in is_p:
        assert not ip
    assert seq[0] == [0, 1, 2, 3, 4, 5]
    assert seq[1] == [6, 7, 8, 9, 10]
