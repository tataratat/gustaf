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


def test_edges_to_polygons():
    """
    test 
    """
    pass
