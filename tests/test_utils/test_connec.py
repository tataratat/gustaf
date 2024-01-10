import numpy as np

import gustaf as gus

shuffle = np.random.default_rng().shuffle
randint = np.random.default_rng().integers


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


def quad_face_3_4():
    return [
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7],
        [6, 7, 10, 9],
        [7, 8, 11, 10],
    ]


def quad_face_7_5():
    return [
        [0, 1, 8, 7],
        [1, 2, 9, 8],
        [2, 3, 10, 9],
        [3, 4, 11, 10],
        [4, 5, 12, 11],
        [5, 6, 13, 12],
        [7, 8, 15, 14],
        [8, 9, 16, 15],
        [9, 10, 17, 16],
        [10, 11, 18, 17],
        [11, 12, 19, 18],
        [12, 13, 20, 19],
        [14, 15, 22, 21],
        [15, 16, 23, 22],
        [16, 17, 24, 23],
        [17, 18, 25, 24],
        [18, 19, 26, 25],
        [19, 20, 27, 26],
        [21, 22, 29, 28],
        [22, 23, 30, 29],
        [23, 24, 31, 30],
        [24, 25, 32, 31],
        [25, 26, 33, 32],
        [26, 27, 34, 33],
    ]


def hexa_volume_2_3_4():
    return [
        [0, 1, 3, 2, 6, 7, 9, 8],
        [2, 3, 5, 4, 8, 9, 11, 10],
        [6, 7, 9, 8, 12, 13, 15, 14],
        [8, 9, 11, 10, 14, 15, 17, 16],
        [12, 13, 15, 14, 18, 19, 21, 20],
        [14, 15, 17, 16, 20, 21, 23, 22],
    ]


def hexa_volume_5_7_3():
    return [
        [0, 1, 6, 5, 35, 36, 41, 40],
        [1, 2, 7, 6, 36, 37, 42, 41],
        [2, 3, 8, 7, 37, 38, 43, 42],
        [3, 4, 9, 8, 38, 39, 44, 43],
        [5, 6, 11, 10, 40, 41, 46, 45],
        [6, 7, 12, 11, 41, 42, 47, 46],
        [7, 8, 13, 12, 42, 43, 48, 47],
        [8, 9, 14, 13, 43, 44, 49, 48],
        [10, 11, 16, 15, 45, 46, 51, 50],
        [11, 12, 17, 16, 46, 47, 52, 51],
        [12, 13, 18, 17, 47, 48, 53, 52],
        [13, 14, 19, 18, 48, 49, 54, 53],
        [15, 16, 21, 20, 50, 51, 56, 55],
        [16, 17, 22, 21, 51, 52, 57, 56],
        [17, 18, 23, 22, 52, 53, 58, 57],
        [18, 19, 24, 23, 53, 54, 59, 58],
        [20, 21, 26, 25, 55, 56, 61, 60],
        [21, 22, 27, 26, 56, 57, 62, 61],
        [22, 23, 28, 27, 57, 58, 63, 62],
        [23, 24, 29, 28, 58, 59, 64, 63],
        [25, 26, 31, 30, 60, 61, 66, 65],
        [26, 27, 32, 31, 61, 62, 67, 66],
        [27, 28, 33, 32, 62, 63, 68, 67],
        [28, 29, 34, 33, 63, 64, 69, 68],
        [35, 36, 41, 40, 70, 71, 76, 75],
        [36, 37, 42, 41, 71, 72, 77, 76],
        [37, 38, 43, 42, 72, 73, 78, 77],
        [38, 39, 44, 43, 73, 74, 79, 78],
        [40, 41, 46, 45, 75, 76, 81, 80],
        [41, 42, 47, 46, 76, 77, 82, 81],
        [42, 43, 48, 47, 77, 78, 83, 82],
        [43, 44, 49, 48, 78, 79, 84, 83],
        [45, 46, 51, 50, 80, 81, 86, 85],
        [46, 47, 52, 51, 81, 82, 87, 86],
        [47, 48, 53, 52, 82, 83, 88, 87],
        [48, 49, 54, 53, 83, 84, 89, 88],
        [50, 51, 56, 55, 85, 86, 91, 90],
        [51, 52, 57, 56, 86, 87, 92, 91],
        [52, 53, 58, 57, 87, 88, 93, 92],
        [53, 54, 59, 58, 88, 89, 94, 93],
        [55, 56, 61, 60, 90, 91, 96, 95],
        [56, 57, 62, 61, 91, 92, 97, 96],
        [57, 58, 63, 62, 92, 93, 98, 97],
        [58, 59, 64, 63, 93, 94, 99, 98],
        [60, 61, 66, 65, 95, 96, 101, 100],
        [61, 62, 67, 66, 96, 97, 102, 101],
        [62, 63, 68, 67, 97, 98, 103, 102],
        [63, 64, 69, 68, 98, 99, 104, 103],
    ]


def hexa_volume_4_8_2():
    return [
        [0, 1, 5, 4, 32, 33, 37, 36],
        [1, 2, 6, 5, 33, 34, 38, 37],
        [2, 3, 7, 6, 34, 35, 39, 38],
        [4, 5, 9, 8, 36, 37, 41, 40],
        [5, 6, 10, 9, 37, 38, 42, 41],
        [6, 7, 11, 10, 38, 39, 43, 42],
        [8, 9, 13, 12, 40, 41, 45, 44],
        [9, 10, 14, 13, 41, 42, 46, 45],
        [10, 11, 15, 14, 42, 43, 47, 46],
        [12, 13, 17, 16, 44, 45, 49, 48],
        [13, 14, 18, 17, 45, 46, 50, 49],
        [14, 15, 19, 18, 46, 47, 51, 50],
        [16, 17, 21, 20, 48, 49, 53, 52],
        [17, 18, 22, 21, 49, 50, 54, 53],
        [18, 19, 23, 22, 50, 51, 55, 54],
        [20, 21, 25, 24, 52, 53, 57, 56],
        [21, 22, 26, 25, 53, 54, 58, 57],
        [22, 23, 27, 26, 54, 55, 59, 58],
        [24, 25, 29, 28, 56, 57, 61, 60],
        [25, 26, 30, 29, 57, 58, 62, 61],
        [26, 27, 31, 30, 58, 59, 63, 62],
    ]


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


def test_tet_to_tri():
    query = randint(0, 1000, (100, 4))
    expected = []
    for q in query:
        # this is connectivity stated in docstring
        expected.append(q[[0, 2, 1, 1, 3, 0, 2, 3, 1, 3, 2, 0]])

    expected = np.vstack(expected).reshape(-1, 3)

    assert (gus.utils.connec.tet_to_tri(query) == expected).all()


def test_hexa_to_quad():
    query = randint(0, 1000, (100, 8))
    expected = []
    for q in query:
        # this is connectivity stated in docstring
        expected.append(
            q[
                [
                    1,
                    0,
                    3,
                    2,
                    0,
                    1,
                    5,
                    4,
                    1,
                    2,
                    6,
                    5,
                    2,
                    3,
                    7,
                    6,
                    3,
                    0,
                    4,
                    7,
                    4,
                    5,
                    6,
                    7,
                ]
            ]
        )

    expected = np.vstack(expected).reshape(-1, 4)

    assert (gus.utils.connec.hexa_to_quad(query) == expected).all()


def test_faces_to_edges():
    # this actually should work for any number of nodes
    for i in range(3, 10):
        query = randint(0, 1000, (100, i))

        expected = []
        for q in query:
            # this should connect nodes from 0 to last and close it
            edges = [[q[0]]]
            for qq in q[1:]:
                edges[-1].append(qq)
                edges.append([qq])
            edges[-1].append(q[0])
            expected.append(edges)

    expected = np.vstack(expected).reshape(-1, 2)

    assert (gus.utils.connec.faces_to_edges(query) == expected).all()


def test_make_quad_faces():
    """
    checks against reference value. Feel free to extend!
    """
    assert (gus.utils.connec.make_quad_faces([3, 4]) == quad_face_3_4()).all()
    assert (gus.utils.connec.make_quad_faces([7, 5]) == quad_face_7_5()).all()


def test_make_hexa_volumes():
    """
    checks against reference value. Feel free to extend!
    """
    assert (
        gus.utils.connec.make_hexa_volumes([4, 8, 2]) == hexa_volume_4_8_2()
    ).all()
    assert (
        gus.utils.connec.make_hexa_volumes([2, 3, 4]) == hexa_volume_2_3_4()
    ).all()
    assert (
        gus.utils.connec.make_hexa_volumes([5, 7, 3]) == hexa_volume_5_7_3()
    ).all()
