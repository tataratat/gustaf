import numpy as np

from gustaf.utils.arr import make_c_contiguous


def test_make_c_contiguous_forNoneValues():
    assert make_c_contiguous(None) is None


def test_make_c_contiguous_for1dArray(
    sample_c_contiguous_data_array, expected_c_contiguous_con
):
    generated_array = make_c_contiguous(sample_c_contiguous_data_array)
    assert np.equal(expected_c_contiguous_con, generated_array).all()
    assert generated_array.flags.c_contiguous


def test_make_c_contiguous_forNdArrays(
    sample_c_contiguous_nd_array, expected_c_contiguous_con_nd
):
    generated_array = make_c_contiguous(sample_c_contiguous_nd_array)
    # check if items in array does not change
    assert np.equal(expected_c_contiguous_con_nd, generated_array).all()
    # test if returned array is c-contiguous
    assert generated_array.flags.c_contiguous


def test_make_c_contiguous_changeDtype(sample_c_contiguous_nd_array):
    # test if dtype of array can be change
    assert (
        make_c_contiguous(sample_c_contiguous_nd_array, float).dtype == float
    )
    assert make_c_contiguous(sample_c_contiguous_nd_array, int).dtype == int
    assert make_c_contiguous(sample_c_contiguous_nd_array).dtype == int


def test_make_c_contiguous_nonContiguousArray(sample_c_contiguous_non_con):
    # test if non contiguous arrays are converting
    assert not sample_c_contiguous_non_con.flags.c_contiguous
    assert make_c_contiguous(sample_c_contiguous_non_con).flags.c_contiguous

    # test convert non contiguous into contiguous float array
    assert float == make_c_contiguous(sample_c_contiguous_non_con, float).dtype
    assert make_c_contiguous(
        sample_c_contiguous_non_con, float
    ).flags.c_contiguous

    assert int == make_c_contiguous(sample_c_contiguous_non_con, int).dtype
    assert make_c_contiguous(
        sample_c_contiguous_non_con, int
    ).flags.c_contiguous
