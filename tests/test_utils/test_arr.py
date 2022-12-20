import unittest
import numpy as np
from gustaf.utils.arr import make_c_contiguous, unique_rows

data_array = np.array([0, 5, 4, 4, 6, 8, 8, 9])
data_nd_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
data_2d_array = np.array([[1, 2, 3], [4, 5, 6]])


class TestUtilsMakeCContingouosFunction(unittest.TestCase):

    def test_make_c_contiguous_forNoneValues(self):
        self.assertEqual(None, make_c_contiguous(None))

    def test_make_c_contiguous_for1dArray(self):
        expected_array = np.ascontiguousarray(data_array)
        generated_array = make_c_contiguous(data_array)

        # check if items in array does not change
        self.assertEqual(expected_array.all(), generated_array.all())
        # test if returned array is c-contiguous
        self.assertTrue(generated_array.flags.c_contiguous)

    def test_make_c_contiguous_forNdArrays(self):
        expected_array = np.ascontiguousarray(data_nd_array)
        generated_array = make_c_contiguous(data_nd_array)

        # check if items in array does not change
        self.assertEqual(expected_array.all(), generated_array.all())
        # test if returned array is c-contiguous
        self.assertTrue(generated_array.flags.c_contiguous)

    def test_make_c_contiguous_changeDtype(self):
        # test if dtype of array can be change
        self.assertEqual(float, make_c_contiguous(data_nd_array, float).dtype)
        self.assertEqual(int, make_c_contiguous(data_nd_array, int).dtype)
        self.assertEqual(int, make_c_contiguous(data_nd_array).dtype)

    def test_make_c_contiguous_nonContiguousArray(self):
        # test if non contiguous arrays are converting
        non_con = data_array.reshape(2, 4).transpose()
        self.assertFalse(non_con.flags.c_contiguous)
        self.assertTrue(make_c_contiguous(non_con).flags.c_contiguous)

        # test convert non contiguous into contiguous float array
        self.assertEqual(float, make_c_contiguous(non_con, float).dtype)
        self.assertTrue(make_c_contiguous(non_con, float).flags.c_contiguous)

        # test convert non contiguous into contiguous int array
        self.assertEqual(int, make_c_contiguous(non_con, int).dtype)
        self.assertTrue(make_c_contiguous(non_con, int).flags.c_contiguous)


class TestUniqueRows(unittest.TestCase):

    def test_unique_rows_throwValueError(self):
        self.assertRaises(ValueError, unique_rows, data_array)


if __name__ == '__main__':
    unittest.main()
