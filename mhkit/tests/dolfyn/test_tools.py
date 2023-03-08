import mhkit.dolfyn.tools.misc as tools
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import unittest


class tools_testcase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.array = np.arange(10, dtype=float)
        self.nan = np.zeros(3)*np.NaN

    @classmethod
    def tearDownClass(self):
        pass

    def test_detrend_array(self):
        d = tools.detrend_array(self.array)
        assert_allclose(d, np.zeros(10), atol=1e-10)

    def test_group(self):
        array = np.concatenate((self.array, self.array))
        d = tools.group(array)

        out = np.array([slice(1, 20, None)], dtype=object)
        assert_equal(d, out)

    def test_slice(self):
        tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                          [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
        out = np.zeros((3, 3, 3))
        slices = list()
        for slc in tools.slice1d_along_axis((3, 3, 3), axis=-1):
            slices.append(slc)
            out[slc] = tensor[slc]

        slc_out = [(0, 0, slice(None, None, None)),
                   (0, 1, slice(None, None, None)),
                   (0, 2, slice(None, None, None)),
                   (1, 0, slice(None, None, None)),
                   (1, 1, slice(None, None, None)),
                   (1, 2, slice(None, None, None)),
                   (2, 0, slice(None, None, None)),
                   (2, 1, slice(None, None, None)),
                   (2, 2, slice(None, None, None))]

        assert_equal(slc_out, slices)
        assert_allclose(tensor, out, atol=1e-10)

    def test_fillgaps(self):
        arr = np.concatenate((self.array, self.nan, self.array))
        d1 = tools.fillgaps(arr.copy())
        d2 = tools.fillgaps(arr.copy(), maxgap=1)

        out1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6.75, 4.5, 2.25,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        out2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan, np.nan,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        assert_allclose(d1, out1, atol=1e-10)
        assert_allclose(d2, out2, atol=1e-10)

    def test_interpgaps(self):
        arr = np.concatenate((self.array, self.nan, self.array, self.nan))

        t = np.arange(0, arr.shape[0], 0.1)
        d1 = tools.interpgaps(arr.copy(), t, extrapFlg=True)
        d2 = tools.interpgaps(arr.copy(), t, maxgap=1)

        out1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6.75, 4.5, 2.25,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9])
        out2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan, np.nan,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan, np.nan])

        assert_allclose(d1, out1, atol=1e-10)
        assert_allclose(d2, out2, atol=1e-10)

    def test_medfiltnan(self):
        arr = np.concatenate((self.array, self.nan, self.array))
        a = np.concatenate((arr[None, :], arr[None, :]), axis=0)

        d = tools.medfiltnan(a, [1, 5], thresh=3)

        out = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, np.nan, np.nan, np.nan, 2, 3, 4, 5,
                         6, 7, 7, 7],
                        [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, np.nan, np.nan, np.nan, 2, 3, 4, 5,
                         6, 7, 7, 7]])

        assert_allclose(d, out, atol=1e-10)

    def test_deg_conv(self):
        d = tools.convert_degrees(self.array)

        out = np.array([90., 89., 88., 87., 86., 85., 84., 83., 82., 81.])

        assert_allclose(d, out, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
