import mhkit.dolfyn.tools as tools
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import unittest


class tools_testcase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.array = np.arange(10, dtype=float)
        self.nan = np.zeros(3) * np.NaN

    @classmethod
    def tearDownClass(self):
        pass

    def test_detrend_array(self):
        d = tools.misc.detrend_array(self.array)
        assert_allclose(d, np.zeros(10), atol=1e-10)

    def test_group(self):
        array = np.concatenate((self.array, self.array))
        d = tools.misc.group(array)

        out = np.array([slice(1, 20, None)], dtype=object)
        assert_equal(d, out)

    def test_slice(self):
        tensor = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
            ]
        )
        out = np.zeros((3, 3, 3))
        slices = list()
        for slc in tools.misc.slice1d_along_axis((3, 3, 3), axis=-1):
            slices.append(slc)
            out[slc] = tensor[slc]

        slc_out = [
            (0, 0, slice(None, None, None)),
            (0, 1, slice(None, None, None)),
            (0, 2, slice(None, None, None)),
            (1, 0, slice(None, None, None)),
            (1, 1, slice(None, None, None)),
            (1, 2, slice(None, None, None)),
            (2, 0, slice(None, None, None)),
            (2, 1, slice(None, None, None)),
            (2, 2, slice(None, None, None)),
        ]

        assert_equal(slc_out, slices)
        assert_allclose(tensor, out, atol=1e-10)

    def test_fillgaps(self):
        arr = np.concatenate((self.array, self.nan, self.array))
        d1 = tools.misc.fillgaps(arr.copy())
        d2 = tools.misc.fillgaps(arr.copy(), maxgap=1)

        out1 = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                6.75,
                4.5,
                2.25,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
            ]
        )
        out2 = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                np.nan,
                np.nan,
                np.nan,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
            ]
        )

        assert_allclose(d1, out1, atol=1e-10)
        assert_allclose(d2, out2, atol=1e-10)

    def test_interpgaps(self):
        arr = np.concatenate((self.array, self.nan, self.array, self.nan))

        t = np.arange(0, arr.shape[0], 0.1)
        d1 = tools.misc.interpgaps(arr.copy(), t, extrapFlg=True)
        d2 = tools.misc.interpgaps(arr.copy(), t, maxgap=1)

        out1 = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                6.75,
                4.5,
                2.25,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                9,
                9,
                9,
            ]
        )
        out2 = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                np.nan,
                np.nan,
                np.nan,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                np.nan,
                np.nan,
                np.nan,
            ]
        )

        assert_allclose(d1, out1, atol=1e-10)
        assert_allclose(d2, out2, atol=1e-10)

    def test_medfiltnan(self):
        arr = np.concatenate((self.array, self.nan, self.array))
        a = np.concatenate((arr[None, :], arr[None, :]), axis=0)

        d = tools.misc.medfiltnan(a, [1, 5], thresh=3)

        out = np.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    7,
                    7,
                    8,
                    9,
                    np.nan,
                    np.nan,
                    np.nan,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    7,
                    7,
                ],
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    7,
                    7,
                    8,
                    9,
                    np.nan,
                    np.nan,
                    np.nan,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    7,
                    7,
                ],
            ]
        )

        assert_allclose(d, out, atol=1e-10)

    def test_deg_conv(self):
        d = tools.misc.convert_degrees(self.array)

        out = np.array([90.0, 89.0, 88.0, 87.0, 86.0, 85.0, 84.0, 83.0, 82.0, 81.0])

        assert_allclose(d, out, atol=1e-10)

    def test_fft_frequency(self):
        fs = 1000  # Sampling frequency
        nfft = 512  # Number of samples in a window

        # Test for full frequency range
        freq_full = tools.fft.fft_frequency(nfft, fs, full=True)
        assert_equal(len(freq_full), nfft)

        # Check symmetry of positive and negative frequencies, ignoring the zero frequency
        positive_freqs = freq_full[1 : int(nfft / 2)]
        negative_freqs = freq_full[int(nfft / 2) + 1 :]
        assert_allclose(positive_freqs, -negative_freqs[::-1])

    def test_stepsize(self):
        # Case 1: l < nfft
        step, nens, nfft = tools.fft._stepsize(100, 200)
        assert_equal((step, nens, nfft), (0, 1, 100))

        # Case 2: l == nfft
        step, nens, nfft = tools.fft._stepsize(200, 200)
        assert_equal((step, nens, nfft), (0, 1, 200))

        # Case 3: l > nfft, no nens
        step, nens, nfft = tools.fft._stepsize(300, 100)
        expected_nens = int(2.0 * 300 / 100)
        expected_step = int((300 - 100) / (expected_nens - 1))
        assert_equal((step, nens, nfft), (expected_step, expected_nens, 100))

        # Case 4: l > nfft, with nens
        step, nens, nfft = tools.fft._stepsize(300, 100, nens=5)
        expected_step = int((300 - 100) / (5 - 1))
        assert_equal((step, nens, nfft), (expected_step, 5, 100))

        # Case 5: l > nfft, with step
        step, nens, nfft = tools.fft._stepsize(300, 100, step=50)
        expected_nens = int((300 - 100) / 50 + 1)
        assert_equal((step, nens, nfft), (50, expected_nens, 100))

        # Case 6: nens is 1
        step, nens, nfft = tools.fft._stepsize(300, 100, nens=1)
        assert_equal((step, nens, nfft), (0, 1, 100))

    def test_cpsd_quasisync_1D(self):
        fs = 1000  # Sample rate
        nfft = 512  # Number of points in the fft

        # Test with signals of same length
        a = np.random.normal(0, 1, 1000)
        b = np.random.normal(0, 1, 1000)
        cpsd = tools.fft.cpsd_quasisync_1D(a, b, nfft, fs)
        self.assertEqual(cpsd.shape, (nfft // 2,))

        # Test with signals of different lengths
        a = np.random.normal(0, 1, 1500)
        b = np.random.normal(0, 1, 1000)
        cpsd = tools.fft.cpsd_quasisync_1D(a, b, nfft, fs)
        self.assertEqual(cpsd.shape, (nfft // 2,))

        # Test with different window types
        for window in [None, 1, "hann"]:
            cpsd = tools.fft.cpsd_quasisync_1D(a, b, nfft, fs, window=window)
            self.assertEqual(cpsd.shape, (nfft // 2,))

        # Test with a custom window
        custom_window = np.hamming(nfft)
        cpsd = tools.fft.cpsd_quasisync_1D(a, b, nfft, fs, window=custom_window)
        self.assertEqual(cpsd.shape, (nfft // 2,))


if __name__ == "__main__":
    unittest.main()
