import numpy as np
import unittest
import mhkit.loads as loads
from numpy.testing import assert_allclose


class TestExtreme(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.t, self.signal = self._example_waveform(self)

    def _example_waveform(self):
        # Create simple wave form to analyse.
        # This has been created to perform
        # a simple independent calcuation that
        # the mhkit functions can be tested against.

        A = np.array([0.5, 0.6, 0.3])
        T = np.array([3, 2, 1])
        w = 2 * np.pi / T

        t = np.linspace(0, 4.5, 100)

        signal = np.zeros(t.size)
        for i in range(A.size):
            signal += A[i] * np.sin(w[i] * t)

        return t, signal

    def _example_crest_analysis(self, t, signal):
        # NB: This only works due to the construction
        # of our test signal. It is not suitable as
        # a general approach.
        grad = np.diff(signal)

        # +1 to get the index at turning point
        turning_points = np.flatnonzero(grad[1:] * grad[:-1] < 0) + 1

        crest_inds = turning_points[signal[turning_points] > 0]
        crests = signal[crest_inds]

        return crests, crest_inds

    def test_global_peaks(self):
        peaks_t, peaks_val = loads.extreme.global_peaks(self.t, self.signal)

        test_crests, test_crests_ind = self._example_crest_analysis(self.t, self.signal)

        assert_allclose(peaks_t, self.t[test_crests_ind])
        assert_allclose(peaks_val, test_crests)
