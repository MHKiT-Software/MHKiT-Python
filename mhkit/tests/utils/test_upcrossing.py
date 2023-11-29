from mhkit.utils import upcrossing, peaks, troughs, heights, periods, custom
import unittest
from numpy.testing import assert_allclose
import numpy as np
from scipy.optimize import fsolve


class TestUpcrossing(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.t = np.linspace(0, 4, 1000)

        self.signal = self._example_waveform(self, self.t)

        # Approximiate points for the zero crossing,
        # used as starting points in numerical
        # solution.
        self.zero_cross_approx = [0, 2.1, 3, 3.8]

    def _example_waveform(self, t):
        # Create simple wave form to analyse.
        # This has been created to perform
        # a simple independent calcuation that
        # the mhkit functions can be tested against.

        A = np.array([0.5, 0.6, 0.3])
        T = np.array([3, 2, 1])
        w = 2 * np.pi / T

        signal = np.zeros(t.size)
        for i in range(A.size):
            signal += A[i] * np.sin(w[i] * t)

        return signal

    def _example_analysis(self, t, signal):
        # NB: This only works due to the construction
        # of our test signal. It is not suitable as
        # a general approach.
        grad = np.diff(signal)

        # +1 to get the index at turning point
        turning_points = np.flatnonzero(grad[1:] * grad[:-1] < 0) + 1

        crest_inds = turning_points[signal[turning_points] > 0]
        trough_inds = turning_points[signal[turning_points] < 0]

        crests = signal[crest_inds]
        troughs = signal[trough_inds]

        heights = crests - troughs

        zero_cross = fsolve(self._example_waveform, self.zero_cross_approx)
        periods = np.diff(zero_cross)

        return crests, troughs, heights, periods

    def test_peaks(self):
        want, _, _, _ = self._example_analysis(self.t, self.signal)

        got = peaks(self.t, self.signal)

        assert_allclose(got, want)

    def test_troughs(self):
        _, want, _, _ = self._example_analysis(self.t, self.signal)

        got = troughs(self.t, self.signal)

        assert_allclose(got, want)

    def test_heights(self):
        _, _, want, _ = self._example_analysis(self.t, self.signal)

        got = heights(self.t, self.signal)

        assert_allclose(got, want)

    def test_periods(self):
        _, _, _, want = self._example_analysis(self.t, self.signal)

        got = periods(self.t, self.signal)

        assert_allclose(got, want, rtol=1e-3, atol=1e-3)

    def test_custom(self):
        want, _, _, _ = self._example_analysis(self.t, self.signal)

        # create a similar function to finding the peaks
        def f(ind1, ind2):
            return np.max(self.signal[ind1:ind2])

        got = custom(self.t, self.signal, f)

        assert_allclose(got, want)

    def test_peaks_with_inds(self):
        want, _, _, _ = self._example_analysis(self.t, self.signal)

        inds = upcrossing(self.t, self.signal)

        got = peaks(self.t, self.signal, inds)

        assert_allclose(got, want)

    def test_trough_with_inds(self):
        _, want, _, _ = self._example_analysis(self.t, self.signal)

        inds = upcrossing(self.t, self.signal)

        got = troughs(self.t, self.signal, inds)

        assert_allclose(got, want)

    def test_heights_with_inds(self):
        _, _, want, _ = self._example_analysis(self.t, self.signal)

        inds = upcrossing(self.t, self.signal)

        got = heights(self.t, self.signal, inds)

        assert_allclose(got, want)

    def test_periods_with_inds(self):
        _, _, _, want = self._example_analysis(self.t, self.signal)

        inds = upcrossing(self.t, self.signal)

        got = periods(self.t, self.signal, inds)

        assert_allclose(got, want, rtol=1e-3, atol=1e-3)

    def test_custom_with_inds(self):
        want, _, _, _ = self._example_analysis(self.t, self.signal)

        inds = upcrossing(self.t, self.signal)

        # create a similar function to finding the peaks
        def f(ind1, ind2):
            return np.max(self.signal[ind1:ind2])

        got = custom(self.t, self.signal, f, inds)

        assert_allclose(got, want)
