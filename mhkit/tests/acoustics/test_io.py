import numpy as np
import unittest
import os
from os.path import abspath, dirname, join, isfile, normpath, relpath

import mhkit.acoustics as acoustics


testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, "..", "..", "..", "examples", "data", "acoustics"))


class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_read_icListen(self):
        file_name = join(datadir, "RBW_6661_20240601_053114.wav")
        td1 = acoustics.io.read_iclisten(file_name)
        td2 = acoustics.io.read_hydrophone(
            file_name, peak_V=3, sensitivity=-177, start_time="2024-06-01T05:31:14"
        )
        # Check time coordinate
        cc = np.array(
            [
                "2024-06-01T05:31:14.000000000",
                "2024-06-01T05:31:14.000001953",
                "2024-06-01T05:31:14.000003906",
                "2024-06-01T05:31:14.000005859",
                "2024-06-01T05:31:14.000007812",
            ],
            dtype="datetime64[ns]",
        )
        # Check data
        cd = np.array([0.31546374, 0.30229832, 0.32229963, 0.3159701, 0.30356423])

        np.testing.assert_allclose(td1.head().values, cd, atol=1e-6)
        np.testing.assert_equal(td1["time"].head().values, cc)
        np.testing.assert_allclose(td2.head().values, cd, atol=1e-6)
        np.testing.assert_equal(td2["time"].head().values, cc)

    def test_read_soundtrap(self):
        file_name = join(datadir, "6247.230204150508.wav")
        td1 = acoustics.io.read_soundtrap(file_name, sensitivity=-177)
        td2 = acoustics.io.read_hydrophone(
            file_name, peak_V=1, sensitivity=-177, start_time="2023-02-04T15:05:08"
        )
        # Check time coordinate
        cc = np.array(
            [
                "2023-02-04T15:05:08.000000000",
                "2023-02-04T15:05:08.000010416",
                "2023-02-04T15:05:08.000020832",
                "2023-02-04T15:05:08.000031249",
                "2023-02-04T15:05:08.000041665",
            ],
            dtype="datetime64[ns]",
        )
        # Check data
        cd = np.array([0.929006, 0.929006, 0.929006, 0.929006, 1.01542517])

        np.testing.assert_allclose(td1.head().values, cd, atol=1e-6)
        np.testing.assert_equal(td1["time"].head().values, cc)
        np.testing.assert_allclose(td2.head().values, cd, atol=1e-6)
        np.testing.assert_equal(td2["time"].head().values, cc)
