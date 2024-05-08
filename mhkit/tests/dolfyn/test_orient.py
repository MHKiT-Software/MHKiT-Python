from .base import load_netcdf as load
from mhkit.dolfyn.rotate.base import euler2orient, orient2euler, quaternion2orient
from mhkit.dolfyn.rotate.api import set_declination
import numpy as np
from numpy.testing import assert_allclose
import unittest


def check_hpr(h, p, r, omatin):
    omat = euler2orient(h, p, r)
    assert_allclose(
        omat,
        omatin,
        atol=1e-13,
        err_msg="Orientation matrix different than expected!\nExpected:\n{}\nGot:\n{}".format(
            np.array(omatin), omat
        ),
    )
    hpr = orient2euler(omat)
    assert_allclose(
        hpr,
        [h, p, r],
        atol=1e-13,
        err_msg="Angles different than specified, orient2euler and euler2orient are "
        "antisymmetric!\nExpected:\n{}\nGot:\n{}".format(
            hpr,
            np.array([h, p, r]),
        ),
    )


class orient_testcase(unittest.TestCase):
    def test_hpr_defs(self):
        """
        These tests confirm that the euler2orient and orient2euler functions
        are consistent, and that they follow the conventions defined in the
        DOLfYN documentation (data-structure.html#heading-pitch-roll), namely:

          - a "ZYX" rotation order. That is, these variables are computed
            assuming that rotation from the earth -> instrument frame happens
            by rotating around the z-axis first (heading), then rotating
            around the y-axis (pitch), then rotating around the x-axis (roll).

          - heading is defined as the direction the x-axis points, positive
            clockwise from North (this is the opposite direction from the
            right-hand-rule around the Z-axis)

          - pitch is positive when the x-axis pitches up (this is opposite the
            right-hand-rule around the Y-axis)

          - roll is positive according to the right-hand-rule around the
            instument's x-axis

        IF YOU MAKE CHANGES TO THESE CONVENTIONS, BE SURE TO UPDATE THE
        DOCUMENTATION.

        """
        check_hpr(
            0,
            0,
            0,
            [
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1],
            ],
        )

        check_hpr(
            90,
            0,
            0,
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        )

        check_hpr(
            90,
            0,
            90,
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ],
        )

        sq2 = 1.0 / np.sqrt(2)
        check_hpr(
            45,
            0,
            0,
            [
                [sq2, sq2, 0],
                [-sq2, sq2, 0],
                [0, 0, 1],
            ],
        )

        check_hpr(
            0,
            45,
            0,
            [
                [0, sq2, sq2],
                [-1, 0, 0],
                [0, -sq2, sq2],
            ],
        )

        check_hpr(
            0,
            0,
            45,
            [
                [0, 1, 0],
                [-sq2, 0, sq2],
                [sq2, 0, sq2],
            ],
        )

        check_hpr(
            90,
            45,
            90,
            [
                [sq2, 0, sq2],
                [-sq2, 0, sq2],
                [0, -1, 0],
            ],
        )

        c30 = np.cos(np.deg2rad(30))
        s30 = np.sin(np.deg2rad(30))
        check_hpr(
            30,
            0,
            0,
            [
                [s30, c30, 0],
                [-c30, s30, 0],
                [0, 0, 1],
            ],
        )

    def test_pr_declination(self):
        # Test to confirm that pitch and roll don't change when you set
        # declination
        declin = 15.37

        dat = load("vector_data_imu01.nc")
        h0, p0, r0 = orient2euler(dat["orientmat"].values)

        set_declination(dat, declin, inplace=True)
        h1, p1, r1 = orient2euler(dat["orientmat"].values)

        assert_allclose(
            p0, p1, atol=1e-5, err_msg="Pitch changes when setting declination"
        )
        assert_allclose(
            r0, r1, atol=1e-5, err_msg="Roll changes when setting declination"
        )
        assert_allclose(
            h0 + declin,
            h1,
            atol=1e-5,
            err_msg="incorrect heading change when " "setting declination",
        )

    def test_q_hpr(self):
        dat = load("Sig1000_IMU.nc")

        dcm = quaternion2orient(dat.quaternions)

        assert_allclose(
            dat.orientmat,
            dcm,
            atol=5e-4,
            err_msg="Disagreement b/t quaternion-calc'd & HPR-calc'd orientmat",
        )


if __name__ == "__main__":
    unittest.main()
