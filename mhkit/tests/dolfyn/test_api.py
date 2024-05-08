from .base import load_netcdf as load, rfnm
import unittest


make_data = False
vec = load("vector_data01.nc")
sig = load("BenchFile01.nc")
rdi = load("RDI_test01.nc")


class api_testcase(unittest.TestCase):
    def test_repr(self):
        _str = []
        for dat, fnm in [
            (vec, rfnm("vector_data01.repr.txt")),
            (sig, rfnm("BenchFile01.repr.txt")),
            (rdi, rfnm("RDI_test01.repr.txt")),
        ]:
            _str = dat.velds.__repr__()
            if make_data:
                with open(fnm, "w") as fl:
                    fl.write(_str)
            else:
                with open(fnm, "r") as fl:
                    test_str = fl.read()
                assert test_str == _str
