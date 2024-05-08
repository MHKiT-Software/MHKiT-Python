import mhkit.dolfyn.io.api as io
from mhkit.dolfyn import time
from xarray.testing import assert_allclose as _assert_allclose
from os.path import abspath, dirname, join, normpath, relpath
import numpy as np


def rfnm(filename):
    testdir = dirname(abspath(__file__))
    datadir = normpath(
        join(testdir, relpath("../../../examples/data/dolfyn/test_data/"))
    )
    return datadir + "/" + filename


def exdt(filename):
    testdir = dirname(abspath(__file__))
    exdir = normpath(join(testdir, relpath("../../../examples/data/dolfyn/")))
    return exdir + "/" + filename


def assert_allclose(dat0, dat1, *args, **kwargs):
    # For problematic time check
    names = []
    for v in dat0.variables:
        if np.issubdtype(dat0[v].dtype, np.datetime64):
            dat0[v] = time.dt642epoch(dat0[v])
            dat1[v] = time.dt642epoch(dat1[v])
            names.append(v)
    # Check coords and data_vars
    _assert_allclose(dat0, dat1, *args, **kwargs)
    # Check attributes
    for nm in dat0.attrs:
        assert dat0.attrs[nm] == dat1.attrs[nm], (
            "The " + nm + " attribute does not match."
        )
    # If test debugging
    for v in names:
        dat0[v] = time.epoch2dt64(dat0[v])
        dat1[v] = time.epoch2dt64(dat1[v])


def load_netcdf(name, *args, **kwargs):
    return io.load(rfnm(name), *args, **kwargs)


def save_netcdf(data, name, *args, **kwargs):
    io.save(data, rfnm(name), *args, **kwargs)


def load_matlab(name, *args, **kwargs):
    return io.load_mat(rfnm(name), *args, **kwargs)


def save_matlab(data, name, *args, **kwargs):
    io.save_mat(data, rfnm(name), *args, **kwargs)
