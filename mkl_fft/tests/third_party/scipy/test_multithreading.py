# This file includes tests from scipy.fft module:
# https://github.com/scipy/scipy/blob/main/scipy/fft/tests/test_multithreading.py.py

import multiprocessing
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mkl_fft.interfaces.scipy_fft as fft


@pytest.fixture(scope="module")
def x():
    return np.random.randn(512, 128)  # Must be large enough to qualify for mt


@pytest.mark.parametrize(
    "func",
    [
        fft.fft,
        fft.ifft,
        fft.fft2,
        fft.ifft2,
        fft.fftn,
        fft.ifftn,
        fft.rfft,
        fft.irfft,
        fft.rfft2,
        fft.irfft2,
        fft.rfftn,
        fft.irfftn,
        # TODO: fft.hfft, fft.ihfft, fft.hfft2, fft.ihfft2, fft.hfftn, fft.ihfftn,
        # TODO: fft.dct, fft.idct, fft.dctn, fft.idctn,
        # TODO: fft.dst, fft.idst, fft.dstn, fft.idstn,
    ],
)
@pytest.mark.parametrize("workers", [2, -1])
def test_threaded_same(x, func, workers):
    expected = func(x, workers=1)
    actual = func(x, workers=workers)
    assert_allclose(actual, expected)


def _mt_fft(x):
    return fft.fft(x, workers=2)


@pytest.mark.slow
def test_mixed_threads_processes(x):
    # Test that the fft threadpool is safe to use before & after fork

    expect = fft.fft(x, workers=2)

    with multiprocessing.Pool(2) as p:
        res = p.map(_mt_fft, [x for _ in range(4)])

    for r in res:
        assert_allclose(r, expect)

    fft.fft(x, workers=2)


def test_invalid_workers(x):
    cpus = os.cpu_count()

    fft.ifft([1], workers=-cpus)

    with pytest.raises(ValueError, match="workers must not be zero"):
        fft.fft(x, workers=0)

    with pytest.raises(ValueError, match="workers value out of range"):
        fft.ifft(x, workers=-cpus - 1)


@pytest.mark.skip()
def test_set_get_workers():
    cpus = os.cpu_count()
    assert fft.get_workers() == 1
    with fft.set_workers(4):
        assert fft.get_workers() == 4

        with fft.set_workers(-1):
            assert fft.get_workers() == cpus

        assert fft.get_workers() == 4

    assert fft.get_workers() == 1

    with fft.set_workers(-cpus):
        assert fft.get_workers() == 1


@pytest.mark.skip("mkl_fft does not validate workers")
def test_set_workers_invalid():

    with pytest.raises(ValueError, match="workers must not be zero"):
        with fft.set_workers(0):
            pass

    with pytest.raises(ValueError, match="workers value out of range"):
        with fft.set_workers(-os.cpu_count() - 1):
            pass
