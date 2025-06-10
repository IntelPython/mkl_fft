# This file includes tests from scipy.fft module:
# https://github.com/scipy/scipy/blob/main/scipy/fft/tests/test_multithreading.py.py

import multiprocessing

import mkl
import numpy as np
import pytest
from numpy.testing import assert_allclose

try:
    import mkl_fft.interfaces.scipy_fft as fft
except ImportError:
    pytest.skip("This test file needs scipy", allow_module_level=True)


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
        fft.hfft,
        fft.ihfft,
        fft.hfft2,
        fft.ihfft2,
        fft.hfftn,
        fft.ihfftn,
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


# @pytest.mark.slow
def test_mixed_threads_processes(x):
    # Test that the fft threadpool is safe to use before & after fork

    expect = fft.fft(x, workers=2)

    with multiprocessing.Pool(2) as p:
        res = p.map(_mt_fft, [x for _ in range(4)])

    for r in res:
        assert_allclose(r, expect)

    fft.fft(x, workers=2)


def test_invalid_workers(x):
    # cpus = os.cpu_count()
    threads = mkl.get_max_threads()  # pylint: disable=no-member
    # cpus and threads are usually the same but in CI, cpus = 4 and threads = 2
    # SciPy uses `os.cpu_count()` to get the number of workers, while
    # `mkl_fft.interfaces.scipy_fft` uses `mkl.get_max_threads()`

    fft.ifft([1], workers=-threads)

    with pytest.raises(ValueError, match="workers must not be zero"):
        fft.fft(x, workers=0)

    with pytest.raises(ValueError, match="workers value out of range"):
        fft.ifft(x, workers=-threads - 1)


def test_set_get_workers():
    # cpus = os.cpu_count()
    threads = mkl.get_max_threads()  # pylint: disable=no-member

    # default value is max number of threads unlike stock SciPy
    assert fft.get_workers() == threads
    with fft.set_workers(4):
        assert fft.get_workers() == 4

        with fft.set_workers(-1):
            assert fft.get_workers() == threads

        assert fft.get_workers() == 4

    # default value is max number of threads unlike stock SciPy
    assert fft.get_workers() == threads

    with fft.set_workers(-threads):
        assert fft.get_workers() == 1


def test_set_workers_invalid():

    with pytest.raises(ValueError, match="workers must not be zero"):
        with fft.set_workers(0):
            pass

    with pytest.raises(ValueError, match="workers value out of range"):
        # pylint: disable=no-member
        with fft.set_workers(-mkl.get_max_threads() - 1):
            pass
