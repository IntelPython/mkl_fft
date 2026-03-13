# This file includes tests from dask.fft module:
# https://github.com/dask/dask/blob/main/dask/array/tests/test_fft.py

import contextlib
from itertools import combinations_with_replacement

import dask
import dask.array as da
import numpy as np
import pytest
from dask.array.numpy_compat import NUMPY_GE_200
from dask.array.utils import assert_eq, same_keys

import mkl_fft.interfaces.dask_fft as dask_fft

requires_dask_2024_8_2 = pytest.mark.skipif(
    dask.__version__ < "2024.8.2",
    reason="norm kwarg requires Dask >= 2024.8.2",
)

all_1d_funcnames = ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"]

all_nd_funcnames = [
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
]

if not da._array_expr_enabled():

    nparr = np.arange(100).reshape(10, 10)
    darr = da.from_array(nparr, chunks=(1, 10))
    darr2 = da.from_array(nparr, chunks=(10, 1))
    darr3 = da.from_array(nparr, chunks=(10, 10))


@pytest.mark.parametrize("funcname", all_1d_funcnames)
def test_cant_fft_chunked_axis(funcname):
    da_fft = getattr(dask_fft, funcname)

    bad_darr = da.from_array(nparr, chunks=(5, 5))
    for i in range(bad_darr.ndim):
        with pytest.raises(ValueError):
            da_fft(bad_darr, axis=i)


@pytest.mark.parametrize("funcname", all_1d_funcnames)
def test_fft(funcname):
    da_fft = getattr(dask_fft, funcname)
    np_fft = getattr(np.fft, funcname)

    # pylint: disable=possibly-used-before-assignment
    assert_eq(da_fft(darr), np_fft(nparr))


@pytest.mark.parametrize("funcname", all_nd_funcnames)
def test_fft2n_shapes(funcname):
    da_fft = getattr(dask_fft, funcname)
    np_fft = getattr(np.fft, funcname)

    # pylint: disable=possibly-used-before-assignment
    assert_eq(da_fft(darr3), np_fft(nparr))
    assert_eq(
        da_fft(darr3, (8, 9), axes=(1, 0)), np_fft(nparr, (8, 9), axes=(1, 0))
    )
    assert_eq(
        da_fft(darr3, (12, 11), axes=(1, 0)),
        np_fft(nparr, (12, 11), axes=(1, 0)),
    )

    if NUMPY_GE_200 and funcname.endswith("fftn"):
        ctx = pytest.warns(
            DeprecationWarning,
            match="`axes` should not be `None` if `s` is not `None`",
        )
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        expect = np_fft(nparr, (8, 9))
    with ctx:
        actual = da_fft(darr3, (8, 9))
    assert_eq(expect, actual)


@requires_dask_2024_8_2
@pytest.mark.parametrize("funcname", all_1d_funcnames)
def test_fft_n_kwarg(funcname):
    da_fft = getattr(dask_fft, funcname)
    np_fft = getattr(np.fft, funcname)

    assert_eq(da_fft(darr, 5), np_fft(nparr, 5))
    assert_eq(da_fft(darr, 13), np_fft(nparr, 13))
    assert_eq(
        da_fft(darr, 13, norm="backward"), np_fft(nparr, 13, norm="backward")
    )
    assert_eq(da_fft(darr, 13, norm="ortho"), np_fft(nparr, 13, norm="ortho"))
    assert_eq(
        da_fft(darr, 13, norm="forward"), np_fft(nparr, 13, norm="forward")
    )
    # pylint: disable=possibly-used-before-assignment
    assert_eq(da_fft(darr2, axis=0), np_fft(nparr, axis=0))
    assert_eq(da_fft(darr2, 5, axis=0), np_fft(nparr, 5, axis=0))
    assert_eq(
        da_fft(darr2, 13, axis=0, norm="backward"),
        np_fft(nparr, 13, axis=0, norm="backward"),
    )
    assert_eq(
        da_fft(darr2, 12, axis=0, norm="ortho"),
        np_fft(nparr, 12, axis=0, norm="ortho"),
    )
    assert_eq(
        da_fft(darr2, 12, axis=0, norm="forward"),
        np_fft(nparr, 12, axis=0, norm="forward"),
    )


@pytest.mark.parametrize("funcname", all_1d_funcnames)
def test_fft_consistent_names(funcname):
    da_fft = getattr(dask_fft, funcname)

    assert same_keys(da_fft(darr, 5), da_fft(darr, 5))
    assert same_keys(da_fft(darr2, 5, axis=0), da_fft(darr2, 5, axis=0))
    assert not same_keys(da_fft(darr, 5), da_fft(darr, 13))


@pytest.mark.parametrize("funcname", all_nd_funcnames)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nd_ffts_axes(funcname, dtype):
    np_fft = getattr(np.fft, funcname)
    da_fft = getattr(dask_fft, funcname)

    shape = (7, 8, 9)
    chunk_size = (3, 3, 3)
    a = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    d = da.from_array(a, chunks=chunk_size)

    for num_axes in range(1, d.ndim):
        for axes in combinations_with_replacement(range(d.ndim), num_axes):
            cs = list(chunk_size)
            for i in axes:
                cs[i] = shape[i]
            d2 = d.rechunk(cs)
            if len(set(axes)) < len(axes):
                with pytest.raises(ValueError):
                    da_fft(d2, axes=axes)
            else:
                r = da_fft(d2, axes=axes)
                er = np_fft(a, axes=axes)
                if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
                    check_dtype = True
                    assert r.dtype == er.dtype
                else:
                    check_dtype = False
                assert r.shape == er.shape

                assert_eq(r, er, check_dtype=check_dtype, rtol=1e-6, atol=1e-4)
