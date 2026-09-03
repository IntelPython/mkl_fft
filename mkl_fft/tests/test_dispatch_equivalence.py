"""Cross-library equivalence checks for axis and axes dispatch.

``third_party/scipy/test_basic.py::test_fft_with_order`` already checks that
mkl_fft agrees with *itself* across C, Fortran, and non-contiguous layouts. It
does not compare against an external reference, so a dispatch change that is
consistently wrong in every layout passes it.

The defect recorded in ``_fft_utils._iter_complementary`` was exactly that
kind: values correct, but an element placed in the other half of the output
relative to NumPy. These tests therefore use ``numpy.fft`` as the reference.

Two deliberate choices:

* Every axis length differs, so an axis permutation cannot produce a
  correctly shaped result and hide behind a shape assertion.
* Output dtype is asserted alongside values, so a dispatch change cannot
  silently upcast.

These cover the paths that dispatch on *which* axes are requested: full-axes
transforms reach the batched N-D descriptor, strict subsets iterate the
complementary axes, and 1-D transforms of rank > 2 arrays are batched only for
the first and last axis.
"""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mkl_fft

_SHAPE_3D = (8, 7, 13)
_SHAPE_4D = (4, 5, 6, 7)

_DTYPES = ["float32", "float64", "complex64", "complex128"]
_REAL_DTYPES = ["float32", "float64"]

_ORDERS = ["C", "F", "non-contiguous"]

# Relative tolerance by input precision. Single-precision transforms of
# random data over these lengths stay well inside 2e-5.
_TOL = {
    "float32": 2e-5,
    "complex64": 2e-5,
    "float64": 1e-12,
    "complex128": 1e-12,
}

# every non-empty subset of the axes of a 3-D array, plus None
_AXES_3D = [
    ax for n in (1, 2, 3) for ax in itertools.combinations(range(3), n)
] + [None]


def _make(shape, dtype, seed=42):
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    if dt.kind == "c":
        x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    else:
        x = rng.standard_normal(shape)
    return x.astype(dt)


def _relayout(x, order):
    """Return *x* laid out as requested; data content may differ by order."""
    if order == "F":
        return np.asfortranarray(x)
    if order == "non-contiguous":
        return x[::-1]
    return np.ascontiguousarray(x)


def _check(got, want, dtype):
    assert got.dtype == want.dtype, f"dtype {got.dtype} != {want.dtype}"
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    tol = _TOL[dtype]
    assert_allclose(
        got, want, rtol=tol, atol=tol * max(1.0, float(np.abs(want).max()))
    )


# ---------------------------------------------------------------------------
# N-D complex transforms over a subset of axes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", ["fftn", "ifftn"])
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("axes", _AXES_3D)
@pytest.mark.parametrize("order", _ORDERS)
def test_fftn_axes_subset(func, dtype, axes, order):
    x = _relayout(_make(_SHAPE_3D, dtype), order)
    got = getattr(mkl_fft, func)(x, axes=axes)
    want = getattr(np.fft, func)(x, axes=axes)
    _check(got, want, dtype)


@pytest.mark.parametrize("func", ["rfftn", "irfftn"])
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("axes", _AXES_3D)
@pytest.mark.parametrize("order", _ORDERS)
def test_rfftn_axes_subset(func, dtype, axes, order):
    if func == "rfftn" and dtype not in _REAL_DTYPES:
        pytest.skip("rfftn takes real input")
    x = _relayout(_make(_SHAPE_3D, dtype), order)
    got = getattr(mkl_fft, func)(x, axes=axes)
    want = getattr(np.fft, func)(x, axes=axes)
    _check(got, want, dtype)


# ---------------------------------------------------------------------------
# 1-D transforms along each axis of a higher-rank array
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", ["fft", "ifft"])
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("axis", range(len(_SHAPE_3D)))
@pytest.mark.parametrize("order", _ORDERS)
def test_fft_axis_3d(func, dtype, axis, order):
    x = _relayout(_make(_SHAPE_3D, dtype), order)
    got = getattr(mkl_fft, func)(x, axis=axis)
    want = getattr(np.fft, func)(x, axis=axis)
    _check(got, want, dtype)


@pytest.mark.parametrize("func", ["fft", "ifft", "rfft"])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("axis", range(len(_SHAPE_4D)))
@pytest.mark.parametrize("order", _ORDERS)
def test_fft_axis_4d(func, dtype, axis, order):
    """A rank-4 array has two interior axes, so the per-vector fallback in the
    C backend is exercised twice within one sweep.
    """
    if func == "rfft" and dtype != "float64":
        pytest.skip("rfft takes real input")
    x = _relayout(_make(_SHAPE_4D, dtype), order)
    got = getattr(mkl_fft, func)(x, axis=axis)
    want = getattr(np.fft, func)(x, axis=axis)
    _check(got, want, dtype)


@pytest.mark.parametrize("func", ["rfft", "irfft"])
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("axis", range(len(_SHAPE_3D)))
@pytest.mark.parametrize("order", _ORDERS)
def test_rfft_axis_3d(func, dtype, axis, order):
    if func == "rfft" and dtype not in _REAL_DTYPES:
        pytest.skip("rfft takes real input")
    x = _relayout(_make(_SHAPE_3D, dtype), order)
    got = getattr(mkl_fft, func)(x, axis=axis)
    want = getattr(np.fft, func)(x, axis=axis)
    _check(got, want, dtype)


# ---------------------------------------------------------------------------
# norm interacts with the scale factor applied at dispatch time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", ["fftn", "ifftn"])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("axes", [(0,), (1,), (2,), (1, 2), None])
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_fftn_axes_subset_norm(func, dtype, axes, norm):
    x = _make(_SHAPE_3D, dtype)
    got = getattr(mkl_fft, func)(x, axes=axes, norm=norm)
    want = getattr(np.fft, func)(x, axes=axes, norm=norm)
    _check(got, want, dtype)


@pytest.mark.parametrize("func", ["rfftn", "irfftn"])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("axes", [(0,), (1,), (2,), (1, 2), None])
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_rfftn_axes_subset_norm(func, dtype, axes, norm):
    """Includes ``axes=None``: for c2r the scale basis is the *output* length
    along the last transformed axis, so a full-axes irfftn is normalized over
    ``2 * (n - 1)`` rather than ``n``.
    """
    if func == "rfftn" and dtype != "float64":
        pytest.skip("rfftn takes real input")
    x = _make(_SHAPE_3D, dtype)
    got = getattr(mkl_fft, func)(x, axes=axes, norm=norm)
    want = getattr(np.fft, func)(x, axes=axes, norm=norm)
    _check(got, want, dtype)


@pytest.mark.parametrize("func", ["fft2", "ifft2", "rfft2", "irfft2"])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_fft2_on_3d_norm(func, dtype, norm):
    """``fft2`` on a rank-3 array transforms 2 of 3 axes, so it is a subset
    transform even though the caller passed no ``axes``.
    """
    if func == "rfft2" and dtype != "float64":
        pytest.skip("rfft2 takes real input")
    x = _make(_SHAPE_3D, dtype)
    got = getattr(mkl_fft, func)(x, norm=norm)
    want = getattr(np.fft, func)(x, norm=norm)
    _check(got, want, dtype)


@pytest.mark.parametrize("func", ["fft", "ifft"])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("axis", range(len(_SHAPE_3D)))
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_fft_axis_norm(func, dtype, axis, norm):
    x = _make(_SHAPE_3D, dtype)
    got = getattr(mkl_fft, func)(x, axis=axis, norm=norm)
    want = getattr(np.fft, func)(x, axis=axis, norm=norm)
    _check(got, want, dtype)


# ---------------------------------------------------------------------------
# out= must not change results on any dispatch path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("axes", _AXES_3D)
def test_fftn_axes_subset_out(dtype, axes):
    x = _make(_SHAPE_3D, dtype)
    want = np.fft.fftn(x, axes=axes)
    out = np.empty(want.shape, dtype=x.dtype)
    got = mkl_fft.fftn(x, axes=axes, out=out)
    assert got is out, "out= should be returned"
    _check(got, want, dtype)
