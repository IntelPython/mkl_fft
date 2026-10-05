# Copyright (c) 2026, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Cross-library equivalence checks for axis and axes dispatch.

``third_party/scipy/test_basic.py::test_fft_with_order`` already checks that
mkl_fft agrees with *itself* across C, Fortran, and non-contiguous layouts. It
does not compare against an external reference, so a dispatch change that is
consistently wrong in every layout passes it.

The defect recorded in ``_fft_utils._iter_complementary`` was exactly that
kind: values correct, but an element placed in the other half of the output
relative to NumPy. These tests therefore use NumPy's own implementation as
the reference.

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

# Reference implementation. mkl_fft.patch_numpy_fft() replaces numpy.fft's
# attributes with mkl_fft's own, so take the unpatched module instead.
try:
    from numpy.fft import _pocketfft as npfft
except ImportError:  # numpy moved it
    import numpy.fft as npfft

if npfft.fftn.__module__.startswith("mkl_fft"):
    raise RuntimeError("reference is mkl_fft; these tests need real numpy.fft")

_SHAPE_3D = (8, 7, 13)
_SHAPE_4D = (4, 5, 6, 7)

_DTYPES = ["float32", "float64", "complex64", "complex128"]

_ORDERS = ["C", "F", "non-contiguous"]


def _cases(funcs, dtypes=_DTYPES):
    """(func, dtype) pairs, dropping the real-input transforms on complex."""
    return [
        (f, dt)
        for f in funcs
        for dt in dtypes
        if not (f.startswith("rfft") and np.dtype(dt).kind == "c")
    ]


# Relative tolerance by input precision. Single-precision transforms of
# random data over these lengths stay well inside 2e-5.
_TOL = {
    "float32": 2e-5,
    "complex64": 2e-5,
    "float64": 1e-12,
    "complex128": 1e-12,
}

_NEG_AXES_3D = [
    (-1,),  # last axis
    (-2,),  # middle axis: per-vector path in the C backend
    (-3,),  # first axis
    (-2, -1),  # the fft2 default
    (-3, -1),
    (-1, -2),  # reversed: picks a different last axis for r2c
    (0, -1),  # mixed sign
    (-3, -2, -1),
]

_AXES_3D = (
    [ax for n in (1, 2, 3) for ax in itertools.combinations(range(3), n)]
    + _NEG_AXES_3D
    + [None]  # every axis
)


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
    want = getattr(npfft, func)(x, axes=axes)
    _check(got, want, dtype)


@pytest.mark.parametrize("func,dtype", _cases(["rfftn", "irfftn"]))
@pytest.mark.parametrize("axes", _AXES_3D)
@pytest.mark.parametrize("order", _ORDERS)
def test_rfftn_axes_subset(func, dtype, axes, order):
    x = _relayout(_make(_SHAPE_3D, dtype), order)
    got = getattr(mkl_fft, func)(x, axes=axes)
    want = getattr(npfft, func)(x, axes=axes)
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
    want = getattr(npfft, func)(x, axis=axis)
    _check(got, want, dtype)


@pytest.mark.parametrize(
    "func,dtype", _cases(["fft", "ifft", "rfft"], ["float64", "complex128"])
)
@pytest.mark.parametrize("axis", range(len(_SHAPE_4D)))
@pytest.mark.parametrize("order", _ORDERS)
def test_fft_axis_4d(func, dtype, axis, order):
    """A rank-4 array has two interior axes, so the per-vector fallback in the
    C backend is exercised twice within one sweep.
    """
    x = _relayout(_make(_SHAPE_4D, dtype), order)
    got = getattr(mkl_fft, func)(x, axis=axis)
    want = getattr(npfft, func)(x, axis=axis)
    _check(got, want, dtype)


@pytest.mark.parametrize("func,dtype", _cases(["rfft", "irfft"]))
@pytest.mark.parametrize("axis", range(len(_SHAPE_3D)))
@pytest.mark.parametrize("order", _ORDERS)
def test_rfft_axis_3d(func, dtype, axis, order):
    x = _relayout(_make(_SHAPE_3D, dtype), order)
    got = getattr(mkl_fft, func)(x, axis=axis)
    want = getattr(npfft, func)(x, axis=axis)
    _check(got, want, dtype)


# norm="ortho" is unitary: it preserves the L2 norm. See gh-336.

_UNITARY_CASES = [
    ((16, 18), (0,)),  # gh-336
    ((16, 18), (1,)),
    ((16, 18), (0, 1)),
    (_SHAPE_3D, (0,)),
    (_SHAPE_3D, (1,)),
    (_SHAPE_3D, (2,)),
    (_SHAPE_3D, (0, 2)),
    (_SHAPE_3D, None),
]

_UNITARY_RTOL = {"complex64": 1e-4, "complex128": 1e-10}


@pytest.mark.parametrize("func", ["fftn", "ifftn"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("shape,axes", _UNITARY_CASES)
def test_ortho_preserves_norm(func, dtype, shape, axes):
    x = _make(shape, dtype)
    x /= np.sqrt((abs(x) ** 2).sum())
    got = getattr(mkl_fft, func)(x, axes=axes, norm="ortho")
    assert np.isclose(
        (abs(got) ** 2).sum(), 1.0, rtol=_UNITARY_RTOL[dtype]
    ), f"norm not preserved: {(abs(got) ** 2).sum()}"


@pytest.mark.parametrize("shape,axes", _UNITARY_CASES)
def test_ortho_roundtrip_is_identity(shape, axes):
    """Insensitive to gh-336: the fwd and bwd scales cancel. Guards only that
    the two compose to the identity.
    """
    x = _make(shape, "complex128")
    fwd = mkl_fft.fftn(x, axes=axes, norm="ortho")
    got = mkl_fft.ifftn(fwd, axes=axes, norm="ortho")
    _check(got, x, "complex128")


# ---------------------------------------------------------------------------
# norm interacts with the scale factor applied at dispatch time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", ["fftn", "ifftn"])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("axes", _AXES_3D)
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_fftn_axes_subset_norm(func, dtype, axes, norm):
    x = _make(_SHAPE_3D, dtype)
    got = getattr(mkl_fft, func)(x, axes=axes, norm=norm)
    want = getattr(npfft, func)(x, axes=axes, norm=norm)
    _check(got, want, dtype)


@pytest.mark.parametrize(
    "func,dtype", _cases(["rfftn", "irfftn"], ["float64", "complex128"])
)
@pytest.mark.parametrize("axes", _AXES_3D)
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_rfftn_axes_subset_norm(func, dtype, axes, norm):
    """Includes ``axes=None``: for c2r the scale basis is the *output* length
    along the last transformed axis, so a full-axes irfftn is normalized over
    ``2 * (n - 1)`` rather than ``n``.
    """
    x = _make(_SHAPE_3D, dtype)
    got = getattr(mkl_fft, func)(x, axes=axes, norm=norm)
    want = getattr(npfft, func)(x, axes=axes, norm=norm)
    _check(got, want, dtype)


# c2c axes=() is covered by test_fftnd.py::test_empty_axes and
# ::test_empty_axes_returns_same_object; only the r2c family is missing there.
@pytest.mark.parametrize("func", ["rfftn", "irfftn"])
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
def test_empty_axes_r2c_raises_like_numpy(func, norm):
    """With no axes there is no last transformed axis to hold the half
    spectrum, so both libraries raise; check the type agrees.
    """
    x = _make(_SHAPE_3D, "float64")
    with pytest.raises(IndexError):
        getattr(npfft, func)(x, axes=(), norm=norm)
    with pytest.raises(IndexError):
        getattr(mkl_fft, func)(x, axes=(), norm=norm)


@pytest.mark.parametrize(
    "func,dtype",
    _cases(["fft2", "ifft2", "rfft2", "irfft2"], ["float64", "complex128"]),
)
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_fft2_on_3d_norm(func, dtype, norm):
    """``fft2`` on a rank-3 array transforms 2 of 3 axes, so it is a subset
    transform even though the caller passed no ``axes``.
    """
    x = _make(_SHAPE_3D, dtype)
    got = getattr(mkl_fft, func)(x, norm=norm)
    want = getattr(npfft, func)(x, norm=norm)
    _check(got, want, dtype)


@pytest.mark.parametrize("func", ["fft", "ifft"])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("axis", range(len(_SHAPE_3D)))
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_fft_axis_norm(func, dtype, axis, norm):
    x = _make(_SHAPE_3D, dtype)
    got = getattr(mkl_fft, func)(x, axis=axis, norm=norm)
    want = getattr(npfft, func)(x, axis=axis, norm=norm)
    _check(got, want, dtype)


# ---------------------------------------------------------------------------
# out= must not change results on any dispatch path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("axes", _AXES_3D)
def test_fftn_axes_subset_out(dtype, axes):
    x = _make(_SHAPE_3D, dtype)
    want = npfft.fftn(x, axes=axes)
    out = np.empty(want.shape, dtype=x.dtype)
    got = mkl_fft.fftn(x, axes=axes, out=out)
    assert got is out, "out= should be returned"
    _check(got, want, dtype)


def _check_out(func, x, dtype, **kwargs):
    """Run *func* with an ``out`` array shaped and typed from the reference."""
    want = getattr(npfft, func)(x, **kwargs)
    out = np.empty(want.shape, dtype=want.dtype)
    got = getattr(mkl_fft, func)(x, out=out, **kwargs)
    assert got is out, "out= should be returned"
    _check(got, want, dtype)


@pytest.mark.parametrize("func", ["fftn", "ifftn"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("axes", [(0,), (2,), (1, 2), None])
@pytest.mark.parametrize("norm", ["forward", "ortho"])
def test_c2c_out_with_norm(func, dtype, axes, norm):
    """The scale is applied while the result is written into ``out``, which is
    the path this fix changes; ``ifftn`` shares it but was never exercised.
    """
    _check_out(func, _make(_SHAPE_3D, dtype), dtype, axes=axes, norm=norm)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("axes", [(0,), (2,), (1, 2), None])
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
def test_rfftn_out_with_norm(dtype, axes, norm):
    """r2c: ``out`` is complex with the last transformed axis reduced to
    ``n // 2 + 1``, a different allocation from the c2c case.
    """
    _check_out("rfftn", _make(_SHAPE_3D, dtype), dtype, axes=axes, norm=norm)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("axes", [(2,), (1, 2), None])
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
def test_irfftn_out_with_norm(dtype, axes, norm):
    """c2r: ``out`` is real with the last transformed axis expanded to
    ``2 * (n - 1)`` -- the length the invreal branch of the scale helper
    computes, so this ties the two together.
    """
    _check_out("irfftn", _make(_SHAPE_3D, dtype), dtype, axes=axes, norm=norm)


# ---------------------------------------------------------------------------
# s= combined with a scaled norm
#
# The scale helper deliberately returns early when s is given, leaving
# _compute_fwd_scale to normalize over prod(s). These lock that branch down.
# axes is always passed explicitly: NumPy deprecated giving s without axes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", ["fftn", "ifftn"])
@pytest.mark.parametrize(
    "axes,s",
    [
        ((0,), (16,)),  # pad one axis
        ((0,), (4,)),  # truncate one axis
        ((1, 2), (10, 20)),  # pad two
        ((1, 2), (4, 6)),  # truncate two
        ((0, 1, 2), (16, 4, 20)),  # pad and truncate together
    ],
)
@pytest.mark.parametrize("norm", ["forward", "ortho"])
def test_c2c_shape_arg_with_norm(func, axes, s, norm):
    """The scale must come from ``prod(s)`` -- the padded or truncated length --
    not from the original axis lengths.
    """
    x = _make(_SHAPE_3D, "complex128")
    got = getattr(mkl_fft, func)(x, s=s, axes=axes, norm=norm)
    want = getattr(npfft, func)(x, s=s, axes=axes, norm=norm)
    _check(got, want, "complex128")


@pytest.mark.parametrize("s", [(8, 7, 20), (8, 7, 10), (8, 7, 24)])
@pytest.mark.parametrize("norm", ["forward", "ortho"])
def test_irfftn_shape_arg_with_norm(s, norm):
    """With ``s`` given, the invreal doubling must *not* be applied: the scale
    normalizes over ``s[-1]``, not ``2 * (x.shape[-1] - 1)``.

    ``s=(8, 7, 24)`` is deliberately ``2 * (13 - 1)``, so a regression that
    ignores ``s`` and falls back to the input-derived basis would still pass
    that one case -- 20 and 10 are what catch it. A regression that instead
    doubles ``s[-1]`` itself is caught by all three.
    """
    x = _make(_SHAPE_3D, "complex128")
    got = mkl_fft.irfftn(x, s=s, axes=(0, 1, 2), norm=norm)
    want = npfft.irfftn(x, s=s, axes=(0, 1, 2), norm=norm)
    _check(got, want, "complex128")


@pytest.mark.parametrize(
    "axes,s", [((1,), (10,)), ((1,), (4,)), ((1, 2), (10, 20))]
)
@pytest.mark.parametrize("norm", ["forward", "ortho"])
def test_rfftn_shape_arg_with_norm(axes, s, norm):
    """Locks the r2c ``s``-given path together with scaling."""
    x = _make(_SHAPE_3D, "float64")
    got = mkl_fft.rfftn(x, s=s, axes=axes, norm=norm)
    want = npfft.rfftn(x, s=s, axes=axes, norm=norm)
    _check(got, want, "float64")


_R2C_AXES_S = [
    ((2,), (20,)),
    ((2,), (10,)),
    ((1, 2), (5, 20)),
    ((0, 2), (10, 24)),
    ((-1,), (20,)),
    ((-2, -1), (5, 20)),
]

_C2C_AXES_S = [
    ((0,), (16,)),
    ((0,), (4,)),
    ((1, 2), (10, 20)),
    ((0, 1, 2), (16, 4, 20)),
]


@pytest.mark.parametrize("func", ["rfftn", "irfftn"])
@pytest.mark.parametrize("axes,s", _R2C_AXES_S)
@pytest.mark.parametrize("norm", ["forward", "ortho"])
def test_r2c_axes_subset_s_and_norm(func, axes, s, norm):
    dtype = "float64" if func == "rfftn" else "complex128"
    x = _make(_SHAPE_3D, dtype)
    got = getattr(mkl_fft, func)(x, s=s, axes=axes, norm=norm)
    want = getattr(npfft, func)(x, s=s, axes=axes, norm=norm)
    _check(got, want, dtype)


@pytest.mark.parametrize("func", ["fftn", "ifftn"])
@pytest.mark.parametrize("axes,s", _C2C_AXES_S)
@pytest.mark.parametrize("norm", ["forward", "ortho"])
def test_out_with_s_and_norm(func, axes, s, norm):
    x = _make(_SHAPE_3D, "complex128")
    _check_out(func, x, "complex128", s=s, axes=axes, norm=norm)


@pytest.mark.parametrize("func", ["fft2", "ifft2", "rfft2", "irfft2"])
@pytest.mark.parametrize("shape", [(16, 18), _SHAPE_3D])
@pytest.mark.parametrize("s", [(8, 9), (32, 36), (8, 36)])
@pytest.mark.parametrize("norm", ["forward", "ortho"])
def test_fft2_family_with_s_and_norm(func, shape, s, norm):
    dtype = "float64" if func == "rfft2" else "complex128"
    x = _make(shape, dtype)
    got = getattr(mkl_fft, func)(x, s=s, norm=norm)
    want = getattr(npfft, func)(x, s=s, norm=norm)
    _check(got, want, dtype)
