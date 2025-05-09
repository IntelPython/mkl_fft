#!/usr/bin/env python
# Copyright (c) 2017, Intel Corporation
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

"""
An interface for FFT module of NumPy (`numpy.fft`) that uses OneMKL FFT
in the backend.
"""

import re
import warnings

import numpy as np

import mkl_fft

from .._fft_utils import _compute_fwd_scale, _swap_direction
from ._float_utils import _downcast_float128_array

__all__ = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
]


# copied with modifications from:
# https://github.com/numpy/numpy/blob/main/numpy/fft/_pocketfft.py
def _cook_nd_args(a, s=None, axes=None, invreal=False):
    if s is None:
        shapeless = True
        if axes is None:
            s = list(a.shape)
        else:
            s = np.take(a.shape, axes)
    else:
        shapeless = False
    s = list(s)
    if axes is None:
        if not shapeless and np.lib.NumpyVersion(np.__version__) >= "2.0.0":
            msg = (
                "`axes` should not be `None` if `s` is not `None` "
                "(Deprecated in NumPy 2.0). In a future version of NumPy, "
                "this will raise an error and `s[i]` will correspond to "
                "the size along the transformed axis specified by "
                "`axes[i]`. To retain current behaviour, pass a sequence "
                "[0, ..., k-1] to `axes` for an array of dimension k."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    if None in s and np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        msg = (
            "Passing an array containing `None` values to `s` is "
            "deprecated in NumPy 2.0 and will raise an error in "
            "a future version of NumPy. To use the default behaviour "
            "of the corresponding 1-D transform, pass the value matching "
            "the default for its `n` parameter. To use the default "
            "behaviour for every axis, the `s` argument can be omitted."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
    # use the whole input array along axis `i` if `s[i] == -1 or None`
    s = [a.shape[_a] if _s in [-1, None] else _s for _s, _a in zip(s, axes)]

    return s, axes


def _trycall(func, args, kwrds):
    try:
        res = func(*args, **kwrds)
    except ValueError as ve:
        if len(ve.args) == 1:
            if re.match("^Dimension n", ve.args[0]):
                raise ValueError("Invalid number of FFT data points")
        raise ve
    return res


def fft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional discrete Fourier Transform.

    For full documentation refer to `numpy.fft.fft`.

    """
    x = _downcast_float128_array(a)
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])

    return _trycall(
        mkl_fft.fft, (x,), {"n": n, "axis": axis, "fwd_scale": fsc, "out": out}
    )


def ifft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    For full documentation refer to `numpy.fft.ifft`.

    """
    x = _downcast_float128_array(a)
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])

    return _trycall(
        mkl_fft.ifft, (x,), {"n": n, "axis": axis, "fwd_scale": fsc, "out": out}
    )


def fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional discrete Fourier Transform

    For full documentation refer to `numpy.fft.fft2`.

    """
    return fftn(a, s=s, axes=axes, norm=norm, out=out)


def ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.

    For full documentation refer to `numpy.fft.ifft2`.

    """
    return ifftn(a, s=s, axes=axes, norm=norm, out=out)


def fftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the N-dimensional discrete Fourier Transform.

    For full documentation refer to `numpy.fft.fftn`.

    """
    x = _downcast_float128_array(a)
    s, axes = _cook_nd_args(x, s, axes)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    return _trycall(
        mkl_fft.fftn, (x,), {"s": s, "axes": axes, "fwd_scale": fsc, "out": out}
    )


def ifftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the N-dimensional inverse discrete Fourier Transform.

    For full documentation refer to `numpy.fft.ifftn`.

    """
    x = _downcast_float128_array(a)
    s, axes = _cook_nd_args(x, s, axes)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    return _trycall(
        mkl_fft.ifftn,
        (x,),
        {"s": s, "axes": axes, "fwd_scale": fsc, "out": out},
    )


def rfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional discrete Fourier Transform for real input.

    For full documentation refer to `numpy.fft.rfft`.

    """
    x = _downcast_float128_array(a)
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])

    return _trycall(
        mkl_fft.rfft, (x,), {"n": n, "axis": axis, "fwd_scale": fsc, "out": out}
    )


def irfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the inverse of `rfft`.

    For full documentation refer to `numpy.fft.irfft`.

    """
    x = _downcast_float128_array(a)
    fsc = _compute_fwd_scale(norm, n, 2 * (x.shape[axis] - 1))

    return _trycall(
        mkl_fft.irfft,
        (x,),
        {"n": n, "axis": axis, "fwd_scale": fsc, "out": out},
    )


def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional FFT of a real array.

    For full documentation refer to `numpy.fft.rfft2`.

    """
    return rfftn(a, s=s, axes=axes, norm=norm, out=out)


def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the inverse FFT of `rfft2`.

    For full documentation refer to `numpy.fft.irfft2`.

    """
    return irfftn(a, s=s, axes=axes, norm=norm, out=out)


def rfftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the N-dimensional discrete Fourier Transform for real input.

    For full documentation refer to `numpy.fft.rfftn`.

    """
    x = _downcast_float128_array(a)
    s, axes = _cook_nd_args(x, s, axes)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    return _trycall(
        mkl_fft.rfftn,
        (x,),
        {"s": s, "axes": axes, "fwd_scale": fsc, "out": out},
    )


def irfftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the inverse of `rfftn`.

    For full documentation refer to `numpy.fft.irfftn`.

    """

    x = _downcast_float128_array(a)
    s, axes = _cook_nd_args(x, s, axes, invreal=True)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    return _trycall(
        mkl_fft.irfftn,
        (x,),
        {"s": s, "axes": axes, "fwd_scale": fsc, "out": out},
    )


def hfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the FFT of a signal which has Hermitian symmetry,
    i.e., a real spectrum..

    For full documentation refer to `numpy.fft.hfft`.

    """
    norm = _swap_direction(norm)
    x = _downcast_float128_array(a)
    fsc = _compute_fwd_scale(norm, n, 2 * (x.shape[axis] - 1))

    return _trycall(
        mkl_fft.irfft,
        (np.conjugate(x),),
        {"n": n, "axis": axis, "fwd_scale": fsc, "out": out},
    )


def ihfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the inverse FFT of a signal which has Hermitian symmetry.

    For full documentation refer to `numpy.fft.ihfft`.

    """
    norm = _swap_direction(norm)
    x = _downcast_float128_array(a)
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])

    result = _trycall(
        mkl_fft.rfft, (x,), {"n": n, "axis": axis, "fwd_scale": fsc, "out": out}
    )

    np.conjugate(result, out=result)
    return result
