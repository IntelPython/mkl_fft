#!/usr/bin/env python
# Copyright (c) 2025, Intel Corporation
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

import numpy as np

from ._fft_utils import (
    _cook_nd_args,
    _datacopied,
    _fftnd_impl,
    _pad_array,
    _remove_axis,
    _trim_array,
)

# pylint: disable=no-name-in-module
from ._pydfti import (
    _fft1d_impl,
    _rc_fft1d_impl,
    _rc_ifft1d_impl,
    _rr_fft1d_impl,
    _rr_ifft1d_impl,
)

__all__ = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfftpack",
    "irfftpack",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
]


def rfftn(x, s=None, axes=None, fwd_scale=1.0):
    a = np.asarray(x)
    no_trim = (s is None) and (axes is None)
    s, axes = _cook_nd_args(a, s, axes)
    la = axes[-1]
    # trim array, so that rfft avoids doing unnecessary computations
    if not no_trim:
        a = _trim_array(a, s, axes)
    a = _rc_fft1d_impl(a, n=s[-1], axis=la, fsc=fwd_scale)
    if len(s) > 1:
        if not no_trim:
            ss = list(s)
            ss[-1] = a.shape[la]
            a = _pad_array(a, tuple(ss), axes)
        len_axes = len(axes)
        if len(set(axes)) == len_axes and len_axes == a.ndim and len_axes > 2:
            ss, aa = _remove_axis(s, axes, -1)
            ind = [slice(None, None, 1)] * len(s)
            for ii in range(a.shape[la]):
                ind[la] = ii
                tind = tuple(ind)
                a_inp = a[tind]
                a_res = _fftnd_impl(
                    a_inp, s=ss, axes=aa, overwrite_x=True, direction=1
                )
                if a_res is not a_inp:
                    a[tind] = a_res  # copy in place
        else:
            for ii in range(len(axes) - 2, -1, -1):
                a = _fft1d_impl(a, s[ii], axes[ii], overwrite_x=True)
    return a


def irfftn(x, s=None, axes=None, fwd_scale=1.0):
    a = np.asarray(x)
    no_trim = (s is None) and (axes is None)
    s, axes = _cook_nd_args(a, s, axes, invreal=True)
    la = axes[-1]
    if not no_trim:
        a = _trim_array(a, s, axes)
    if len(s) > 1:
        if not no_trim:
            a = _pad_array(a, s, axes)
        ovr_x = True if _datacopied(a, x) else False
        len_axes = len(axes)
        if len(set(axes)) == len_axes and len_axes == a.ndim and len_axes > 2:
            # due to need to write into a, we must copy
            if not ovr_x:
                a = a.copy()
                ovr_x = True
            if not np.issubdtype(a.dtype, np.complexfloating):
                # complex output will be copied to input, copy is needed
                if a.dtype == np.float32:
                    a = a.astype(np.complex64)
                else:
                    a = a.astype(np.complex128)
                ovr_x = True
            ss, aa = _remove_axis(s, axes, -1)
            ind = [slice(None, None, 1)] * len(s)
            for ii in range(a.shape[la]):
                ind[la] = ii
                tind = tuple(ind)
                a_inp = a[tind]
                a_res = _fftnd_impl(
                    a_inp, s=ss, axes=aa, overwrite_x=True, direction=-1
                )
                if a_res is not a_inp:
                    a[tind] = a_res  # copy in place
        else:
            for ii in range(len(axes) - 1):
                a = _fft1d_impl(
                    a, s[ii], axes[ii], overwrite_x=ovr_x, direction=-1
                )
                ovr_x = True
    a = _rc_ifft1d_impl(a, n=s[-1], axis=la, fsc=fwd_scale)
    return a


def fft(x, n=None, axis=-1, overwrite_x=False, fwd_scale=1.0):
    return _fft1d_impl(
        x, n=n, axis=axis, overwrite_x=overwrite_x, direction=+1, fsc=fwd_scale
    )


def ifft(x, n=None, axis=-1, overwrite_x=False, fwd_scale=1.0):
    return _fft1d_impl(
        x, n=n, axis=axis, overwrite_x=overwrite_x, direction=-1, fsc=fwd_scale
    )


def rfft(x, n=None, axis=-1, fwd_scale=1.0):
    return _rc_fft1d_impl(x, n=n, axis=axis, fsc=fwd_scale)


def irfft(x, n=None, axis=-1, fwd_scale=1.0):
    return _rc_ifft1d_impl(x, n=n, axis=axis, fsc=fwd_scale)


def fft2(x, s=None, axes=(-2, -1), overwrite_x=False, fwd_scale=1.0):
    return _fftnd_impl(
        x, s=s, axes=axes, overwrite_x=overwrite_x, direction=+1, fsc=fwd_scale
    )


def ifft2(x, s=None, axes=(-2, -1), overwrite_x=False, fwd_scale=1.0):
    return _fftnd_impl(
        x, s=s, axes=axes, overwrite_x=overwrite_x, direction=-1, fsc=fwd_scale
    )


def fftn(x, s=None, axes=None, overwrite_x=False, fwd_scale=1.0):
    return _fftnd_impl(
        x, s=s, axes=axes, overwrite_x=overwrite_x, direction=+1, fsc=fwd_scale
    )


def ifftn(x, s=None, axes=None, overwrite_x=False, fwd_scale=1.0):
    return _fftnd_impl(
        x, s=s, axes=axes, overwrite_x=overwrite_x, direction=-1, fsc=fwd_scale
    )


def rfft2(x, s=None, axes=(-2, -1), fwd_scale=1.0):
    return rfftn(x, s=s, axes=axes, fwd_scale=fwd_scale)


def irfft2(x, s=None, axes=(-2, -1), fwd_scale=1.0):
    return irfftn(x, s=s, axes=axes, fwd_scale=fwd_scale)


# deprecated functions
def rfftpack(x, n=None, axis=-1, overwrite_x=False, fwd_scale=1.0):
    """Packed real-valued harmonics of FFT of a real sequence x"""
    return _rr_fft1d_impl(
        x, n=n, axis=axis, overwrite_x=overwrite_x, fsc=fwd_scale
    )


def irfftpack(x, n=None, axis=-1, overwrite_x=False, fwd_scale=1.0):
    """IFFT of a real sequence, takes packed real-valued harmonics of FFT"""
    return _rr_ifft1d_impl(
        x, n=n, axis=axis, overwrite_x=overwrite_x, fsc=fwd_scale
    )
