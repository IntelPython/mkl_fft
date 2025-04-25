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

from ._fft_utils import (
    _c2c_fftnd_impl,
    _c2r_fftnd_impl,
    _compute_fwd_scale,
    _r2c_fftnd_impl,
)

# pylint: disable=no-name-in-module
from ._pydfti import _c2c_fft1d_impl, _c2r_fft1d_impl, _r2c_fft1d_impl

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
]


def fft(x, n=None, axis=-1, norm=None, out=None):
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])
    return _c2c_fft1d_impl(x, n=n, axis=axis, out=out, direction=+1, fsc=fsc)


def ifft(x, n=None, axis=-1, norm=None, out=None):
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])
    return _c2c_fft1d_impl(x, n=n, axis=axis, out=out, direction=-1, fsc=fsc)


def fft2(x, s=None, axes=(-2, -1), norm=None, out=None):
    return fftn(x, s=s, axes=axes, norm=norm, out=out)


def ifft2(x, s=None, axes=(-2, -1), norm=None, out=None):
    return ifftn(x, s=s, axes=axes, norm=norm, out=out)


def fftn(x, s=None, axes=None, norm=None, out=None):
    fsc = _compute_fwd_scale(norm, s, x.shape)
    return _c2c_fftnd_impl(x, s=s, axes=axes, out=out, direction=+1, fsc=fsc)


def ifftn(x, s=None, axes=None, norm=None, out=None):
    fsc = _compute_fwd_scale(norm, s, x.shape)
    return _c2c_fftnd_impl(x, s=s, axes=axes, out=out, direction=-1, fsc=fsc)


def rfft(x, n=None, axis=-1, norm=None, out=None):
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])
    return _r2c_fft1d_impl(x, n=n, axis=axis, out=out, fsc=fsc)


def irfft(x, n=None, axis=-1, norm=None, out=None):
    fsc = _compute_fwd_scale(norm, n, 2 * (x.shape[axis] - 1))
    return _c2r_fft1d_impl(x, n=n, axis=axis, out=out, fsc=fsc)


def rfft2(x, s=None, axes=(-2, -1), norm=None, out=None):
    return rfftn(x, s=s, axes=axes, norm=norm, out=out)


def irfft2(x, s=None, axes=(-2, -1), norm=None, out=None):
    return irfftn(x, s=s, axes=axes, norm=norm, out=out)


def rfftn(x, s=None, axes=None, norm=None, out=None):
    fsc = _compute_fwd_scale(norm, s, x.shape)
    return _r2c_fftnd_impl(x, s=s, axes=axes, out=out, fsc=fsc)


def irfftn(x, s=None, axes=None, norm=None, out=None):
    fsc = _compute_fwd_scale(norm, s, x.shape)
    return _c2r_fftnd_impl(x, s=s, axes=axes, out=out, fsc=fsc)
