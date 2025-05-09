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

import mkl_fft

from ._float_utils import _upcast_float16_array

__all__ = ["fft", "ifft", "fftn", "ifftn", "fft2", "ifft2", "rfft", "irfft"]


def fft(a, n=None, axis=-1, overwrite_x=False):
    x = _upcast_float16_array(a)
    return mkl_fft.fft(x, n=n, axis=axis, overwrite_x=overwrite_x)


def ifft(a, n=None, axis=-1, overwrite_x=False):
    x = _upcast_float16_array(a)
    return mkl_fft.ifft(x, n=n, axis=axis, overwrite_x=overwrite_x)


def fftn(a, shape=None, axes=None, overwrite_x=False):
    x = _upcast_float16_array(a)
    return mkl_fft.fftn(x, s=shape, axes=axes, overwrite_x=overwrite_x)


def ifftn(a, shape=None, axes=None, overwrite_x=False):
    x = _upcast_float16_array(a)
    return mkl_fft.ifftn(x, s=shape, axes=axes, overwrite_x=overwrite_x)


def fft2(a, shape=None, axes=(-2, -1), overwrite_x=False):
    x = _upcast_float16_array(a)
    return mkl_fft.fftn(x, s=shape, axes=axes, overwrite_x=overwrite_x)


def ifft2(a, shape=None, axes=(-2, -1), overwrite_x=False):
    x = _upcast_float16_array(a)
    return mkl_fft.ifftn(x, s=shape, axes=axes, overwrite_x=overwrite_x)


def rfft(a, n=None, axis=-1, overwrite_x=False):
    x = _upcast_float16_array(a)
    return mkl_fft.rfftpack(x, n=n, axis=axis, overwrite_x=overwrite_x)


def irfft(a, n=None, axis=-1, overwrite_x=False):
    x = _upcast_float16_array(a)
    return mkl_fft.irfftpack(x, n=n, axis=axis, overwrite_x=overwrite_x)
