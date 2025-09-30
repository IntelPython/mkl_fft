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
FFT helper functions copied from `scipy.fft` (with some modification) to
prevent circular dependencies when patching NumPy.
"""

import numpy as np
from scipy._lib._array_api import array_namespace

__all__ = ["fftshift", "ifftshift", "fftfreq", "rfftfreq"]


def fftfreq(n, d=1.0, *, xp=None, device=None):
    """
    Return the Discrete Fourier Transform sample frequencies.

    For full documentation refer to `scipy.fft.fftfreq`.

    """
    xp = np if xp is None else xp
    if hasattr(xp, "fft"):
        return xp.fft.fftfreq(n, d=d, device=device)
    return np.fft.fftfreq(n, d=d, device=device)


def rfftfreq(n, d=1.0, *, xp=None, device=None):
    """
    Return the Discrete Fourier Transform sample frequencies (for usage with
    `rfft`, `irfft`).

    For full documentation refer to `scipy.fft.rfftfreq`.

    """
    xp = np if xp is None else xp
    if hasattr(xp, "fft"):
        return xp.fft.rfftfreq(n, d=d, device=device)
    return np.fft.rfftfreq(n, d=d, device=device)


def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    For full documentation refer to `scipy.fft.fftshift`.

    """
    xp = array_namespace(x)
    if hasattr(xp, "fft"):
        return xp.fft.fftshift(x, axes=axes)
    x = np.asarray(x)
    y = np.fft.fftshift(x, axes=axes)
    return xp.asarray(y)


def ifftshift(x, axes=None):
    """
    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    For full documentation refer to `scipy.fft.ifftshift`.

    """
    xp = array_namespace(x)
    if hasattr(xp, "fft"):
        return xp.fft.ifftshift(x, axes=axes)
    x = np.asarray(x)
    y = np.fft.ifftshift(x, axes=axes)
    return xp.asarray(y)
