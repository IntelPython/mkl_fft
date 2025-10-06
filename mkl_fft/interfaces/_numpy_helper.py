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
FFT helper functions copied from `numpy.fft` (with some modification) to
prevent circular dependencies when patching NumPy.
"""

import numpy as np

__all__ = ["fftshift", "ifftshift", "fftfreq", "rfftfreq"]


def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    For full documentation refer to `numpy.fft.fftshift`.

    """
    x = np.asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, (int, np.integer)):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]

    return np.roll(x, shift, axes)


def ifftshift(x, axes=None):
    """
    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    For full documentation refer to `numpy.fft.ifftshift`.

    """
    x = np.asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, (int, np.integer)):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    return np.roll(x, shift, axes)


def fftfreq(n, d=1.0, device=None):
    """
    Return the Discrete Fourier Transform sample frequencies.

    For full documentation refer to `numpy.fft.fftfreq`.

    """
    if not isinstance(n, (int, np.integer)):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    # pylint: disable=unexpected-keyword-arg
    results = np.empty(n, int, device=device)
    # pylint: enable=unexpected-keyword-arg
    N = (n - 1) // 2 + 1
    # pylint: disable=unexpected-keyword-arg
    p1 = np.arange(0, N, dtype=int, device=device)
    # pylint: enable=unexpected-keyword-arg
    results[:N] = p1
    # pylint: disable=unexpected-keyword-arg
    p2 = np.arange(-(n // 2), 0, dtype=int, device=device)
    # pylint: enable=unexpected-keyword-arg
    results[N:] = p2
    return results * val


def rfftfreq(n, d=1.0, device=None):
    """
    Return the Discrete Fourier Transform sample frequencies (for usage with
    `rfft`, `irfft`).

    For full documentation refer to `numpy.fft.rfftfreq`.

    """
    if not isinstance(n, (int, np.integer)):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    N = n // 2 + 1
    # pylint: disable=unexpected-keyword-arg
    results = np.arange(0, N, dtype=int, device=device)
    # pylint: enable=unexpected-keyword-arg
    return results * val
