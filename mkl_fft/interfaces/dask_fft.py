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

from dask.array.fft import fft_wrap, fftfreq, fftshift, ifftshift, rfftfreq

from . import numpy_fft as _numpy_fft

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
    "fftshift",
    "ifftshift",
    "fftfreq",
    "rfftfreq",
    "fft_wrap",
]


fft = fft_wrap(_numpy_fft.fft)
ifft = fft_wrap(_numpy_fft.ifft)
fft2 = fft_wrap(_numpy_fft.fft2)
ifft2 = fft_wrap(_numpy_fft.ifft2)
fftn = fft_wrap(_numpy_fft.fftn)
ifftn = fft_wrap(_numpy_fft.ifftn)
rfft = fft_wrap(_numpy_fft.rfft)
irfft = fft_wrap(_numpy_fft.irfft)
rfft2 = fft_wrap(_numpy_fft.rfft2)
irfft2 = fft_wrap(_numpy_fft.irfft2)
rfftn = fft_wrap(_numpy_fft.rfftn)
irfftn = fft_wrap(_numpy_fft.irfftn)
hfft = fft_wrap(_numpy_fft.hfft)
ihfft = fft_wrap(_numpy_fft.ihfft)
