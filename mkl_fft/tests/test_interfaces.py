#!/usr/bin/env python
# Copyright (c) 2017-2023, Intel Corporation
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

import mkl_fft.interfaces as mfi
import pytest
import numpy as np


def test_interfaces_has_numpy():
    assert hasattr(mfi, 'numpy_fft')


def test_interfaces_has_scipy():
    assert hasattr(mfi, 'scipy_fft')


@pytest.mark.parametrize('norm', [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
def test_scipy_fft(norm, dtype):
    x = np.ones(511, dtype=dtype)
    w = mfi.scipy_fft.fft(x, norm=norm)
    xx = mfi.scipy_fft.ifft(w, norm=norm)
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)


@pytest.mark.parametrize('norm', [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_scipy_rfft(norm, dtype):
    x = np.ones(511, dtype=dtype)
    w = mfi.scipy_fft.rfft(x, norm=norm)
    xx = mfi.scipy_fft.irfft(w, n=x.shape[0], norm=norm)
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)


@pytest.mark.parametrize('norm', [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
def test_scipy_fftn(norm, dtype):
    x = np.ones((37, 83), dtype=dtype)
    w = mfi.scipy_fft.fftn(x, norm=norm)
    xx = mfi.scipy_fft.ifftn(w, norm=norm)
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)


@pytest.mark.parametrize('norm', [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_scipy_rftn(norm, dtype):
    x = np.ones((37, 83), dtype=dtype)
    w = mfi.scipy_fft.rfftn(x, norm=norm)
    xx = mfi.scipy_fft.ifftn(w, s=x.shape, norm=norm)
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)
