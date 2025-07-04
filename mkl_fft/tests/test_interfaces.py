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

import numpy as np
import pytest
from numpy.testing import assert_raises

import mkl_fft.interfaces as mfi

try:
    scipy_fft = mfi.scipy_fft
except AttributeError:
    scipy_fft = None

interfaces = []
ids = []
if scipy_fft is not None:
    interfaces.append(scipy_fft)
    ids.append("scipy")
interfaces.append(mfi.numpy_fft)
ids.append("numpy")


@pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_scipy_fft(norm, dtype):
    pytest.importorskip("scipy", reason="requires scipy")
    x = np.ones(511, dtype=dtype)
    w = mfi.scipy_fft.fft(x, norm=norm, workers=None, plan=None)
    xx = mfi.scipy_fft.ifft(w, norm=norm, workers=None, plan=None)
    tol = 128 * np.finfo(np.dtype(dtype)).eps
    np.testing.assert_allclose(x, xx, atol=tol, rtol=tol)


@pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_numpy_fft(norm, dtype):
    x = np.ones(511, dtype=dtype)
    w = mfi.numpy_fft.fft(x, norm=norm)
    xx = mfi.numpy_fft.ifft(w, norm=norm)
    tol = 128 * np.finfo(np.dtype(dtype)).eps
    np.testing.assert_allclose(x, xx, atol=tol, rtol=tol)


@pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scipy_rfft(norm, dtype):
    pytest.importorskip("scipy", reason="requires scipy")
    x = np.ones(511, dtype=dtype)
    w = mfi.scipy_fft.rfft(x, norm=norm, workers=None, plan=None)
    xx = mfi.scipy_fft.irfft(
        w, n=x.shape[0], norm=norm, workers=None, plan=None
    )
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)

    x = np.ones(510, dtype=dtype)
    w = mfi.scipy_fft.rfft(x, norm=norm, workers=None, plan=None)
    xx = mfi.scipy_fft.irfft(w, norm=norm, workers=None, plan=None)
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)


@pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_rfft(norm, dtype):
    x = np.ones(511, dtype=dtype)
    w = mfi.numpy_fft.rfft(x, norm=norm)
    xx = mfi.numpy_fft.irfft(w, n=x.shape[0], norm=norm)
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)


@pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_scipy_fftn(norm, dtype):
    pytest.importorskip("scipy", reason="requires scipy")
    x = np.ones((37, 83), dtype=dtype)
    w = mfi.scipy_fft.fftn(x, norm=norm, workers=None, plan=None)
    xx = mfi.scipy_fft.ifftn(w, norm=norm, workers=None, plan=None)
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)


@pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_numpy_fftn(norm, dtype):
    x = np.ones((37, 83), dtype=dtype)
    w = mfi.numpy_fft.fftn(x, norm=norm)
    xx = mfi.numpy_fft.ifftn(w, norm=norm)
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)


@pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scipy_rfftn(norm, dtype):
    pytest.importorskip("scipy", reason="requires scipy")
    x = np.ones((37, 83), dtype=dtype)
    w = mfi.scipy_fft.rfftn(x, norm=norm, workers=None, plan=None)
    xx = mfi.scipy_fft.irfftn(w, s=x.shape, norm=norm, workers=None, plan=None)
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)

    x = np.ones((36, 82), dtype=dtype)
    w = mfi.scipy_fft.rfftn(x, norm=norm, workers=None, plan=None)
    xx = mfi.scipy_fft.irfftn(w, norm=norm, workers=None, plan=None)
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_rfftn(norm, dtype):
    x = np.ones((37, 83), dtype=dtype)
    w = mfi.numpy_fft.rfftn(x, norm=norm)
    xx = mfi.numpy_fft.irfftn(w, s=x.shape, norm=norm)
    tol = 64 * np.finfo(np.dtype(dtype)).eps
    assert np.allclose(x, xx, atol=tol, rtol=tol)


def _get_blacklisted_dtypes():
    bl_list = []
    for dt in ["float16", "float128", "complex256"]:
        if hasattr(np, dt):
            bl_list.append(getattr(np, dt))
    return bl_list


@pytest.mark.parametrize("dtype", _get_blacklisted_dtypes())
def test_scipy_no_support_for(dtype):
    pytest.importorskip("scipy", reason="requires scipy")
    x = np.ones(16, dtype=dtype)
    assert_raises(NotImplementedError, mfi.scipy_fft.ifft, x)


def test_scipy_fft_arg_validate():
    pytest.importorskip("scipy", reason="requires scipy")
    with pytest.raises(ValueError):
        mfi.scipy_fft.fft([1, 2, 3, 4], norm=b"invalid")

    with pytest.raises(NotImplementedError):
        mfi.scipy_fft.fft([1, 2, 3, 4], plan="magic")


@pytest.mark.parametrize("interface", interfaces, ids=ids)
def test_axes(interface):
    x = np.arange(24.0).reshape(2, 3, 4)
    res = interface.rfft2(x, axes=(1, 2))
    exp = np.fft.rfft2(x, axes=(1, 2))
    tol = 64 * np.finfo(np.float64).eps
    assert np.allclose(res, exp, atol=tol, rtol=tol)


@pytest.mark.parametrize("interface", interfaces, ids=ids)
@pytest.mark.parametrize(
    "func", ["fftshift", "ifftshift", "fftfreq", "rfftfreq"]
)
def test_interface_helper_functions(interface, func):
    assert hasattr(interface, func)
