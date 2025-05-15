# This file includes tests from numpy.fft module:
# https://github.com/numpy/numpy/blob/main/numpy/fft/tests/test_pocketfft.py

import queue
import threading

import numpy as np
import pytest
from numpy.random import random
from numpy.testing import (
    IS_WASM,
    assert_allclose,
    assert_array_equal,
    assert_raises,
)

import mkl_fft.interfaces.numpy_fft as mkl_fft
from mkl_fft.tests.helper import requires_numpy_2


def fft1(x):
    L = len(x)
    phase = -2j * np.pi * (np.arange(L) / L)
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(x * np.exp(phase), axis=1)


class TestFFTShift:

    def test_fft_n(self):
        assert_raises(ValueError, mkl_fft.fft, [1, 2, 3], 0)


class TestFFT1D:

    def test_identity(self):
        maxlen = 512
        x = random(maxlen) + 1j * random(maxlen)
        xr = random(maxlen)
        for i in range(1, maxlen):
            assert_allclose(
                mkl_fft.ifft(mkl_fft.fft(x[0:i])), x[0:i], atol=1e-12
            )
            assert_allclose(
                mkl_fft.irfft(mkl_fft.rfft(xr[0:i]), i), xr[0:i], atol=1e-12
            )

    @pytest.mark.parametrize(
        "dtype", [np.single, np.double]
    )  # , np.longdouble])
    def test_identity_long_short(self, dtype):
        # Test with explicitly given number of points, both for n
        # smaller and for n larger than the input size.
        maxlen = 16
        atol = 5 * np.spacing(np.array(1.0, dtype=dtype))
        x = random(maxlen).astype(dtype) + 1j * random(maxlen).astype(dtype)
        xx = np.concatenate([x, np.zeros_like(x)])
        xr = random(maxlen).astype(dtype)
        xxr = np.concatenate([xr, np.zeros_like(xr)])
        for i in range(1, maxlen * 2):
            check_c = mkl_fft.ifft(mkl_fft.fft(x, n=i), n=i)
            assert check_c.real.dtype == dtype
            assert_allclose(check_c, xx[0:i], atol=atol, rtol=0)
            check_r = mkl_fft.irfft(mkl_fft.rfft(xr, n=i), n=i)
            assert check_r.dtype == dtype
            assert_allclose(check_r, xxr[0:i], atol=atol, rtol=0)

    @pytest.mark.parametrize(
        "dtype", [np.single, np.double]
    )  # , np.longdouble])
    def test_identity_long_short_reversed(self, dtype):
        # Also test explicitly given number of points in reversed order.
        maxlen = 16
        atol = 5 * np.spacing(np.array(1.0, dtype=dtype))
        x = random(maxlen).astype(dtype) + 1j * random(maxlen).astype(dtype)
        xx = np.concatenate([x, np.zeros_like(x)])
        for i in range(1, maxlen * 2):
            check_via_c = mkl_fft.fft(mkl_fft.ifft(x, n=i), n=i)
            assert check_via_c.dtype == x.dtype
            assert_allclose(check_via_c, xx[0:i], atol=atol, rtol=1e-14)
            # For irfft, we can neither recover the imaginary part of
            # the first element, nor the imaginary part of the last
            # element if npts is even.  So, set to 0 for the comparison.
            y = x.copy()
            n = i // 2 + 1
            y.imag[0] = 0
            if i % 2 == 0:
                y.imag[n - 1 :] = 0
            yy = np.concatenate([y, np.zeros_like(y)])
            check_via_r = mkl_fft.rfft(mkl_fft.irfft(x, n=i), n=i)
            assert check_via_r.dtype == x.dtype
            assert_allclose(check_via_r, yy[0:n], atol=atol, rtol=0)

    def test_fft(self):
        x = random(30) + 1j * random(30)
        assert_allclose(fft1(x), mkl_fft.fft(x), atol=1e-6)
        assert_allclose(fft1(x), mkl_fft.fft(x, norm="backward"), atol=1e-6)
        assert_allclose(
            fft1(x) / np.sqrt(30), mkl_fft.fft(x, norm="ortho"), atol=1e-6
        )
        assert_allclose(
            fft1(x) / 30.0, mkl_fft.fft(x, norm="forward"), atol=1e-6
        )

    @requires_numpy_2
    @pytest.mark.parametrize("axis", (0, 1))
    @pytest.mark.parametrize("dtype", (complex, float))
    @pytest.mark.parametrize("transpose", (True, False))
    def test_fft_out_argument(self, dtype, transpose, axis):
        def zeros_like(x):
            if transpose:
                return np.zeros_like(x.T).T
            else:
                return np.zeros_like(x)

        # tests below only test the out parameter
        if dtype is complex:
            y = random((10, 20)) + 1j * random((10, 20))
            fft, ifft = mkl_fft.fft, mkl_fft.ifft
        else:
            y = random((10, 20))
            fft, ifft = mkl_fft.rfft, mkl_fft.irfft

        expected = fft(y, axis=axis)
        out = zeros_like(expected)
        result = fft(y, out=out, axis=axis)
        assert result is out
        assert_array_equal(result, expected)

        expected2 = ifft(expected, axis=axis)
        out2 = out if dtype is complex else zeros_like(expected2)
        result2 = ifft(out, out=out2, axis=axis)
        assert result2 is out2
        assert_array_equal(result2, expected2)

    @requires_numpy_2
    @pytest.mark.parametrize("axis", [0, 1])
    def test_fft_inplace_out(self, axis):
        # Test some weirder in-place combinations
        y = random((20, 20)) + 1j * random((20, 20))
        # Fully in-place.
        y1 = y.copy()
        expected1 = mkl_fft.fft(y1, axis=axis)
        result1 = mkl_fft.fft(y1, axis=axis, out=y1)
        assert result1 is y1
        assert_array_equal(result1, expected1)
        # In-place of part of the array; rest should be unchanged.
        y2 = y.copy()
        out2 = y2[:10] if axis == 0 else y2[:, :10]
        expected2 = mkl_fft.fft(y2, n=10, axis=axis)
        result2 = mkl_fft.fft(y2, n=10, axis=axis, out=out2)
        assert result2 is out2
        assert_array_equal(result2, expected2)
        if axis == 0:
            assert_array_equal(y2[10:], y[10:])
        else:
            assert_array_equal(y2[:, 10:], y[:, 10:])
        # In-place of another part of the array.
        y3 = y.copy()
        y3_sel = y3[5:] if axis == 0 else y3[:, 5:]
        out3 = y3[5:15] if axis == 0 else y3[:, 5:15]
        expected3 = mkl_fft.fft(y3_sel, n=10, axis=axis)
        result3 = mkl_fft.fft(y3_sel, n=10, axis=axis, out=out3)
        assert result3 is out3
        assert_array_equal(result3, expected3)
        if axis == 0:
            assert_array_equal(y3[:5], y[:5])
            assert_array_equal(y3[15:], y[15:])
        else:
            assert_array_equal(y3[:, :5], y[:, :5])
            assert_array_equal(y3[:, 15:], y[:, 15:])
        # In-place with n > n_in; rest should be unchanged.
        y4 = y.copy()
        y4_sel = y4[:10] if axis == 0 else y4[:, :10]
        out4 = y4[:15] if axis == 0 else y4[:, :15]
        expected4 = mkl_fft.fft(y4_sel, n=15, axis=axis)
        result4 = mkl_fft.fft(y4_sel, n=15, axis=axis, out=out4)
        assert result4 is out4
        assert_array_equal(result4, expected4)
        if axis == 0:
            assert_array_equal(y4[15:], y[15:])
        else:
            assert_array_equal(y4[:, 15:], y[:, 15:])
        # Overwrite in a transpose.
        y5 = y.copy()
        out5 = y5.T
        result5 = mkl_fft.fft(y5, axis=axis, out=out5)
        assert result5 is out5
        assert_array_equal(result5, expected1)
        # Reverse strides.
        y6 = y.copy()
        out6 = y6[::-1] if axis == 0 else y6[:, ::-1]
        result6 = mkl_fft.fft(y6, axis=axis, out=out6)
        assert result6 is out6
        assert_array_equal(result6, expected1)

    @requires_numpy_2
    def test_fft_bad_out(self):
        x = np.arange(30.0)
        with pytest.raises(TypeError, match="must be of ArrayType"):
            mkl_fft.fft(x, out="")
        with pytest.raises(ValueError, match="has wrong shape"):
            mkl_fft.fft(x, out=np.zeros_like(x).reshape(5, -1))
        with pytest.raises(TypeError, match="Cannot cast"):
            mkl_fft.fft(x, out=np.zeros_like(x, dtype=float))

    @pytest.mark.parametrize("norm", (None, "backward", "ortho", "forward"))
    def test_ifft(self, norm):
        x = random(30) + 1j * random(30)
        assert_allclose(
            x, mkl_fft.ifft(mkl_fft.fft(x, norm=norm), norm=norm), atol=1e-6
        )
        # Ensure we get the correct error message
        with pytest.raises(
            ValueError, match="Invalid number of FFT data points"
        ):
            mkl_fft.ifft([], norm=norm)

    def test_fft2(self):
        x = random((30, 20)) + 1j * random((30, 20))
        assert_allclose(
            mkl_fft.fft(mkl_fft.fft(x, axis=1), axis=0),
            mkl_fft.fft2(x),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.fft2(x), mkl_fft.fft2(x, norm="backward"), atol=1e-6
        )
        assert_allclose(
            mkl_fft.fft2(x) / np.sqrt(30 * 20),
            mkl_fft.fft2(x, norm="ortho"),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.fft2(x) / (30.0 * 20.0),
            mkl_fft.fft2(x, norm="forward"),
            atol=1e-6,
        )

    def test_ifft2(self):
        x = random((30, 20)) + 1j * random((30, 20))
        assert_allclose(
            mkl_fft.ifft(mkl_fft.ifft(x, axis=1), axis=0),
            mkl_fft.ifft2(x),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.ifft2(x), mkl_fft.ifft2(x, norm="backward"), atol=1e-6
        )
        assert_allclose(
            mkl_fft.ifft2(x) * np.sqrt(30 * 20),
            mkl_fft.ifft2(x, norm="ortho"),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.ifft2(x) * (30.0 * 20.0),
            mkl_fft.ifft2(x, norm="forward"),
            atol=1e-6,
        )

    def test_fftn(self):
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        assert_allclose(
            mkl_fft.fft(mkl_fft.fft(mkl_fft.fft(x, axis=2), axis=1), axis=0),
            mkl_fft.fftn(x),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.fftn(x), mkl_fft.fftn(x, norm="backward"), atol=1e-6
        )
        assert_allclose(
            mkl_fft.fftn(x) / np.sqrt(30 * 20 * 10),
            mkl_fft.fftn(x, norm="ortho"),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.fftn(x) / (30.0 * 20.0 * 10.0),
            mkl_fft.fftn(x, norm="forward"),
            atol=1e-6,
        )

    def test_ifftn(self):
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        assert_allclose(
            mkl_fft.ifft(mkl_fft.ifft(mkl_fft.ifft(x, axis=2), axis=1), axis=0),
            mkl_fft.ifftn(x),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.ifftn(x), mkl_fft.ifftn(x, norm="backward"), atol=1e-6
        )
        assert_allclose(
            mkl_fft.ifftn(x) * np.sqrt(30 * 20 * 10),
            mkl_fft.ifftn(x, norm="ortho"),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.ifftn(x) * (30.0 * 20.0 * 10.0),
            mkl_fft.ifftn(x, norm="forward"),
            atol=1e-6,
        )

    def test_rfft(self):
        x = random(30)
        for n in [x.size, 2 * x.size]:
            for norm in [None, "backward", "ortho", "forward"]:
                assert_allclose(
                    mkl_fft.fft(x, n=n, norm=norm)[: (n // 2 + 1)],
                    mkl_fft.rfft(x, n=n, norm=norm),
                    atol=1e-6,
                )
            assert_allclose(
                mkl_fft.rfft(x, n=n),
                mkl_fft.rfft(x, n=n, norm="backward"),
                atol=1e-6,
            )
            assert_allclose(
                mkl_fft.rfft(x, n=n) / np.sqrt(n),
                mkl_fft.rfft(x, n=n, norm="ortho"),
                atol=1e-6,
            )
            assert_allclose(
                mkl_fft.rfft(x, n=n) / n,
                mkl_fft.rfft(x, n=n, norm="forward"),
                atol=1e-6,
            )

    def test_rfft_even(self):
        x = np.arange(8)
        n = 4
        y = mkl_fft.rfft(x, n)
        assert_allclose(y, mkl_fft.fft(x[:n])[: n // 2 + 1], rtol=1e-14)

    def test_rfft_odd(self):
        x = np.array([1, 0, 2, 3, -3])
        y = mkl_fft.rfft(x)
        assert_allclose(y, mkl_fft.fft(x)[:3], rtol=1e-14)

    def test_irfft(self):
        x = random(30)
        assert_allclose(x, mkl_fft.irfft(mkl_fft.rfft(x)), atol=1e-6)
        assert_allclose(
            x,
            mkl_fft.irfft(mkl_fft.rfft(x, norm="backward"), norm="backward"),
            atol=1e-6,
        )
        assert_allclose(
            x,
            mkl_fft.irfft(mkl_fft.rfft(x, norm="ortho"), norm="ortho"),
            atol=1e-6,
        )
        assert_allclose(
            x,
            mkl_fft.irfft(mkl_fft.rfft(x, norm="forward"), norm="forward"),
            atol=1e-6,
        )

    def test_rfft2(self):
        x = random((30, 20))
        assert_allclose(mkl_fft.fft2(x)[:, :11], mkl_fft.rfft2(x), atol=1e-6)
        assert_allclose(
            mkl_fft.rfft2(x), mkl_fft.rfft2(x, norm="backward"), atol=1e-6
        )
        assert_allclose(
            mkl_fft.rfft2(x) / np.sqrt(30 * 20),
            mkl_fft.rfft2(x, norm="ortho"),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.rfft2(x) / (30.0 * 20.0),
            mkl_fft.rfft2(x, norm="forward"),
            atol=1e-6,
        )

    def test_irfft2(self):
        x = random((30, 20))
        assert_allclose(x, mkl_fft.irfft2(mkl_fft.rfft2(x)), atol=1e-6)
        assert_allclose(
            x,
            mkl_fft.irfft2(mkl_fft.rfft2(x, norm="backward"), norm="backward"),
            atol=1e-6,
        )
        assert_allclose(
            x,
            mkl_fft.irfft2(mkl_fft.rfft2(x, norm="ortho"), norm="ortho"),
            atol=1e-6,
        )
        assert_allclose(
            x,
            mkl_fft.irfft2(mkl_fft.rfft2(x, norm="forward"), norm="forward"),
            atol=1e-6,
        )

    def test_rfftn(self):
        x = random((30, 20, 10))
        assert_allclose(mkl_fft.fftn(x)[:, :, :6], mkl_fft.rfftn(x), atol=1e-6)
        assert_allclose(
            mkl_fft.rfftn(x), mkl_fft.rfftn(x, norm="backward"), atol=1e-6
        )
        assert_allclose(
            mkl_fft.rfftn(x) / np.sqrt(30 * 20 * 10),
            mkl_fft.rfftn(x, norm="ortho"),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.rfftn(x) / (30.0 * 20.0 * 10.0),
            mkl_fft.rfftn(x, norm="forward"),
            atol=1e-6,
        )
        # Regression test for gh-27159
        x = np.ones((2, 3))
        result = mkl_fft.rfftn(x, axes=(0, 0, 1), s=(10, 20, 40))
        assert result.shape == (10, 21)
        expected = mkl_fft.fft(
            mkl_fft.fft(mkl_fft.rfft(x, axis=1, n=40), axis=0, n=20),
            axis=0,
            n=10,
        )
        assert expected.shape == (10, 21)
        assert_allclose(result, expected, atol=1e-6)

    def test_irfftn(self):
        x = random((30, 20, 10))
        assert_allclose(x, mkl_fft.irfftn(mkl_fft.rfftn(x)), atol=1e-6)
        assert_allclose(
            x,
            mkl_fft.irfftn(mkl_fft.rfftn(x, norm="backward"), norm="backward"),
            atol=1e-6,
        )
        assert_allclose(
            x,
            mkl_fft.irfftn(mkl_fft.rfftn(x, norm="ortho"), norm="ortho"),
            atol=1e-6,
        )
        assert_allclose(
            x,
            mkl_fft.irfftn(mkl_fft.rfftn(x, norm="forward"), norm="forward"),
            atol=1e-6,
        )

    def test_hfft(self):
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        assert_allclose(mkl_fft.fft(x), mkl_fft.hfft(x_herm), atol=1e-6)
        assert_allclose(
            mkl_fft.hfft(x_herm),
            mkl_fft.hfft(x_herm, norm="backward"),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.hfft(x_herm) / np.sqrt(30),
            mkl_fft.hfft(x_herm, norm="ortho"),
            atol=1e-6,
        )
        assert_allclose(
            mkl_fft.hfft(x_herm) / 30.0,
            mkl_fft.hfft(x_herm, norm="forward"),
            atol=1e-6,
        )

    def test_ihfft(self):
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        assert_allclose(x_herm, mkl_fft.ihfft(mkl_fft.hfft(x_herm)), atol=1e-6)
        assert_allclose(
            x_herm,
            mkl_fft.ihfft(
                mkl_fft.hfft(x_herm, norm="backward"), norm="backward"
            ),
            atol=1e-6,
        )
        assert_allclose(
            x_herm,
            mkl_fft.ihfft(mkl_fft.hfft(x_herm, norm="ortho"), norm="ortho"),
            atol=1e-6,
        )
        assert_allclose(
            x_herm,
            mkl_fft.ihfft(mkl_fft.hfft(x_herm, norm="forward"), norm="forward"),
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "op", [mkl_fft.fftn, mkl_fft.ifftn, mkl_fft.rfftn, mkl_fft.irfftn]
    )
    def test_axes(self, op):
        x = random((30, 20, 10))
        axes = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]
        for a in axes:
            op_tr = op(np.transpose(x, a))
            tr_op = np.transpose(op(x, axes=a), a)
            assert_allclose(op_tr, tr_op, atol=1e-6)

    @pytest.mark.parametrize(
        "op", [mkl_fft.fftn, mkl_fft.ifftn, mkl_fft.fft2, mkl_fft.ifft2]
    )
    def test_s_negative_1(self, op):
        x = np.arange(100).reshape(10, 10)
        # should use the whole input array along the first axis
        assert op(x, s=(-1, 5), axes=(0, 1)).shape == (10, 5)

    @requires_numpy_2
    @pytest.mark.parametrize(
        "op", [mkl_fft.fftn, mkl_fft.ifftn, mkl_fft.rfftn, mkl_fft.irfftn]
    )
    def test_s_axes_none(self, op):
        x = np.arange(100).reshape(10, 10)
        with pytest.warns(match="`axes` should not be `None` if `s`"):
            op(x, s=(-1, 5))

    @requires_numpy_2
    @pytest.mark.parametrize("op", [mkl_fft.fft2, mkl_fft.ifft2])
    def test_s_axes_none_2D(self, op):
        x = np.arange(100).reshape(10, 10)
        with pytest.warns(match="`axes` should not be `None` if `s`"):
            op(x, s=(-1, 5), axes=None)

    @requires_numpy_2
    @pytest.mark.parametrize(
        "op",
        [
            mkl_fft.fftn,
            mkl_fft.ifftn,
            mkl_fft.rfftn,
            mkl_fft.irfftn,
            mkl_fft.fft2,
            mkl_fft.ifft2,
        ],
    )
    def test_s_contains_none(self, op):
        x = random((30, 20, 10))
        with pytest.warns(match="array containing `None` values to `s`"):
            op(x, s=(10, None, 10), axes=(0, 1, 2))

    def test_all_1d_norm_preserving(self):
        # verify that round-trip transforms are norm-preserving
        x = random(30)
        x_norm = np.linalg.norm(x)
        n = x.size * 2
        func_pairs = [
            (mkl_fft.fft, mkl_fft.ifft),
            (mkl_fft.rfft, mkl_fft.irfft),
            # hfft: order so the first function takes x.size samples
            #       (necessary for comparison to x_norm above)
            (mkl_fft.ihfft, mkl_fft.hfft),
        ]
        for forw, back in func_pairs:
            for n in [x.size, 2 * x.size]:
                for norm in [None, "backward", "ortho", "forward"]:
                    tmp = forw(x, n=n, norm=norm)
                    tmp = back(tmp, n=n, norm=norm)
                    assert_allclose(x_norm, np.linalg.norm(tmp), atol=1e-6)

    @requires_numpy_2
    @pytest.mark.parametrize("axes", [(0, 1), (0, 2), None])
    @pytest.mark.parametrize("dtype", (complex, float))
    @pytest.mark.parametrize("transpose", (True, False))
    def test_fftn_out_argument(self, dtype, transpose, axes):
        def zeros_like(x):
            if transpose:
                return np.zeros_like(x.T).T
            else:
                return np.zeros_like(x)

        # tests below only test the out parameter
        if dtype is complex:
            x = random((10, 5, 6)) + 1j * random((10, 5, 6))
            fft, ifft = mkl_fft.fftn, mkl_fft.ifftn
        else:
            x = random((10, 5, 6))
            fft, ifft = mkl_fft.rfftn, mkl_fft.irfftn

        expected = fft(x, axes=axes)
        out = zeros_like(expected)
        result = fft(x, out=out, axes=axes)
        assert result is out
        assert_array_equal(result, expected)

        expected2 = ifft(expected, axes=axes)
        out2 = out if dtype is complex else zeros_like(expected2)
        result2 = ifft(out, out=out2, axes=axes)
        assert result2 is out2
        assert_array_equal(result2, expected2)

    @requires_numpy_2
    @pytest.mark.parametrize(
        "fft", [mkl_fft.fftn, mkl_fft.ifftn, mkl_fft.rfftn]
    )
    def test_fftn_out_and_s_interaction(self, fft):
        # With s, shape varies, so generally one cannot pass in out.
        if fft is mkl_fft.rfftn:
            x = random((10, 5, 6))
        else:
            x = random((10, 5, 6)) + 1j * random((10, 5, 6))
        with pytest.raises(ValueError, match="has wrong shape"):
            fft(x, out=np.zeros_like(x), s=(3, 3, 3), axes=(0, 1, 2))
        # Except on the first axis done (which is the last of axes).
        s = (10, 5, 5)
        expected = fft(x, s=s, axes=(0, 1, 2))
        out = np.zeros_like(expected)
        result = fft(x, s=s, axes=(0, 1, 2), out=out)
        assert result is out
        assert_array_equal(result, expected)

    @requires_numpy_2
    @pytest.mark.parametrize("s", [(9, 5, 5), (3, 3, 3)])
    def test_irfftn_out_and_s_interaction(self, s):
        # Since for irfftn, the output is real and thus cannot be used for
        # intermediate steps, it should always work.
        x = random((9, 5, 6, 2)) + 1j * random((9, 5, 6, 2))
        expected = mkl_fft.irfftn(x, s=s, axes=(0, 1, 2))
        out = np.zeros_like(expected)
        result = mkl_fft.irfftn(x, s=s, axes=(0, 1, 2), out=out)
        assert result is out
        assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize("order", ["F", "non-contiguous"])
@pytest.mark.parametrize(
    "fft",
    [
        mkl_fft.fft,
        mkl_fft.fft2,
        mkl_fft.fftn,
        mkl_fft.ifft,
        mkl_fft.ifft2,
        mkl_fft.ifftn,
    ],
)
def test_fft_with_order(dtype, order, fft):
    # Check that FFT/IFFT produces identical results for C, Fortran and
    # non contiguous arrays
    np.random.seed(42)
    X = np.random.rand(8, 7, 13).astype(dtype, copy=False)
    # See discussion in pull/14178
    _tol = 8.0 * np.sqrt(np.log2(X.size)) * np.finfo(X.dtype).eps
    if order == "F":
        Y = np.asfortranarray(X)
    else:
        # Make a non contiguous array
        Y = X[::-1]
        X = np.ascontiguousarray(X[::-1])

    if fft.__name__.endswith("fft"):
        for axis in range(3):
            X_res = fft(X, axis=axis)
            Y_res = fft(Y, axis=axis)
            assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
    elif fft.__name__.endswith(("fft2", "fftn")):
        axes = [(0, 1), (1, 2), (0, 2)]
        if fft.__name__.endswith("fftn"):
            axes.extend([(0,), (1,), (2,), None])
        for ax in axes:
            X_res = fft(X, axes=ax)
            Y_res = fft(Y, axes=ax)
            assert_allclose(X_res, Y_res, atol=_tol, rtol=10 * _tol)
    else:
        raise ValueError


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("n", [None, 7, 12])
def test_fft_output_order(order, n):
    np.random.seed(42)
    x = np.random.rand(10)
    x = np.asarray(x, dtype=np.complex64, order=order)
    res = mkl_fft.fft(x, n=n)
    assert res.flags.c_contiguous == x.flags.c_contiguous
    assert res.flags.f_contiguous == x.flags.f_contiguous


@pytest.mark.skipif(IS_WASM, reason="Cannot start thread")
class TestFFTThreadSafe:
    threads = 16
    input_shape = (800, 200)

    def _test_mtsame(self, func, *args):
        def worker(args, q):
            q.put(func(*args))

        q = queue.Queue()
        expected = func(*args)

        # Spin off a bunch of threads to call the same function simultaneously
        t = [
            threading.Thread(target=worker, args=(args, q))
            for i in range(self.threads)
        ]
        [x.start() for x in t]

        [x.join() for x in t]
        # Make sure all threads returned the correct value
        for _ in range(self.threads):
            assert_array_equal(
                q.get(timeout=5),
                expected,
                "Function returned wrong value in multithreaded context",
            )

    def test_fft(self):
        a = np.ones(self.input_shape) * 1 + 0j
        self._test_mtsame(mkl_fft.fft, a)

    def test_ifft(self):
        a = np.ones(self.input_shape) * 1 + 0j
        self._test_mtsame(mkl_fft.ifft, a)

    def test_rfft(self):
        a = np.ones(self.input_shape)
        self._test_mtsame(mkl_fft.rfft, a)

    def test_irfft(self):
        a = np.ones(self.input_shape) * 1 + 0j
        self._test_mtsame(mkl_fft.irfft, a)


def test_irfft_with_n_1_regression():
    # Regression test for gh-25661
    x = np.arange(10)
    mkl_fft.irfft(x, n=1)
    mkl_fft.hfft(x, n=1)
    mkl_fft.irfft(np.array([0], complex), n=10)


def test_irfft_with_n_large_regression():
    # Regression test for gh-25679
    x = np.arange(5) * (1 + 1j)
    result = mkl_fft.hfft(x, n=10)
    expected = np.array(
        [
            20.0,
            9.91628173,
            -11.8819096,
            7.1048486,
            -6.62459848,
            4.0,
            -3.37540152,
            -0.16057669,
            1.8819096,
            -20.86055364,
        ]
    )
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "fft", [mkl_fft.fft, mkl_fft.ifft, mkl_fft.rfft, mkl_fft.irfft]
)
@pytest.mark.parametrize(
    "data",
    [
        np.array([False, True, False]),
        np.arange(10, dtype=np.uint8),
        np.arange(5, dtype=np.int16),
    ],
)
def test_fft_with_integer_or_bool_input(data, fft):
    # Regression test for gh-25819
    result = fft(data)
    float_data = data.astype(np.result_type(data, 1.0))
    expected = fft(float_data)
    assert_allclose(result, expected, rtol=1e-15)
