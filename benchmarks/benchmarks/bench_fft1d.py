"""Benchmarks for 1-D FFT operations using the mkl_fft root API."""

import numpy as np

import mkl_fft

from ._utils import _make_input

_RNG_SEED = 42


# ---------------------------------------------------------------------------
# Complex-to-complex 1-D (power-of-two sizes)
# ---------------------------------------------------------------------------


class TimeFFT1D:
    """Forward and inverse complex FFT — power-of-two sizes."""

    params = [
        [64, 256, 1024, 4096, 16384, 65536],
        ["float32", "float64", "complex64", "complex128"],
    ]
    param_names = ["n", "dtype"]

    def setup(self, n, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        self.x = _make_input(rng, n, dtype)

    def time_fft(self, n, dtype):
        mkl_fft.fft(self.x)

    def time_ifft(self, n, dtype):
        mkl_fft.ifft(self.x)


# ---------------------------------------------------------------------------
# Real-to-complex / complex-to-real 1-D (power-of-two sizes)
# ---------------------------------------------------------------------------


class TimeRFFT1D:
    """Forward rfft and inverse irfft — power-of-two sizes."""

    params = [
        [64, 256, 1024, 4096, 16384, 65536],
        ["float32", "float64"],
    ]
    param_names = ["n", "dtype"]

    def setup(self, n, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        self.x_real = rng.standard_normal(n).astype(dtype)
        # irfft input: complex half-spectrum of length n//2+1
        self.x_complex = (
            rng.standard_normal(n // 2 + 1)
            + 1j * rng.standard_normal(n // 2 + 1)
        ).astype(cdtype)

    def time_rfft(self, n, dtype):
        mkl_fft.rfft(self.x_real)

    def time_irfft(self, n, dtype):
        mkl_fft.irfft(self.x_complex, n=n)


# ---------------------------------------------------------------------------
# Complex-to-complex 1-D (non-power-of-two sizes)
# ---------------------------------------------------------------------------


class TimeFFT1DNonPow2:
    """Forward and inverse complex FFT — non-power-of-two sizes.

    MKL uses a different code path for non-power-of-two transforms;
    this suite catches regressions in that path.
    """

    params = [
        [127, 509, 1000, 4001, 10007],
        ["float64", "complex128", "complex64"],
    ]
    param_names = ["n", "dtype"]

    def setup(self, n, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        self.x = _make_input(rng, n, dtype)

    def time_fft(self, n, dtype):
        mkl_fft.fft(self.x)

    def time_ifft(self, n, dtype):
        mkl_fft.ifft(self.x)


# ---------------------------------------------------------------------------
# Real-to-complex / complex-to-real 1-D (non-power-of-two sizes)
# ---------------------------------------------------------------------------


class TimeRFFT1DNonPow2:
    """Forward rfft and inverse irfft — non-power-of-two sizes."""

    params = [
        [127, 509, 1000, 4001, 10007],
        ["float32", "float64"],
    ]
    param_names = ["n", "dtype"]

    def setup(self, n, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        self.x_real = rng.standard_normal(n).astype(dtype)
        self.x_complex = (
            rng.standard_normal(n // 2 + 1)
            + 1j * rng.standard_normal(n // 2 + 1)
        ).astype(cdtype)

    def time_rfft(self, n, dtype):
        mkl_fft.rfft(self.x_real)

    def time_irfft(self, n, dtype):
        mkl_fft.irfft(self.x_complex, n=n)
