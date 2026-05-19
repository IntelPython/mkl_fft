"""Benchmarks for 1-D FFT operations using the mkl_fft root API."""

import mkl_fft

from ._utils import _DTYPES_ALL, _DTYPES_REAL, BenchC2C, BenchR2C

_SIZES_POW2 = [64, 256, 1024, 4096, 16384, 65536]
_SIZES_NONPOW2 = [127, 509, 1000, 4001, 10007]


# ---------------------------------------------------------------------------
# Complex-to-complex 1-D (power-of-two sizes)
# ---------------------------------------------------------------------------


class BenchFFT1D(BenchC2C):
    """Forward and inverse complex FFT — power-of-two sizes."""

    params = [_SIZES_POW2, _DTYPES_ALL]
    param_names = ["n", "dtype"]

    def time_fft(self, n, dtype):
        mkl_fft.fft(self.x)

    def time_ifft(self, n, dtype):
        mkl_fft.ifft(self.x)


# ---------------------------------------------------------------------------
# Real-to-complex / complex-to-real 1-D (power-of-two sizes)
# ---------------------------------------------------------------------------


class BenchRFFT1D(BenchR2C):
    """Forward rfft and inverse irfft — power-of-two sizes."""

    params = [_SIZES_POW2, _DTYPES_REAL]
    param_names = ["n", "dtype"]

    def time_rfft(self, n, dtype):
        mkl_fft.rfft(self.x_real)

    def time_irfft(self, n, dtype):
        mkl_fft.irfft(self.x_complex, n=n)


# ---------------------------------------------------------------------------
# Complex-to-complex 1-D (non-power-of-two sizes)
# ---------------------------------------------------------------------------


class BenchFFT1DNonPow2(BenchC2C):
    """Forward and inverse complex FFT — non-power-of-two sizes.

    MKL uses a different code path for non-power-of-two transforms;
    this suite catches regressions in that path.
    """

    params = [_SIZES_NONPOW2, ["float64", "complex128", "complex64"]]
    param_names = ["n", "dtype"]

    def time_fft(self, n, dtype):
        mkl_fft.fft(self.x)

    def time_ifft(self, n, dtype):
        mkl_fft.ifft(self.x)


# ---------------------------------------------------------------------------
# Real-to-complex / complex-to-real 1-D (non-power-of-two sizes)
# ---------------------------------------------------------------------------


class BenchRFFT1DNonPow2(BenchR2C):
    """Forward rfft and inverse irfft — non-power-of-two sizes."""

    params = [_SIZES_NONPOW2, _DTYPES_REAL]
    param_names = ["n", "dtype"]

    def time_rfft(self, n, dtype):
        mkl_fft.rfft(self.x_real)

    def time_irfft(self, n, dtype):
        mkl_fft.irfft(self.x_complex, n=n)
