"""Peak-memory benchmarks for FFT operations.

Measures peak RSS (resident set size) to detect memory regressions
in the mkl_fft root API across 1-D, 2-D, and 3-D transforms.
"""

import mkl_fft

from ._utils import (
    _DTYPES_REAL,
    _DTYPES_REDUCED,
    _SHAPES_2D,
    _SHAPES_3D,
    BenchC2C,
    BenchR2C,
)

_SIZES_1D = [1024, 16384, 65536, 262144]


# ---------------------------------------------------------------------------
# 1-D complex FFT
# ---------------------------------------------------------------------------


class PeakMemFFT1D(BenchC2C):
    """Peak RSS for 1-D complex FFT."""

    params = [_SIZES_1D, _DTYPES_REDUCED]
    param_names = ["n", "dtype"]

    def peakmem_fft(self, n, dtype):
        mkl_fft.fft(self.x)

    def peakmem_ifft(self, n, dtype):
        mkl_fft.ifft(self.x)


# ---------------------------------------------------------------------------
# 1-D real FFT
# ---------------------------------------------------------------------------


class PeakMemRFFT1D(BenchR2C):
    """Peak RSS for 1-D real FFT (forward and inverse)."""

    params = [_SIZES_1D, _DTYPES_REAL]
    param_names = ["n", "dtype"]

    def peakmem_rfft(self, n, dtype):
        mkl_fft.rfft(self.x_real)

    def peakmem_irfft(self, n, dtype):
        mkl_fft.irfft(self.x_complex, n=n)


# ---------------------------------------------------------------------------
# 2-D complex FFT
# ---------------------------------------------------------------------------


class PeakMemFFT2D(BenchC2C):
    """Peak RSS for 2-D complex FFT."""

    params = [_SHAPES_2D, _DTYPES_REDUCED]
    param_names = ["shape", "dtype"]

    def peakmem_fft2(self, shape, dtype):
        mkl_fft.fft2(self.x)

    def peakmem_ifft2(self, shape, dtype):
        mkl_fft.ifft2(self.x)


# ---------------------------------------------------------------------------
# N-D complex FFT (3-D)
# ---------------------------------------------------------------------------


class PeakMemFFTnD(BenchC2C):
    """Peak RSS for N-D complex FFT (3-D shapes)."""

    params = [_SHAPES_3D, _DTYPES_REDUCED]
    param_names = ["shape", "dtype"]

    def peakmem_fftn(self, shape, dtype):
        mkl_fft.fftn(self.x)

    def peakmem_ifftn(self, shape, dtype):
        mkl_fft.ifftn(self.x)
