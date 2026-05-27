"""Benchmarks for 2-D and N-D FFT operations using the mkl_fft root API."""

import mkl_fft

from ._utils import (
    _DTYPES_ALL,
    _DTYPES_REAL,
    _DTYPES_REDUCED,
    _SHAPES_2D,
    _SHAPES_3D,
    BenchC2C,
    BenchR2C,
)

# ---------------------------------------------------------------------------
# 2-D complex-to-complex (power-of-two, square + non-square)
# ---------------------------------------------------------------------------


class BenchFFT2D(BenchC2C):
    """Forward and inverse 2-D FFT — square and non-square shapes."""

    params = [
        _SHAPES_2D + [(256, 128), (512, 256)],
        _DTYPES_ALL,
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        super().setup(shape, dtype)
        # Prime the MKL DFTI descriptor cache so the first measured
        # iteration doesn't pay the one-time descriptor-creation cost.
        # ASV's warmup_time (default 0.1s) would normally cover this,
        # but doing it explicitly removes the dependency on that default.
        mkl_fft.fft2(self.x)
        mkl_fft.ifft2(self.x)

    def time_fft2(self, shape, dtype):
        mkl_fft.fft2(self.x)

    def time_ifft2(self, shape, dtype):
        mkl_fft.ifft2(self.x)


# ---------------------------------------------------------------------------
# 2-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class BenchRFFT2D(BenchR2C):
    """Forward rfft2 and inverse irfft2."""

    params = [_SHAPES_2D, _DTYPES_REAL]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        super().setup(shape, dtype)
        # Prime the DFTI descriptor cache (see BenchFFT2D.setup).
        mkl_fft.rfft2(self.x_real)
        mkl_fft.irfft2(self.x_complex, s=shape)

    def time_rfft2(self, shape, dtype):
        mkl_fft.rfft2(self.x_real)

    def time_irfft2(self, shape, dtype):
        mkl_fft.irfft2(self.x_complex, s=shape)


# ---------------------------------------------------------------------------
# 2-D complex-to-complex (non-power-of-two)
# ---------------------------------------------------------------------------


class BenchFFT2DNonPow2(BenchC2C):
    """Forward and inverse 2-D FFT — non-power-of-two sizes."""

    params = [
        [
            (96, 96),
            (100, 100),
            (270, 270),
            (500, 500),
            (100, 200),  # non-square non-pow2
        ],
        _DTYPES_REDUCED,
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        super().setup(shape, dtype)
        # Prime the DFTI descriptor cache (see BenchFFT2D.setup).
        mkl_fft.fft2(self.x)
        mkl_fft.ifft2(self.x)

    def time_fft2(self, shape, dtype):
        mkl_fft.fft2(self.x)

    def time_ifft2(self, shape, dtype):
        mkl_fft.ifft2(self.x)


# ---------------------------------------------------------------------------
# N-D complex-to-complex (3-D cubes + non-cubic shape)
# ---------------------------------------------------------------------------


class BenchFFTnD(BenchC2C):
    """Forward and inverse N-D FFT."""

    params = [
        _SHAPES_3D + [(32, 64, 128)],
        _DTYPES_ALL,
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        super().setup(shape, dtype)
        # Prime the DFTI descriptor cache (see BenchFFT2D.setup).
        mkl_fft.fftn(self.x)
        mkl_fft.ifftn(self.x)

    def time_fftn(self, shape, dtype):
        mkl_fft.fftn(self.x)

    def time_ifftn(self, shape, dtype):
        mkl_fft.ifftn(self.x)


# ---------------------------------------------------------------------------
# N-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class BenchRFFTnD(BenchR2C):
    """Forward rfftn and inverse irfftn."""

    params = [_SHAPES_3D, _DTYPES_REAL]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        super().setup(shape, dtype)
        # Prime the DFTI descriptor cache (see BenchFFT2D.setup).
        mkl_fft.rfftn(self.x_real)
        mkl_fft.irfftn(self.x_complex, s=shape)

    def time_rfftn(self, shape, dtype):
        mkl_fft.rfftn(self.x_real)

    def time_irfftn(self, shape, dtype):
        mkl_fft.irfftn(self.x_complex, s=shape)


# ---------------------------------------------------------------------------
# N-D complex-to-complex (non-power-of-two 3-D)
# ---------------------------------------------------------------------------


class BenchFFTnDNonPow2(BenchC2C):
    """Forward and inverse N-D FFT — non-power-of-two sizes."""

    params = [
        [
            (24, 24, 24),
            (30, 30, 30),
            (50, 50, 50),
            (30, 40, 50),  # non-cubic non-pow2
        ],
        _DTYPES_REDUCED,
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        super().setup(shape, dtype)
        # Prime the DFTI descriptor cache (see BenchFFT2D.setup).
        mkl_fft.fftn(self.x)
        mkl_fft.ifftn(self.x)

    def time_fftn(self, shape, dtype):
        mkl_fft.fftn(self.x)

    def time_ifftn(self, shape, dtype):
        mkl_fft.ifftn(self.x)
