"""Peak-memory benchmarks for FFT operations.

Measures peak RSS (resident set size) to detect memory regressions
in the mkl_fft root API across 1-D, 2-D, and 3-D transforms.
"""

import numpy as np

import mkl_fft

_RNG_SEED = 42


def _make_input(rng, shape, dtype):
    dt = np.dtype(dtype)
    s = (shape,) if isinstance(shape, int) else shape
    if dt.kind == "c":
        return (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(dt)
    return rng.standard_normal(s).astype(dt)


# ---------------------------------------------------------------------------
# 1-D complex FFT
# ---------------------------------------------------------------------------


class PeakMemFFT1D:
    """Peak RSS for 1-D complex FFT."""

    params = [
        [1024, 16384, 65536, 262144],
        ["float64", "complex128"],
    ]
    param_names = ["n", "dtype"]

    def setup(self, n, dtype):
        self.x = _make_input(np.random.default_rng(_RNG_SEED), n, dtype)

    def peakmem_fft(self, n, dtype):
        mkl_fft.fft(self.x)

    def peakmem_ifft(self, n, dtype):
        mkl_fft.ifft(self.x)


# ---------------------------------------------------------------------------
# 1-D real FFT
# ---------------------------------------------------------------------------


class PeakMemRFFT1D:
    """Peak RSS for 1-D real FFT (forward and inverse)."""

    params = [
        [1024, 16384, 65536, 262144],
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

    def peakmem_rfft(self, n, dtype):
        mkl_fft.rfft(self.x_real)

    def peakmem_irfft(self, n, dtype):
        mkl_fft.irfft(self.x_complex, n=n)


# ---------------------------------------------------------------------------
# 2-D complex FFT
# ---------------------------------------------------------------------------


class PeakMemFFT2D:
    """Peak RSS for 2-D complex FFT."""

    params = [
        [(64, 64), (128, 128), (256, 256), (512, 512)],
        ["float64", "complex128"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        self.x = _make_input(np.random.default_rng(_RNG_SEED), shape, dtype)

    def peakmem_fft2(self, shape, dtype):
        mkl_fft.fft2(self.x)

    def peakmem_ifft2(self, shape, dtype):
        mkl_fft.ifft2(self.x)


# ---------------------------------------------------------------------------
# N-D complex FFT (3-D)
# ---------------------------------------------------------------------------


class PeakMemFFTnD:
    """Peak RSS for N-D complex FFT (3-D shapes)."""

    params = [
        [(16, 16, 16), (32, 32, 32), (64, 64, 64)],
        ["float64", "complex128"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        self.x = _make_input(np.random.default_rng(_RNG_SEED), shape, dtype)

    def peakmem_fftn(self, shape, dtype):
        mkl_fft.fftn(self.x)

    def peakmem_ifftn(self, shape, dtype):
        mkl_fft.ifftn(self.x)
