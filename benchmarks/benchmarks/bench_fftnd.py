"""Benchmarks for 2-D and N-D FFT operations using the mkl_fft root API."""

import numpy as np

import mkl_fft

_RNG_SEED = 42


def _make_input(rng, shape, dtype):
    """Return an array of the given *shape* and *dtype*.

    Complex dtypes are populated with non-zero imaginary parts so the
    benchmark exercises a genuine complex transform path.
    """
    dt = np.dtype(dtype)
    if dt.kind == "c":
        return (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ).astype(dt)
    return rng.standard_normal(shape).astype(dt)


# ---------------------------------------------------------------------------
# 2-D complex-to-complex (power-of-two, square + non-square)
# ---------------------------------------------------------------------------


class TimeFFT2D:
    """Forward and inverse 2-D FFT — square and non-square shapes."""

    params = [
        [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (256, 128),
            (512, 256),  # non-square
        ],
        ["float32", "float64", "complex64", "complex128"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        self.x = _make_input(rng, shape, dtype)

    def time_fft2(self, shape, dtype):
        mkl_fft.fft2(self.x)

    def time_ifft2(self, shape, dtype):
        mkl_fft.ifft2(self.x)


# ---------------------------------------------------------------------------
# 2-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class TimeRFFT2D:
    """Forward rfft2 and inverse irfft2."""

    params = [
        [(64, 64), (128, 128), (256, 256), (512, 512)],
        ["float32", "float64"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        half_shape = (shape[0], shape[1] // 2 + 1)
        self.x_real = rng.standard_normal(shape).astype(dtype)
        # irfft2 input: complex half-spectrum — shape (M, N//2+1)
        self.x_complex = (
            rng.standard_normal(half_shape)
            + 1j * rng.standard_normal(half_shape)
        ).astype(cdtype)

    def time_rfft2(self, shape, dtype):
        mkl_fft.rfft2(self.x_real)

    def time_irfft2(self, shape, dtype):
        mkl_fft.irfft2(self.x_complex, s=shape)


# ---------------------------------------------------------------------------
# 2-D complex-to-complex (non-power-of-two)
# ---------------------------------------------------------------------------


class TimeFFT2DNonPow2:
    """Forward and inverse 2-D FFT — non-power-of-two sizes."""

    params = [
        [
            (96, 96),
            (100, 100),
            (270, 270),
            (500, 500),
            (100, 200),  # non-square non-pow2
        ],
        ["float64", "complex128"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        self.x = _make_input(rng, shape, dtype)

    def time_fft2(self, shape, dtype):
        mkl_fft.fft2(self.x)

    def time_ifft2(self, shape, dtype):
        mkl_fft.ifft2(self.x)


# ---------------------------------------------------------------------------
# N-D complex-to-complex (3-D cubes + non-cubic shape)
# ---------------------------------------------------------------------------


class TimeFFTnD:
    """Forward and inverse N-D FFT."""

    params = [
        [
            (16, 16, 16),
            (32, 32, 32),
            (64, 64, 64),
            (32, 64, 128),  # non-cubic
        ],
        ["float32", "float64", "complex64", "complex128"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        self.x = _make_input(rng, shape, dtype)

    def time_fftn(self, shape, dtype):
        mkl_fft.fftn(self.x)

    def time_ifftn(self, shape, dtype):
        mkl_fft.ifftn(self.x)


# ---------------------------------------------------------------------------
# N-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class TimeRFFTnD:
    """Forward rfftn and inverse irfftn."""

    params = [
        [(16, 16, 16), (32, 32, 32), (64, 64, 64)],
        ["float32", "float64"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        # irfftn input: complex half-spectrum — last axis is shape[-1]//2+1
        half_shape = shape[:-1] + (shape[-1] // 2 + 1,)
        self.x_real = rng.standard_normal(shape).astype(dtype)
        self.x_complex = (
            rng.standard_normal(half_shape)
            + 1j * rng.standard_normal(half_shape)
        ).astype(cdtype)

    def time_rfftn(self, shape, dtype):
        mkl_fft.rfftn(self.x_real)

    def time_irfftn(self, shape, dtype):
        mkl_fft.irfftn(self.x_complex, s=shape)


# ---------------------------------------------------------------------------
# N-D complex-to-complex (non-power-of-two 3-D)
# ---------------------------------------------------------------------------


class TimeFFTnDNonPow2:
    """Forward and inverse N-D FFT — non-power-of-two sizes."""

    params = [
        [
            (24, 24, 24),
            (30, 30, 30),
            (50, 50, 50),
            (30, 40, 50),  # non-cubic non-pow2
        ],
        ["float64", "complex128"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        self.x = _make_input(rng, shape, dtype)

    def time_fftn(self, shape, dtype):
        mkl_fft.fftn(self.x)

    def time_ifftn(self, shape, dtype):
        mkl_fft.ifftn(self.x)
