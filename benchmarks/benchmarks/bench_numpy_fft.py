"""Benchmarks for mkl_fft.interfaces.numpy_fft.

Covers every function exported by the interface:
  fft / ifft        — 1-D C2C
  rfft / irfft      — 1-D R2C / C2R
  hfft / ihfft      — 1-D Hermitian
  fft2 / ifft2      — 2-D C2C
  rfft2 / irfft2    — 2-D R2C / C2R
  fftn / ifftn      — N-D C2C
  rfftn / irfftn    — N-D R2C / C2R
"""

import numpy as np

from mkl_fft.interfaces import numpy_fft

_RNG_SEED = 42


def _make_input(rng, shape, dtype):
    """Return an array of *shape* and *dtype*.

    Complex dtypes get non-zero imaginary parts for a realistic signal.
    *shape* may be an int (1-D) or a tuple.
    """
    dt = np.dtype(dtype)
    s = (shape,) if isinstance(shape, int) else shape
    if dt.kind == "c":
        return (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(dt)
    return rng.standard_normal(s).astype(dt)


# ---------------------------------------------------------------------------
# 1-D complex-to-complex
# ---------------------------------------------------------------------------


class TimeC2C1D:
    """numpy_fft.fft / ifft — 1-D."""

    params = [
        [256, 1024, 16384],
        ["float32", "float64", "complex64", "complex128"],
    ]
    param_names = ["n", "dtype"]

    def setup(self, n, dtype):
        self.x = _make_input(np.random.default_rng(_RNG_SEED), n, dtype)

    def time_fft(self, n, dtype):
        numpy_fft.fft(self.x)

    def time_ifft(self, n, dtype):
        numpy_fft.ifft(self.x)


# ---------------------------------------------------------------------------
# 1-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class TimeRC1D:
    """numpy_fft.rfft / irfft — 1-D."""

    params = [
        [256, 1024, 16384],
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
        numpy_fft.rfft(self.x_real)

    def time_irfft(self, n, dtype):
        numpy_fft.irfft(self.x_complex, n=n)


# ---------------------------------------------------------------------------
# 1-D Hermitian
# hfft:  input complex length n//2+1  →  output real length n
# ihfft: input real  length n         →  output complex length n//2+1
# ---------------------------------------------------------------------------


class TimeHermitian1D:
    """numpy_fft.hfft / ihfft — 1-D Hermitian.

    *dtype* is the **output** dtype of hfft (real); the inverse ihfft
    takes the same real input and produces the corresponding complex output.
    """

    params = [
        [256, 1024, 16384],
        ["float32", "float64"],
    ]
    param_names = ["n", "dtype"]

    def setup(self, n, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        # hfft input: complex half-spectrum of length n//2+1
        self.x_hfft = (
            rng.standard_normal(n // 2 + 1)
            + 1j * rng.standard_normal(n // 2 + 1)
        ).astype(cdtype)
        # ihfft input: real signal of length n
        self.x_ihfft = rng.standard_normal(n).astype(dtype)

    def time_hfft(self, n, dtype):
        numpy_fft.hfft(self.x_hfft, n=n)

    def time_ihfft(self, n, dtype):
        numpy_fft.ihfft(self.x_ihfft)


# ---------------------------------------------------------------------------
# 2-D complex-to-complex
# ---------------------------------------------------------------------------


class TimeC2C2D:
    """numpy_fft.fft2 / ifft2 — 2-D."""

    params = [
        [(64, 64), (256, 256), (512, 512)],
        ["float64", "complex128"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        self.x = _make_input(np.random.default_rng(_RNG_SEED), shape, dtype)

    def time_fft2(self, shape, dtype):
        numpy_fft.fft2(self.x)

    def time_ifft2(self, shape, dtype):
        numpy_fft.ifft2(self.x)


# ---------------------------------------------------------------------------
# 2-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class TimeRC2D:
    """numpy_fft.rfft2 / irfft2 — 2-D."""

    params = [
        [(64, 64), (256, 256), (512, 512)],
        ["float32", "float64"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        half_shape = (shape[0], shape[1] // 2 + 1)
        self.x_real = rng.standard_normal(shape).astype(dtype)
        self.x_complex = (
            rng.standard_normal(half_shape)
            + 1j * rng.standard_normal(half_shape)
        ).astype(cdtype)

    def time_rfft2(self, shape, dtype):
        numpy_fft.rfft2(self.x_real)

    def time_irfft2(self, shape, dtype):
        numpy_fft.irfft2(self.x_complex, s=shape)


# ---------------------------------------------------------------------------
# N-D complex-to-complex
# ---------------------------------------------------------------------------


class TimeCCND:
    """numpy_fft.fftn / ifftn — N-D."""

    params = [
        [(16, 16, 16), (32, 32, 32), (64, 64, 64)],
        ["float64", "complex128"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        self.x = _make_input(np.random.default_rng(_RNG_SEED), shape, dtype)

    def time_fftn(self, shape, dtype):
        numpy_fft.fftn(self.x)

    def time_ifftn(self, shape, dtype):
        numpy_fft.ifftn(self.x)


# ---------------------------------------------------------------------------
# N-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class TimeRCND:
    """numpy_fft.rfftn / irfftn — N-D."""

    params = [
        [(16, 16, 16), (32, 32, 32), (64, 64, 64)],
        ["float32", "float64"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        half_shape = shape[:-1] + (shape[-1] // 2 + 1,)
        self.x_real = rng.standard_normal(shape).astype(dtype)
        self.x_complex = (
            rng.standard_normal(half_shape)
            + 1j * rng.standard_normal(half_shape)
        ).astype(cdtype)

    def time_rfftn(self, shape, dtype):
        numpy_fft.rfftn(self.x_real)

    def time_irfftn(self, shape, dtype):
        numpy_fft.irfftn(self.x_complex, s=shape)
