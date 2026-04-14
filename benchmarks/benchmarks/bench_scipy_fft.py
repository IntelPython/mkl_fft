"""Benchmarks for mkl_fft.interfaces.scipy_fft.

Covers every function exported by the interface:
  fft / ifft          — 1-D C2C
  rfft / irfft        — 1-D R2C / C2R
  hfft / ihfft        — 1-D Hermitian
  fft2 / ifft2        — 2-D C2C
  rfft2 / irfft2      — 2-D R2C / C2R
  hfft2 / ihfft2      — 2-D Hermitian  (scipy_fft only)
  fftn / ifftn        — N-D C2C
  rfftn / irfftn      — N-D R2C / C2R
  hfftn / ihfftn      — N-D Hermitian  (scipy_fft only)
"""

import numpy as np
from mkl_fft.interfaces import scipy_fft


def _make_input(rng, shape, dtype):
    """Return an array of *shape* and *dtype*.

    Complex dtypes get non-zero imaginary parts for a realistic signal.
    *shape* may be an int (1-D) or a tuple.
    """
    dt = np.dtype(dtype)
    s = (shape,) if isinstance(shape, int) else shape
    if dt.kind == "c":
        return (rng.randn(*s) + 1j * rng.randn(*s)).astype(dt)
    return rng.randn(*s).astype(dt)


# ---------------------------------------------------------------------------
# 1-D complex-to-complex
# ---------------------------------------------------------------------------

class TimeC2C1D:
    """scipy_fft.fft / ifft — 1-D."""

    params = [
        [256, 1024, 16384],
        ["float32", "float64", "complex64", "complex128"],
    ]
    param_names = ["n", "dtype"]

    def setup(self, n, dtype):
        self.x = _make_input(np.random.RandomState(42), n, dtype)

    def time_fft(self, n, dtype):
        scipy_fft.fft(self.x)

    def time_ifft(self, n, dtype):
        scipy_fft.ifft(self.x)


# ---------------------------------------------------------------------------
# 1-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------

class TimeRC1D:
    """scipy_fft.rfft / irfft — 1-D."""

    params = [
        [256, 1024, 16384],
        ["float32", "float64"],
    ]
    param_names = ["n", "dtype"]

    def setup(self, n, dtype):
        rng = np.random.RandomState(42)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        self.x_real = rng.randn(n).astype(dtype)
        self.x_complex = (
            rng.randn(n // 2 + 1) + 1j * rng.randn(n // 2 + 1)
        ).astype(cdtype)

    def time_rfft(self, n, dtype):
        scipy_fft.rfft(self.x_real)

    def time_irfft(self, n, dtype):
        scipy_fft.irfft(self.x_complex, n=n)


# ---------------------------------------------------------------------------
# 1-D Hermitian
# hfft:  input complex length n//2+1  →  output real length n
# ihfft: input real  length n         →  output complex length n//2+1
# ---------------------------------------------------------------------------

class TimeHermitian1D:
    """scipy_fft.hfft / ihfft — 1-D Hermitian.

    *dtype* is the **output** dtype of hfft (real); the corresponding
    complex input dtype is derived automatically.
    """

    params = [
        [256, 1024, 16384],
        ["float32", "float64"],
    ]
    param_names = ["n", "dtype"]

    def setup(self, n, dtype):
        rng = np.random.RandomState(42)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        self.x_hfft = (
            rng.randn(n // 2 + 1) + 1j * rng.randn(n // 2 + 1)
        ).astype(cdtype)
        self.x_ihfft = rng.randn(n).astype(dtype)

    def time_hfft(self, n, dtype):
        scipy_fft.hfft(self.x_hfft, n=n)

    def time_ihfft(self, n, dtype):
        scipy_fft.ihfft(self.x_ihfft)


# ---------------------------------------------------------------------------
# 2-D complex-to-complex
# ---------------------------------------------------------------------------

class TimeC2C2D:
    """scipy_fft.fft2 / ifft2 — 2-D."""

    params = [
        [(64, 64), (256, 256), (512, 512)],
        ["float64", "complex128"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        self.x = _make_input(np.random.RandomState(42), shape, dtype)

    def time_fft2(self, shape, dtype):
        scipy_fft.fft2(self.x)

    def time_ifft2(self, shape, dtype):
        scipy_fft.ifft2(self.x)


# ---------------------------------------------------------------------------
# 2-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------

class TimeRC2D:
    """scipy_fft.rfft2 / irfft2 — 2-D."""

    params = [
        [(64, 64), (256, 256), (512, 512)],
        ["float32", "float64"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.RandomState(42)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        half_shape = (shape[0], shape[1] // 2 + 1)
        self.x_real = rng.randn(*shape).astype(dtype)
        self.x_complex = (
            rng.randn(*half_shape) + 1j * rng.randn(*half_shape)
        ).astype(cdtype)

    def time_rfft2(self, shape, dtype):
        scipy_fft.rfft2(self.x_real)

    def time_irfft2(self, shape, dtype):
        scipy_fft.irfft2(self.x_complex, s=shape)


# ---------------------------------------------------------------------------
# 2-D Hermitian  (scipy_fft only — not in numpy_fft interface)
# hfft2:  input complex shape (M, N//2+1)  →  output real shape (M, N)
# ihfft2: input real  shape (M, N)         →  output complex shape (M, N//2+1)
# ---------------------------------------------------------------------------

class TimeHermitian2D:
    """scipy_fft.hfft2 / ihfft2 — 2-D Hermitian.

    *dtype* is the **output** dtype of hfft2 (real).
    """

    params = [
        [(64, 64), (256, 256), (512, 512)],
        ["float32", "float64"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.RandomState(42)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        half_shape = (shape[0], shape[1] // 2 + 1)
        self.x_hfft2 = (
            rng.randn(*half_shape) + 1j * rng.randn(*half_shape)
        ).astype(cdtype)
        self.x_ihfft2 = rng.randn(*shape).astype(dtype)

    def time_hfft2(self, shape, dtype):
        scipy_fft.hfft2(self.x_hfft2, s=shape)

    def time_ihfft2(self, shape, dtype):
        scipy_fft.ihfft2(self.x_ihfft2)


# ---------------------------------------------------------------------------
# N-D complex-to-complex
# ---------------------------------------------------------------------------

class TimeCCND:
    """scipy_fft.fftn / ifftn — N-D."""

    params = [
        [(16, 16, 16), (32, 32, 32), (64, 64, 64)],
        ["float64", "complex128"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        self.x = _make_input(np.random.RandomState(42), shape, dtype)

    def time_fftn(self, shape, dtype):
        scipy_fft.fftn(self.x)

    def time_ifftn(self, shape, dtype):
        scipy_fft.ifftn(self.x)


# ---------------------------------------------------------------------------
# N-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------

class TimeRCND:
    """scipy_fft.rfftn / irfftn — N-D."""

    params = [
        [(16, 16, 16), (32, 32, 32), (64, 64, 64)],
        ["float32", "float64"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.RandomState(42)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        half_shape = shape[:-1] + (shape[-1] // 2 + 1,)
        self.x_real = rng.randn(*shape).astype(dtype)
        self.x_complex = (
            rng.randn(*half_shape) + 1j * rng.randn(*half_shape)
        ).astype(cdtype)

    def time_rfftn(self, shape, dtype):
        scipy_fft.rfftn(self.x_real)

    def time_irfftn(self, shape, dtype):
        scipy_fft.irfftn(self.x_complex, s=shape)


# ---------------------------------------------------------------------------
# N-D Hermitian  (scipy_fft only)
# hfftn:  input complex, last axis length s[-1]//2+1  →  output real shape s
# ihfftn: input real  shape s  →  output complex, last axis length s[-1]//2+1
# ---------------------------------------------------------------------------

class TimeHermitianND:
    """scipy_fft.hfftn / ihfftn — N-D Hermitian.

    *dtype* is the **output** dtype of hfftn (real).
    """

    params = [
        [(16, 16, 16), (32, 32, 32), (64, 64, 64)],
        ["float32", "float64"],
    ]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        rng = np.random.RandomState(42)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        # hfftn input: last axis has length shape[-1]//2+1
        half_shape = shape[:-1] + (shape[-1] // 2 + 1,)
        self.x_hfftn = (
            rng.randn(*half_shape) + 1j * rng.randn(*half_shape)
        ).astype(cdtype)
        self.x_ihfftn = rng.randn(*shape).astype(dtype)

    def time_hfftn(self, shape, dtype):
        scipy_fft.hfftn(self.x_hfftn, s=shape)

    def time_ihfftn(self, shape, dtype):
        scipy_fft.ihfftn(self.x_ihfftn)
