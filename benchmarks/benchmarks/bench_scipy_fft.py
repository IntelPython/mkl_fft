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

from mkl_fft.interfaces import scipy_fft

from ._utils import (
    _DTYPES_ALL,
    _DTYPES_REAL,
    _DTYPES_REDUCED,
    _SHAPES_2D_IFACE,
    _SHAPES_3D,
    BenchC2C,
    BenchR2C,
)

_SIZES_1D = [256, 1024, 16384]


# ---------------------------------------------------------------------------
# 1-D complex-to-complex
# ---------------------------------------------------------------------------


class BenchC2C1D(BenchC2C):
    """scipy_fft.fft / ifft — 1-D."""

    params = [_SIZES_1D, _DTYPES_ALL]
    param_names = ["n", "dtype"]

    def time_fft(self, n, dtype):
        scipy_fft.fft(self.x)

    def time_ifft(self, n, dtype):
        scipy_fft.ifft(self.x)


# ---------------------------------------------------------------------------
# 1-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class BenchRC1D(BenchR2C):
    """scipy_fft.rfft / irfft — 1-D."""

    params = [_SIZES_1D, _DTYPES_REAL]
    param_names = ["n", "dtype"]

    def time_rfft(self, n, dtype):
        scipy_fft.rfft(self.x_real)

    def time_irfft(self, n, dtype):
        scipy_fft.irfft(self.x_complex, n=n)


# ---------------------------------------------------------------------------
# 1-D Hermitian
# hfft:  input complex length n//2+1  →  output real length n
# ihfft: input real  length n         →  output complex length n//2+1
# ---------------------------------------------------------------------------


class BenchHermitian1D(BenchR2C):
    """scipy_fft.hfft / ihfft — 1-D Hermitian.

    *dtype* is the **output** dtype of hfft (real); the corresponding
    complex input dtype is derived automatically.
    """

    params = [_SIZES_1D, _DTYPES_REAL]
    param_names = ["n", "dtype"]

    def time_hfft(self, n, dtype):
        scipy_fft.hfft(self.x_complex, n=n)

    def time_ihfft(self, n, dtype):
        scipy_fft.ihfft(self.x_real)


# ---------------------------------------------------------------------------
# 2-D complex-to-complex
# ---------------------------------------------------------------------------


class BenchC2C2D(BenchC2C):
    """scipy_fft.fft2 / ifft2 — 2-D."""

    params = [_SHAPES_2D_IFACE, _DTYPES_REDUCED]
    param_names = ["shape", "dtype"]

    def time_fft2(self, shape, dtype):
        scipy_fft.fft2(self.x)

    def time_ifft2(self, shape, dtype):
        scipy_fft.ifft2(self.x)


# ---------------------------------------------------------------------------
# 2-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class BenchRC2D(BenchR2C):
    """scipy_fft.rfft2 / irfft2 — 2-D."""

    params = [_SHAPES_2D_IFACE, _DTYPES_REAL]
    param_names = ["shape", "dtype"]

    def time_rfft2(self, shape, dtype):
        scipy_fft.rfft2(self.x_real)

    def time_irfft2(self, shape, dtype):
        scipy_fft.irfft2(self.x_complex, s=shape)


# ---------------------------------------------------------------------------
# 2-D Hermitian  (scipy_fft only — not in numpy_fft interface)
# hfft2:  input complex shape (M, N//2+1)  →  output real shape (M, N)
# ihfft2: input real  shape (M, N)         →  output complex shape (M, N//2+1)
# ---------------------------------------------------------------------------


class BenchHermitian2D(BenchR2C):
    """scipy_fft.hfft2 / ihfft2 — 2-D Hermitian.

    *dtype* is the **output** dtype of hfft2 (real).
    """

    params = [_SHAPES_2D_IFACE, _DTYPES_REAL]
    param_names = ["shape", "dtype"]

    def time_hfft2(self, shape, dtype):
        scipy_fft.hfft2(self.x_complex, s=shape)

    def time_ihfft2(self, shape, dtype):
        scipy_fft.ihfft2(self.x_real)


# ---------------------------------------------------------------------------
# N-D complex-to-complex
# ---------------------------------------------------------------------------


class BenchC2CND(BenchC2C):
    """scipy_fft.fftn / ifftn — N-D."""

    params = [_SHAPES_3D, _DTYPES_REDUCED]
    param_names = ["shape", "dtype"]

    def time_fftn(self, shape, dtype):
        scipy_fft.fftn(self.x)

    def time_ifftn(self, shape, dtype):
        scipy_fft.ifftn(self.x)


# ---------------------------------------------------------------------------
# N-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class BenchRCND(BenchR2C):
    """scipy_fft.rfftn / irfftn — N-D."""

    params = [_SHAPES_3D, _DTYPES_REAL]
    param_names = ["shape", "dtype"]

    def time_rfftn(self, shape, dtype):
        scipy_fft.rfftn(self.x_real)

    def time_irfftn(self, shape, dtype):
        scipy_fft.irfftn(self.x_complex, s=shape)


# ---------------------------------------------------------------------------
# N-D Hermitian  (scipy_fft only)
# hfftn:  input complex, last axis length s[-1]//2+1  →  output real shape s
# ihfftn: input real  shape s  →  output complex, last axis length s[-1]//2+1
# ---------------------------------------------------------------------------


class BenchHermitianND(BenchR2C):
    """scipy_fft.hfftn / ihfftn — N-D Hermitian.

    *dtype* is the **output** dtype of hfftn (real).
    """

    params = [_SHAPES_3D, _DTYPES_REAL]
    param_names = ["shape", "dtype"]

    def time_hfftn(self, shape, dtype):
        scipy_fft.hfftn(self.x_complex, s=shape)

    def time_ihfftn(self, shape, dtype):
        scipy_fft.ihfftn(self.x_real)
