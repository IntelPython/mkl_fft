"""Benchmarks for mkl_fft.interfaces.{numpy_fft, scipy_fft}.

A single ``module`` parameter selects the interface, following SciPy's
benchmark layout (scipy/benchmarks/benchmarks/fft_basic.py).

Covered transforms:
  fft / ifft        — 1-D C2C
  rfft / irfft      — 1-D R2C / C2R
  hfft / ihfft      — 1-D Hermitian
  fft2 / ifft2      — 2-D C2C
  rfft2 / irfft2    — 2-D R2C / C2R
  hfft2 / ihfft2    — 2-D Hermitian (scipy_fft only)
  fftn / ifftn      — N-D C2C
  rfftn / irfftn    — N-D R2C / C2R
  hfftn / ihfftn    — N-D Hermitian (scipy_fft only)
"""

from mkl_fft.interfaces import numpy_fft, scipy_fft

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
_MODULES = ["numpy_fft", "scipy_fft"]
_MODULES_SCIPY_ONLY = ["scipy_fft"]
_MODULE_MAP = {"numpy_fft": numpy_fft, "scipy_fft": scipy_fft}


# ---------------------------------------------------------------------------
# 1-D complex-to-complex
# ---------------------------------------------------------------------------


class BenchC2C1D(BenchC2C):
    """fft / ifft — 1-D."""

    params = [_SIZES_1D, _DTYPES_ALL, _MODULES]
    param_names = ["n", "dtype", "module"]

    def setup(self, n, dtype, module):
        super().setup(n, dtype)
        mod = _MODULE_MAP[module]
        self.fft = mod.fft
        self.ifft = mod.ifft
        # Prime the MKL DFTI descriptor cache so the first measured
        # iteration doesn't pay the one-time descriptor-creation cost.
        # ASV's warmup_time (default 0.1s) would normally cover this,
        # but doing it explicitly removes the dependency on that default.
        self.fft(self.x)
        self.ifft(self.x)

    def time_fft(self, n, dtype, module):
        self.fft(self.x)

    def time_ifft(self, n, dtype, module):
        self.ifft(self.x)


# ---------------------------------------------------------------------------
# 1-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class BenchRC1D(BenchR2C):
    """rfft / irfft — 1-D."""

    params = [_SIZES_1D, _DTYPES_REAL, _MODULES]
    param_names = ["n", "dtype", "module"]

    def setup(self, n, dtype, module):
        super().setup(n, dtype)
        mod = _MODULE_MAP[module]
        self.rfft = mod.rfft
        self.irfft = mod.irfft
        # Prime the DFTI descriptor cache (see BenchC2C1D.setup).
        self.rfft(self.x_real)
        self.irfft(self.x_complex, n=n)

    def time_rfft(self, n, dtype, module):
        self.rfft(self.x_real)

    def time_irfft(self, n, dtype, module):
        self.irfft(self.x_complex, n=n)


# ---------------------------------------------------------------------------
# 1-D Hermitian
# hfft:  input complex length n//2+1  →  output real length n
# ihfft: input real  length n         →  output complex length n//2+1
# ---------------------------------------------------------------------------


class BenchHermitian1D(BenchR2C):
    """hfft / ihfft — 1-D Hermitian.

    *dtype* is the **output** dtype of hfft (real); the inverse ihfft
    takes the same real input and produces the corresponding complex output.
    """

    params = [_SIZES_1D, _DTYPES_REAL, _MODULES]
    param_names = ["n", "dtype", "module"]

    def setup(self, n, dtype, module):
        super().setup(n, dtype)
        mod = _MODULE_MAP[module]
        self.hfft = mod.hfft
        self.ihfft = mod.ihfft
        # Prime the DFTI descriptor cache (see BenchC2C1D.setup).
        self.hfft(self.x_complex, n=n)
        self.ihfft(self.x_real)

    def time_hfft(self, n, dtype, module):
        self.hfft(self.x_complex, n=n)

    def time_ihfft(self, n, dtype, module):
        self.ihfft(self.x_real)


# ---------------------------------------------------------------------------
# 2-D complex-to-complex
# ---------------------------------------------------------------------------


class BenchC2C2D(BenchC2C):
    """fft2 / ifft2 — 2-D."""

    params = [_SHAPES_2D_IFACE, _DTYPES_REDUCED, _MODULES]
    param_names = ["shape", "dtype", "module"]

    def setup(self, shape, dtype, module):
        super().setup(shape, dtype)
        mod = _MODULE_MAP[module]
        self.fft2 = mod.fft2
        self.ifft2 = mod.ifft2
        # Prime the DFTI descriptor cache (see BenchC2C1D.setup).
        self.fft2(self.x)
        self.ifft2(self.x)

    def time_fft2(self, shape, dtype, module):
        self.fft2(self.x)

    def time_ifft2(self, shape, dtype, module):
        self.ifft2(self.x)


# ---------------------------------------------------------------------------
# 2-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class BenchRC2D(BenchR2C):
    """rfft2 / irfft2 — 2-D."""

    params = [_SHAPES_2D_IFACE, _DTYPES_REAL, _MODULES]
    param_names = ["shape", "dtype", "module"]

    def setup(self, shape, dtype, module):
        super().setup(shape, dtype)
        mod = _MODULE_MAP[module]
        self.rfft2 = mod.rfft2
        self.irfft2 = mod.irfft2
        # Prime the DFTI descriptor cache (see BenchC2C1D.setup).
        self.rfft2(self.x_real)
        self.irfft2(self.x_complex, s=shape)

    def time_rfft2(self, shape, dtype, module):
        self.rfft2(self.x_real)

    def time_irfft2(self, shape, dtype, module):
        self.irfft2(self.x_complex, s=shape)


# ---------------------------------------------------------------------------
# 2-D Hermitian (scipy_fft only — not in numpy_fft interface)
# hfft2:  input complex shape (M, N//2+1)  →  output real shape (M, N)
# ihfft2: input real  shape (M, N)         →  output complex shape (M, N//2+1)
# ---------------------------------------------------------------------------


class BenchHermitian2D(BenchR2C):
    """scipy_fft.hfft2 / ihfft2 — 2-D Hermitian.

    *dtype* is the **output** dtype of hfft2 (real).
    """

    params = [_SHAPES_2D_IFACE, _DTYPES_REAL, _MODULES_SCIPY_ONLY]
    param_names = ["shape", "dtype", "module"]

    def setup(self, shape, dtype, module):
        super().setup(shape, dtype)
        mod = _MODULE_MAP[module]
        self.hfft2 = mod.hfft2
        self.ihfft2 = mod.ihfft2
        # Prime the DFTI descriptor cache (see BenchC2C1D.setup).
        self.hfft2(self.x_complex, s=shape)
        self.ihfft2(self.x_real)

    def time_hfft2(self, shape, dtype, module):
        self.hfft2(self.x_complex, s=shape)

    def time_ihfft2(self, shape, dtype, module):
        self.ihfft2(self.x_real)


# ---------------------------------------------------------------------------
# N-D complex-to-complex
# ---------------------------------------------------------------------------


class BenchC2CND(BenchC2C):
    """fftn / ifftn — N-D."""

    params = [_SHAPES_3D, _DTYPES_REDUCED, _MODULES]
    param_names = ["shape", "dtype", "module"]

    def setup(self, shape, dtype, module):
        super().setup(shape, dtype)
        mod = _MODULE_MAP[module]
        self.fftn = mod.fftn
        self.ifftn = mod.ifftn
        # Prime the DFTI descriptor cache (see BenchC2C1D.setup).
        self.fftn(self.x)
        self.ifftn(self.x)

    def time_fftn(self, shape, dtype, module):
        self.fftn(self.x)

    def time_ifftn(self, shape, dtype, module):
        self.ifftn(self.x)


# ---------------------------------------------------------------------------
# N-D real-to-complex / complex-to-real
# ---------------------------------------------------------------------------


class BenchRCND(BenchR2C):
    """rfftn / irfftn — N-D."""

    params = [_SHAPES_3D, _DTYPES_REAL, _MODULES]
    param_names = ["shape", "dtype", "module"]

    def setup(self, shape, dtype, module):
        super().setup(shape, dtype)
        mod = _MODULE_MAP[module]
        self.rfftn = mod.rfftn
        self.irfftn = mod.irfftn
        # Prime the DFTI descriptor cache (see BenchC2C1D.setup).
        self.rfftn(self.x_real)
        self.irfftn(self.x_complex, s=shape)

    def time_rfftn(self, shape, dtype, module):
        self.rfftn(self.x_real)

    def time_irfftn(self, shape, dtype, module):
        self.irfftn(self.x_complex, s=shape)


# ---------------------------------------------------------------------------
# N-D Hermitian (scipy_fft only)
# hfftn:  input complex, last axis length s[-1]//2+1  →  output real shape s
# ihfftn: input real  shape s  →  output complex, last axis length s[-1]//2+1
# ---------------------------------------------------------------------------


class BenchHermitianND(BenchR2C):
    """scipy_fft.hfftn / ihfftn — N-D Hermitian.

    *dtype* is the **output** dtype of hfftn (real).
    """

    params = [_SHAPES_3D, _DTYPES_REAL, _MODULES_SCIPY_ONLY]
    param_names = ["shape", "dtype", "module"]

    def setup(self, shape, dtype, module):
        super().setup(shape, dtype)
        mod = _MODULE_MAP[module]
        self.hfftn = mod.hfftn
        self.ihfftn = mod.ihfftn
        # Prime the DFTI descriptor cache (see BenchC2C1D.setup).
        self.hfftn(self.x_complex, s=shape)
        self.ihfftn(self.x_real)

    def time_hfftn(self, shape, dtype, module):
        self.hfftn(self.x_complex, s=shape)

    def time_ihfftn(self, shape, dtype, module):
        self.ihfftn(self.x_real)
