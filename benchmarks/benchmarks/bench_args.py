"""Benchmarks for argument handling overhead.

These suites hold the transform itself fixed and vary only the keyword
arguments, isolating the cost of the Python-level argument processing that
runs before MKL is reached:

* ``norm`` other than ``None``/``"backward"`` goes through
  ``_fft_utils._compute_fwd_scale``. On small transforms the scale computation
  is a measurable fraction of the whole call.
* An explicit ``s`` that matches the input shape still reaches the direct N-D
  path, but padding or truncating dispatches through ``_fft_utils._iter_fftnd``,
  which re-normalizes shape and axes on every call and performs the pad/trim.

Every other suite in this directory calls with default arguments only.
"""

import mkl_fft

from ._utils import _DTYPES_REAL, _DTYPES_REDUCED, BenchC2C, BenchR2C

_NORMS = [None, "backward", "forward", "ortho"]

# "match" is the in-suite control: it should stay on the direct N-D path.
_S_MODES = ["match", "trunc", "pad"]


def _shape_for_mode(shape, mode):
    """Return an ``s`` argument that matches, truncates, or pads *shape*."""
    if mode == "trunc":
        return tuple(d // 2 for d in shape)
    if mode == "pad":
        return tuple(d + d // 2 for d in shape)
    return tuple(shape)


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------


class BenchNorm1D(BenchC2C):
    """fft with each ``norm`` mode.

    ``n=64`` is small enough that the transform costs about a microsecond, so
    scale-factor computation shows up directly.
    """

    params = [[64, 1024, 16384], _DTYPES_REDUCED, _NORMS]
    param_names = ["n", "dtype", "norm"]

    def setup(self, n, dtype, norm):
        super().setup(n, dtype)
        mkl_fft.fft(self.x, norm=norm)

    def time_fft(self, n, dtype, norm):
        mkl_fft.fft(self.x, norm=norm)

    def time_ifft(self, n, dtype, norm):
        mkl_fft.ifft(self.x, norm=norm)


class BenchNormND(BenchC2C):
    """fftn with each ``norm`` mode.

    For N-D transforms the forward scale is a product over the whole shape,
    so this path does strictly more work than the 1-D one.
    """

    params = [[(64, 64), (256, 256), (32, 32, 32)], ["complex128"], _NORMS]
    param_names = ["shape", "dtype", "norm"]

    def setup(self, shape, dtype, norm):
        super().setup(shape, dtype)
        mkl_fft.fftn(self.x, norm=norm)

    def time_fftn(self, shape, dtype, norm):
        mkl_fft.fftn(self.x, norm=norm)

    def time_ifftn(self, shape, dtype, norm):
        mkl_fft.ifftn(self.x, norm=norm)


class BenchNormR2C1D(BenchR2C):
    """rfft / irfft with each ``norm`` mode."""

    params = [[64, 16384], _DTYPES_REAL, _NORMS]
    param_names = ["n", "dtype", "norm"]

    def setup(self, n, dtype, norm):
        super().setup(n, dtype)
        mkl_fft.rfft(self.x_real, norm=norm)
        mkl_fft.irfft(self.x_complex, n=n, norm=norm)

    def time_rfft(self, n, dtype, norm):
        mkl_fft.rfft(self.x_real, norm=norm)

    def time_irfft(self, n, dtype, norm):
        mkl_fft.irfft(self.x_complex, n=n, norm=norm)


# ---------------------------------------------------------------------------
# explicit output shape (s)
# ---------------------------------------------------------------------------


class BenchShapeArg2D(BenchC2C):
    """fftn with an explicit ``s`` that matches, truncates, or pads."""

    params = [[(256, 256)], _DTYPES_REDUCED, _S_MODES]
    param_names = ["shape", "dtype", "mode"]

    def setup(self, shape, dtype, mode):
        super().setup(shape, dtype)
        self.s = _shape_for_mode(shape, mode)
        mkl_fft.fftn(self.x, s=self.s)

    def time_fftn(self, shape, dtype, mode):
        mkl_fft.fftn(self.x, s=self.s)

    def time_ifftn(self, shape, dtype, mode):
        mkl_fft.ifftn(self.x, s=self.s)


class BenchShapeArg3D(BenchC2C):
    """fftn with an explicit ``s`` on a 3-D array."""

    params = [[(32, 32, 32)], _DTYPES_REDUCED, _S_MODES]
    param_names = ["shape", "dtype", "mode"]

    def setup(self, shape, dtype, mode):
        super().setup(shape, dtype)
        self.s = _shape_for_mode(shape, mode)
        mkl_fft.fftn(self.x, s=self.s)

    def time_fftn(self, shape, dtype, mode):
        mkl_fft.fftn(self.x, s=self.s)

    def time_ifftn(self, shape, dtype, mode):
        mkl_fft.ifftn(self.x, s=self.s)


class BenchShapeArgR2C2D(BenchR2C):
    """rfftn with an explicit ``s`` that matches, truncates, or pads."""

    params = [[(256, 256)], _DTYPES_REAL, _S_MODES]
    param_names = ["shape", "dtype", "mode"]

    def setup(self, shape, dtype, mode):
        super().setup(shape, dtype)
        self.s = _shape_for_mode(shape, mode)
        mkl_fft.rfftn(self.x_real, s=self.s)

    def time_rfftn(self, shape, dtype, mode):
        mkl_fft.rfftn(self.x_real, s=self.s)


# ---------------------------------------------------------------------------
# 1-D explicit length (n)
# ---------------------------------------------------------------------------


class BenchLengthArg1D(BenchC2C):
    """fft with an explicit ``n`` that matches, truncates, or pads.

    Padding forces a copy into a larger buffer inside ``_pydfti._pad_array``;
    truncating returns a view of a longer in-place result.
    """

    params = [[16384], _DTYPES_REDUCED, _S_MODES]
    param_names = ["n", "dtype", "mode"]

    def setup(self, n, dtype, mode):
        super().setup(n, dtype)
        self.n = _shape_for_mode((n,), mode)[0]
        mkl_fft.fft(self.x, n=self.n)

    def time_fft(self, n, dtype, mode):
        mkl_fft.fft(self.x, n=self.n)

    def time_ifft(self, n, dtype, mode):
        mkl_fft.ifft(self.x, n=self.n)
