"""Benchmarks for axis and axes selection.

Which axes a transform is asked for changes the dispatch path, not just the
amount of arithmetic:

* ``axes=None`` (or a tuple covering every axis) reaches the batched N-D MKL
  descriptor in one call.
* A strict *subset* of axes is dispatched slice by slice over the
  complementary axes (``_fft_utils._iter_complementary``), so cost scales with
  the number of complementary slices rather than with transform size.
* For a 1-D transform of an array of rank > 2, the C backend can issue a
  single batched ``DftiCompute`` only when the axis is the first or the last
  one (``mklfft.c.src``, ``compute_strides_and_distances``). Any interior axis
  falls back to iterating one vector at a time.

The other suites in this directory always request every axis, so none of these
paths were previously visible to the dashboard.
"""

import mkl_fft

from ._utils import _DTYPES_REAL, _DTYPES_REDUCED, BenchC2C, BenchR2C

# axes subsets, plus the full-dimensional tuple as an in-suite control
_AXES_2D = [(0,), (1,), (0, 1)]
_AXES_3D = [(0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)]

# For c2r the half-spectrum input is laid out along the array's last axis, so
# only axes tuples ending on that axis are meaningful here.
_AXES_3D_R2C = [(2,), (1, 2), (0, 2), (0, 1, 2)]


# ---------------------------------------------------------------------------
# 2-D complex-to-complex over a subset of axes
# ---------------------------------------------------------------------------


class BenchAxes2D(BenchC2C):
    """fftn / ifftn over a subset of the axes of a 2-D array."""

    params = [[(128, 128), (512, 512)], _DTYPES_REDUCED, _AXES_2D]
    param_names = ["shape", "dtype", "axes"]

    def setup(self, shape, dtype, axes):
        super().setup(shape, dtype)
        mkl_fft.fftn(self.x, axes=axes)

    def time_fftn(self, shape, dtype, axes):
        mkl_fft.fftn(self.x, axes=axes)

    def time_ifftn(self, shape, dtype, axes):
        mkl_fft.ifftn(self.x, axes=axes)


# ---------------------------------------------------------------------------
# 3-D complex-to-complex over a subset of axes
# ---------------------------------------------------------------------------


class BenchAxes3D(BenchC2C):
    """fftn / ifftn over a subset of the axes of a 3-D array.

    The number of complementary slices spans three orders of magnitude across
    this parameter set: ``(0, 1, 2)`` is a single batched call, ``(1, 2)``
    iterates 32 slices, and ``(2,)`` iterates 1024. The slice count is the
    product of the untransformed axes, so larger shapes amplify the effect;
    this one is kept modest to bound suite runtime.
    """

    params = [[(32, 32, 32)], _DTYPES_REDUCED, _AXES_3D]
    param_names = ["shape", "dtype", "axes"]

    def setup(self, shape, dtype, axes):
        super().setup(shape, dtype)
        mkl_fft.fftn(self.x, axes=axes)

    def time_fftn(self, shape, dtype, axes):
        mkl_fft.fftn(self.x, axes=axes)

    def time_ifftn(self, shape, dtype, axes):
        mkl_fft.ifftn(self.x, axes=axes)


# ---------------------------------------------------------------------------
# 3-D real-to-complex / complex-to-real over a subset of axes
# ---------------------------------------------------------------------------


class BenchAxesR2C3D(BenchR2C):
    """rfftn / irfftn over a subset of the axes of a 3-D array."""

    params = [[(32, 32, 32)], _DTYPES_REAL, _AXES_3D_R2C]
    param_names = ["shape", "dtype", "axes"]

    def setup(self, shape, dtype, axes):
        super().setup(shape, dtype)
        # shape of the result along the requested axes
        self.s = tuple(shape[ax] for ax in axes)
        mkl_fft.rfftn(self.x_real, axes=axes)
        mkl_fft.irfftn(self.x_complex, s=self.s, axes=axes)

    def time_rfftn(self, shape, dtype, axes):
        mkl_fft.rfftn(self.x_real, axes=axes)

    def time_irfftn(self, shape, dtype, axes):
        mkl_fft.irfftn(self.x_complex, s=self.s, axes=axes)


# ---------------------------------------------------------------------------
# 1-D transform along each axis of a higher-rank array
# ---------------------------------------------------------------------------


class BenchAxis3D(BenchC2C):
    """fft / ifft along each individual axis of a 3-D array.

    The backend batches the whole transform into one ``DftiCompute`` call only
    for ``axis=0`` and ``axis=2``; ``axis=1`` iterates vector by vector.
    """

    params = [[(32, 32, 32), (64, 64, 64)], _DTYPES_REDUCED, [0, 1, 2]]
    param_names = ["shape", "dtype", "axis"]

    def setup(self, shape, dtype, axis):
        super().setup(shape, dtype)
        mkl_fft.fft(self.x, axis=axis)

    def time_fft(self, shape, dtype, axis):
        mkl_fft.fft(self.x, axis=axis)

    def time_ifft(self, shape, dtype, axis):
        mkl_fft.ifft(self.x, axis=axis)


class BenchAxis4D(BenchC2C):
    """fft along each individual axis of a 4-D array.

    A rank-4 array has two interior axes, so the batched and per-vector paths
    are exercised twice each within one parameter sweep.
    """

    params = [[(16, 16, 16, 16)], ["complex128"], [0, 1, 2, 3]]
    param_names = ["shape", "dtype", "axis"]

    def setup(self, shape, dtype, axis):
        super().setup(shape, dtype)
        mkl_fft.fft(self.x, axis=axis)

    def time_fft(self, shape, dtype, axis):
        mkl_fft.fft(self.x, axis=axis)


class BenchAxisR2C3D(BenchR2C):
    """rfft / irfft along each individual axis of a 3-D array."""

    params = [[(64, 64, 64)], _DTYPES_REAL, [0, 1, 2]]
    param_names = ["shape", "dtype", "axis"]

    def setup(self, shape, dtype, axis):
        super().setup(shape, dtype)
        mkl_fft.rfft(self.x_real, axis=axis)

    def time_rfft(self, shape, dtype, axis):
        mkl_fft.rfft(self.x_real, axis=axis)
