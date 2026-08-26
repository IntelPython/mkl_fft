"""Benchmarks for input and output memory layout.

Every other suite in this directory transforms a freshly allocated
C-contiguous array. Callers routinely pass something else, and layout decides
which backend path runs:

* A C- or F-contiguous array is one segment, so the C backend can describe the
  whole batch to MKL with a single stride/distance pair.
* A non-contiguous view is not one segment, so for rank > 2 the backend falls
  back to iterating one vector at a time
  (``mklfft.c.src``, ``compute_strides_and_distances``).
* OneMKL requires ``out`` to have the same element strides as the input. When
  they differ, ``_pydfti`` allocates a temporary and copies the result into
  ``out``, costing an extra full pass.
"""

import numpy as np

import mkl_fft

from ._utils import _DTYPES_REDUCED, _RNG_SEED, _make_input

_LAYOUTS = ["C", "F", "strided"]

_OUT_KINDS = ["none", "contig", "strided"]


def _layout_input(shape, dtype, layout):
    """Return an array of *shape* and *dtype* in the requested *layout*."""
    rng = np.random.default_rng(_RNG_SEED)
    if layout == "strided":
        # A view with a gap between elements on every axis; not one segment.
        big = _make_input(rng, tuple(2 * d for d in shape), dtype)
        return big[(slice(None, None, 2),) * len(shape)]
    x = _make_input(rng, shape, dtype)
    if layout == "F":
        return np.asfortranarray(x)
    return x


# ---------------------------------------------------------------------------
# 2-D input layout
# ---------------------------------------------------------------------------


class BenchLayout2D:
    """fft2 / ifft2 over C-contiguous, F-ordered, and strided input."""

    params = [[(256, 256), (512, 512)], _DTYPES_REDUCED, _LAYOUTS]
    param_names = ["shape", "dtype", "layout"]

    def setup(self, shape, dtype, layout):
        self.x = _layout_input(shape, dtype, layout)
        mkl_fft.fft2(self.x)

    def time_fft2(self, shape, dtype, layout):
        mkl_fft.fft2(self.x)

    def time_ifft2(self, shape, dtype, layout):
        mkl_fft.ifft2(self.x)


# ---------------------------------------------------------------------------
# 3-D input layout
# ---------------------------------------------------------------------------


class BenchLayout3D:
    """fftn / ifftn over C-contiguous, F-ordered, and strided input."""

    params = [[(64, 64, 64)], _DTYPES_REDUCED, _LAYOUTS]
    param_names = ["shape", "dtype", "layout"]

    def setup(self, shape, dtype, layout):
        self.x = _layout_input(shape, dtype, layout)
        mkl_fft.fftn(self.x)

    def time_fftn(self, shape, dtype, layout):
        mkl_fft.fftn(self.x)

    def time_ifftn(self, shape, dtype, layout):
        mkl_fft.ifftn(self.x)


class BenchLayoutAxis3D:
    """1-D fft along the last axis of a 3-D array of each layout.

    A strided rank-3 input cannot be handled as one batched call even along
    the last axis, so this pairs with ``bench_axes.BenchAxis3D`` to separate
    the layout effect from the axis-position effect.
    """

    params = [[(64, 64, 64)], ["complex128"], _LAYOUTS]
    param_names = ["shape", "dtype", "layout"]

    def setup(self, shape, dtype, layout):
        self.x = _layout_input(shape, dtype, layout)
        mkl_fft.fft(self.x, axis=-1)

    def time_fft(self, shape, dtype, layout):
        mkl_fft.fft(self.x, axis=-1)


# ---------------------------------------------------------------------------
# out= layout
# ---------------------------------------------------------------------------


class BenchOut2D:
    """fft2 writing into a caller-supplied ``out`` of each layout.

    ``none`` is the control; ``contig`` should hit the in-place-into-out fast
    path; ``strided`` forces a temporary allocation plus a copy.
    """

    params = [[(512, 512)], ["complex128"], _OUT_KINDS]
    param_names = ["shape", "dtype", "out_kind"]

    def setup(self, shape, dtype, out_kind):
        rng = np.random.default_rng(_RNG_SEED)
        self.x = _make_input(rng, shape, dtype)
        if out_kind == "contig":
            self.out = np.empty(shape, dtype=dtype)
        elif out_kind == "strided":
            wide = shape[:-1] + (2 * shape[-1],)
            self.out = np.empty(wide, dtype=dtype)[..., ::2]
        else:
            self.out = None
        mkl_fft.fft2(self.x, out=self.out)

    def time_fft2(self, shape, dtype, out_kind):
        mkl_fft.fft2(self.x, out=self.out)


class BenchOut1D:
    """fft writing into a caller-supplied ``out`` of each layout."""

    params = [[65536], ["complex128"], _OUT_KINDS]
    param_names = ["n", "dtype", "out_kind"]

    def setup(self, n, dtype, out_kind):
        rng = np.random.default_rng(_RNG_SEED)
        self.x = _make_input(rng, n, dtype)
        if out_kind == "contig":
            self.out = np.empty(n, dtype=dtype)
        elif out_kind == "strided":
            self.out = np.empty(2 * n, dtype=dtype)[::2]
        else:
            self.out = None
        mkl_fft.fft(self.x, out=self.out)

    def time_fft(self, n, dtype, out_kind):
        mkl_fft.fft(self.x, out=self.out)
