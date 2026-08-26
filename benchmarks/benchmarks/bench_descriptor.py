"""Benchmarks for DFTI descriptor lifecycle and call patterns.

Every other suite in this directory calls one transform shape repeatedly, so
the thread-local DFTI descriptor cache in ``_pydfti`` always hits after
warmup. Real callers interleave transforms of different length, dtype, or
domain. The cache holds a single descriptor, so each switch frees the cached
one and builds and commits a replacement.

Each suite here pairs an alternating call sequence with a same-parameter
control that issues the same number of transforms, so the switch cost is
readable as the difference between the two.

``BenchFixedCost`` measures the per-call floor at sizes where the transform
itself is negligible. The N-D entry points in ``mklfft.c.src`` take no cache
argument and build a descriptor on every call, so the N-D floor is expected to
sit above the 1-D one.
"""

import numpy as np

import mkl_fft

from ._utils import _RNG_SEED, _make_input

# Offset used to build a second, differently sized input. Small enough that
# both lengths factor similarly, so the difference measured is descriptor
# rebuild rather than a change of MKL algorithm class.
_SIZE_DELTA = 24


# ---------------------------------------------------------------------------
# 1-D descriptor switching
# ---------------------------------------------------------------------------


class BenchDescriptorSwitch1D:
    """Alternating vs repeated transform parameters, 1-D.

    ``time_repeat`` is the control for ``time_switch_size`` and
    ``time_switch_dtype``. ``time_switch_domain`` is compared against
    ``time_repeat_domain``; those two are not an exact control pair, because
    rfft and fft do not cost the same, but at the smaller ``n`` the descriptor
    rebuild dominates that difference.
    """

    params = [[1024, 65536]]
    param_names = ["n"]

    def setup(self, n):
        rng = np.random.default_rng(_RNG_SEED)
        self.a = _make_input(rng, n, "complex128")
        self.b = _make_input(rng, n - _SIZE_DELTA, "complex128")
        self.a32 = self.a.astype("complex64")
        self.real = _make_input(rng, n, "float64")
        mkl_fft.fft(self.a)
        mkl_fft.fft(self.b)
        mkl_fft.fft(self.a32)
        mkl_fft.rfft(self.real)

    def time_repeat(self, n):
        mkl_fft.fft(self.a)
        mkl_fft.fft(self.a)

    def time_switch_size(self, n):
        mkl_fft.fft(self.a)
        mkl_fft.fft(self.b)

    def time_switch_dtype(self, n):
        mkl_fft.fft(self.a)
        mkl_fft.fft(self.a32)

    def time_repeat_domain(self, n):
        mkl_fft.rfft(self.real)
        mkl_fft.rfft(self.real)

    def time_switch_domain(self, n):
        mkl_fft.rfft(self.real)
        mkl_fft.fft(self.a)

    def time_switch_direction(self, n):
        mkl_fft.fft(self.a)
        mkl_fft.ifft(self.a)


# ---------------------------------------------------------------------------
# Descriptor switching driven by axis choice
# ---------------------------------------------------------------------------


class BenchDescriptorSwitchAxis:
    """Alternating the transformed axis of a 2-D array.

    This is the call pattern ``_fft_utils._iter_fftnd`` generates internally,
    and that 2-D user code generates directly.

    The two axes of a C-contiguous array do not cost the same — axis 0 walks
    the long stride — so ``time_switch_axis`` cannot be read against a single
    control. Both per-axis controls are therefore reported, and the descriptor
    contribution is ``switch - (repeat_axis0 + repeat_axis1) / 2``. On a
    non-square array the transform length changes with the axis, so the cached
    descriptor is discarded on every call; on a square array it survives.
    """

    params = [[(512, 512), (512, 256)]]
    param_names = ["shape"]

    def setup(self, shape):
        rng = np.random.default_rng(_RNG_SEED)
        self.x = _make_input(rng, shape, "complex128")
        mkl_fft.fft(self.x, axis=0)
        mkl_fft.fft(self.x, axis=1)

    def time_repeat_axis0(self, shape):
        mkl_fft.fft(self.x, axis=0)
        mkl_fft.fft(self.x, axis=0)

    def time_repeat_axis1(self, shape):
        mkl_fft.fft(self.x, axis=1)
        mkl_fft.fft(self.x, axis=1)

    def time_switch_axis(self, shape):
        mkl_fft.fft(self.x, axis=1)
        mkl_fft.fft(self.x, axis=0)


# ---------------------------------------------------------------------------
# Per-call fixed cost
# ---------------------------------------------------------------------------


class BenchFixedCost:
    """Per-call cost at sizes where the transform itself is negligible.

    Anything these report above a few hundred nanoseconds is argument
    processing, dispatch, allocation, and descriptor setup rather than
    arithmetic.
    """

    params = [["float64", "complex128"]]
    param_names = ["dtype"]

    def setup(self, dtype):
        rng = np.random.default_rng(_RNG_SEED)
        self.x1 = _make_input(rng, 8, dtype)
        self.x2 = _make_input(rng, (4, 4), dtype)
        self.x3 = _make_input(rng, (4, 4, 4), dtype)
        self.real1 = _make_input(rng, 8, "float64")
        mkl_fft.fft(self.x1)
        mkl_fft.fft2(self.x2)
        mkl_fft.fftn(self.x3)
        mkl_fft.rfft(self.real1)

    def time_fft_1d(self, dtype):
        mkl_fft.fft(self.x1)

    def time_fft2_2d(self, dtype):
        mkl_fft.fft2(self.x2)

    def time_fftn_3d(self, dtype):
        mkl_fft.fftn(self.x3)

    def time_rfft_1d(self, dtype):
        mkl_fft.rfft(self.real1)
