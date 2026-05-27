"""Shared utilities for mkl_fft benchmarks."""

import numpy as np

_RNG_SEED = 42


def _make_input(rng, shape, dtype):
    """Return an array of *shape* and *dtype*.

    Complex dtypes get non-zero imaginary parts for a realistic signal.
    `shape` may be an int (1-D) or a tuple.
    """
    dt = np.dtype(dtype)
    s = (shape,) if isinstance(shape, int) else shape
    if dt.kind == "c":
        return (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(dt)
    return rng.standard_normal(s).astype(dt)


class BenchC2C:
    """Base setup for complex-to-complex benchmarks.

    Subclasses define params, param_names, and time_* / peakmem_* methods.
    Other positional params are ignored.
    """

    def setup(self, shape, dtype, *_):
        rng = np.random.default_rng(_RNG_SEED)
        self.x = _make_input(rng, shape, dtype)


# dtype axes
_DTYPES_ALL = ["float32", "float64", "complex64", "complex128"]
_DTYPES_REAL = ["float32", "float64"]
_DTYPES_REDUCED = ["float64", "complex128"]

# shape/size axes shared across multiple files
_SHAPES_2D = [(64, 64), (128, 128), (256, 256), (512, 512)]
_SHAPES_2D_IFACE = [(64, 64), (256, 256), (512, 512)]
_SHAPES_3D = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]


class BenchR2C:
    """Base setup for real-to-complex / complex-to-real and Hermitian benchmarks.

    Prepares:
      self.x_real    — real array of full shape (rfft / ihfft input)
      self.x_complex — complex half-spectrum array (irfft / hfft input)

    DC (index 0 of the last axis) of x_complex has its imaginary part zeroed,
    and when the full last-axis length is even the Nyquist bin imaginary part
    is also zeroed, satisfying Hermitian symmetry expected by hfft / ihfft2 /
    hfftn. Extra positional params are accepted and ignored.
    """

    def setup(self, shape, dtype, *_):
        rng = np.random.default_rng(_RNG_SEED)
        cdtype = "complex64" if dtype == "float32" else "complex128"
        if isinstance(shape, int):
            n_last = shape
            half_shape = shape // 2 + 1
        else:
            n_last = shape[-1]
            half_shape = shape[:-1] + (shape[-1] // 2 + 1,)
        self.x_real = rng.standard_normal(shape).astype(dtype)
        self.x_complex = (
            rng.standard_normal(half_shape)
            + 1j * rng.standard_normal(half_shape)
        ).astype(cdtype)
        self.x_complex[..., 0] = self.x_complex[..., 0].real
        if n_last % 2 == 0:
            self.x_complex[..., -1] = self.x_complex[..., -1].real
