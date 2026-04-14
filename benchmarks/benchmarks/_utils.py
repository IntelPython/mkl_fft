"""Shared utilities for mkl_fft benchmarks."""

import numpy as np


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
