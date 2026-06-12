"""ASV benchmarks for mkl_fft"""

import os

import psutil

from ._patch_setup import _apply_patches

_MIN_THREADS = 4  # minimum physical cores required for multi-threaded mode


def _physical_cores():
    """Return physical core count; fall back to 1 (conservative)."""
    return psutil.cpu_count(logical=False) or 1


def _thread_count():
    physical = _physical_cores()
    return str(_MIN_THREADS) if physical >= _MIN_THREADS else "1"


_THREADS = os.environ.get("MKL_NUM_THREADS", _thread_count())
os.environ["MKL_NUM_THREADS"] = _THREADS

_apply_patches()
del _apply_patches
