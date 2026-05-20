"""ASV benchmarks for mkl_fft.

Thread control — design rationale
----------------------------------
Since we do not have a dedicated CI benchmark machine, benchmarks run on a shared CI pool
whose machines vary in core count over time.
Using the full physical core count of each machine would make results
incomparable across runs on different machines.

Strategy:
  - Physical cores >= 4  →  fix MKL_NUM_THREADS = 4
      4 is the lowest common denominator that guarantees multi-threaded MKL
      behavior and is achievable on any modern CI machine.  Results from
      different machines in the pool are therefore directly comparable.
  - Physical cores < 4   →  fall back to MKL_NUM_THREADS = 1 (single-threaded)
      Prevents over-subscription on under-resourced machines and avoids
      misleading comparisons against 4-thread baselines.

MKL recommendation: use physical cores, not logical (hyperthreaded) CPUs.
"""

import os

import psutil

_MIN_THREADS = 4  # minimum physical cores required for multi-threaded mode


def _physical_cores():
    """Return physical core count; fall back to 1 (conservative)."""
    return psutil.cpu_count(logical=False) or 1


def _thread_count():
    physical = _physical_cores()
    return str(_MIN_THREADS) if physical >= _MIN_THREADS else "1"


_THREADS = os.environ.get("MKL_NUM_THREADS", _thread_count())
os.environ["MKL_NUM_THREADS"] = _THREADS
