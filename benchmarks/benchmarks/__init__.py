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
import re

_MIN_THREADS = 4  # minimum physical cores required for multi-threaded mode


def _physical_cores():
    """Return physical core count from /proc/cpuinfo; fall back to os.cpu_count()."""
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
        cpu_cores = int(re.search(r"cpu cores\s*:\s*(\d+)", content).group(1))
        sockets = max(
            len(set(re.findall(r"physical id\s*:\s*(\d+)", content))), 1
        )
        return cpu_cores * sockets
    except Exception:
        return os.cpu_count() or 1


def _thread_count():
    physical = _physical_cores()
    return str(_MIN_THREADS) if physical >= _MIN_THREADS else "1"


_THREADS = os.environ.get("MKL_NUM_THREADS", _thread_count())
os.environ["MKL_NUM_THREADS"] = _THREADS
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)
