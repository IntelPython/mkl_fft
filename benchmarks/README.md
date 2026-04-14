# mkl_fft ASV Benchmarks

Performance benchmarks for [mkl_fft](https://github.com/IntelPython/mkl_fft) using
[Airspeed Velocity (ASV)](https://asv.readthedocs.io/en/stable/).

## Structure

```
benchmarks/
├── asv.conf.json          # ASV configuration (CI-only, no env/build settings)
└── benchmarks/
    ├── __init__.py        # Thread pinning (MKL_NUM_THREADS)
    ├── bench_fft1d.py     # mkl_fft root API — 1-D transforms
    ├── bench_fftnd.py     # mkl_fft root API — 2-D and N-D transforms
    ├── bench_numpy_fft.py # mkl_fft.interfaces.numpy_fft — full coverage
    ├── bench_scipy_fft.py # mkl_fft.interfaces.scipy_fft — full coverage
    └── bench_memory.py    # Peak RSS memory benchmarks
```

### Coverage

| File | API | Transforms |
|------|-----|-----------|
| `bench_fft1d.py` | `mkl_fft` | `fft`, `ifft`, `rfft`, `irfft` — power-of-two and non-power-of-two |
| `bench_fftnd.py` | `mkl_fft` | `fft2`, `ifft2`, `rfft2`, `irfft2`, `fftn`, `ifftn`, `rfftn`, `irfftn` |
| `bench_numpy_fft.py` | `mkl_fft.interfaces.numpy_fft` | All exported functions including Hermitian (`hfft`, `ihfft`) |
| `bench_scipy_fft.py` | `mkl_fft.interfaces.scipy_fft` | All exported functions including Hermitian 2-D/N-D (`hfft2`, `hfftn`) |
| `bench_memory.py` | `mkl_fft` | Peak RSS for 1-D, 2-D, and 3-D transforms |

Benchmarks cover float32, float64, complex64, complex128 dtypes, power-of-two
and non-power-of-two sizes, square and non-square/non-cubic shapes.

## Threading

`__init__.py` pins `MKL_NUM_THREADS` to **4** when the machine has 4 or more
physical cores, or falls back to **1** (single-threaded) otherwise. This keeps
results comparable across CI machines in the shared pool regardless of their
total core count. Physical cores are read from `/proc/cpuinfo` — hyperthreads
are excluded per MKL recommendation.

Override by setting `MKL_NUM_THREADS` in the environment before running ASV.

## Running Locally

> Benchmarks are designed for CI. Local runs require mkl_fft to be installed
> in the active Python environment.

```bash
cd benchmarks/

# Quick smoke-run against the current working tree (no env management)
asv run --python=same --quick --show-stderr HEAD^!

# Run a specific benchmark file
asv run --python=same --quick --bench bench_fft1d HEAD^!

# View and publish results
asv publish          # generates .asv/html/
asv preview          # serves at http://localhost:8080
```

## CI

Benchmarks run automatically in Jenkins on the `auto-bench` node via
`benchmarkHelper.performanceTest()` from the shared library. The pipeline uses:

```bash
asv run --environment existing:<python> --set-commit-hash $COMMIT_SHA
```

This bypasses ASV environment management entirely — mkl_fft is pre-installed
into a conda environment by the pipeline before ASV is invoked.

- **Nightly (prod):** results are published to the benchmark dashboard
- **PR (dev):** `asv compare` output is evaluated for regressions; a 30% slowdown
  triggers a failed GitHub commit status

Results are stored in the `mkl_fft-results` branch of
`intel-innersource/libraries.python.intel.infrastructure.benchmark-dashboards`.
