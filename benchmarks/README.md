# mkl_fft ASV Benchmarks

Performance benchmarks for [mkl_fft](https://github.com/IntelPython/mkl_fft) using
[Airspeed Velocity (ASV)](https://asv.readthedocs.io/en/stable/).

### Coverage

| File | API | Transforms | Dtypes | Sizes/Shapes |
|------|-----|-----------|--------|-------------|
| `bench_fft1d.py` | `mkl_fft` | `fft`, `ifft`, `rfft`, `irfft` | float32, float64, complex64, complex128 | power-of-two and non-power-of-two |
| `bench_fftnd.py` | `mkl_fft` | `fft2`, `ifft2`, `rfft2`, `irfft2`, `fftn`, `ifftn`, `rfftn`, `irfftn` | float32, float64, complex64, complex128 | square and non-square/non-cubic |
| `bench_interfaces.py` | `mkl_fft.interfaces.{numpy_fft, scipy_fft}` | All exported functions; selected by a `module` parameter. Hermitian 2-D/N-D (`hfft2`, `hfftn`) are scipy-only. | float32, float64, complex64, complex128 | power-of-two and cubic |
| `bench_memory.py` | `mkl_fft` | Peak RSS for 1-D, 2-D, and 3-D transforms | float32, float64, complex128 | power-of-two |

## Threading

Set `MKL_NUM_THREADS` in the environment before running ASV to control the
thread count used by MKL:

```bash
MKL_NUM_THREADS=8 asv run --python=same --quick HEAD^!
```

If `MKL_NUM_THREADS` is not set, `__init__.py` applies a default: **4** threads
when the machine has 4 or more physical cores, or **1** (single-threaded)
otherwise. This keeps results comparable across CI machines in the shared pool
regardless of their total core count. Physical cores are detected via
`psutil.cpu_count(logical=False)` — hyperthreads are excluded per MKL
recommendation.

## Notes on Measurement

### DFTI descriptor warmup

MKL creates a DFTI descriptor on the first FFT call for a given (size, dtype,
strides) combination and reuses it on subsequent calls. To avoid charging
that one-time cost to the first measured iteration, each benchmark's `setup`
performs an explicit warmup call after preparing the input array. ASV's
default `warmup_time` (0.1s) already amortizes this for sub-millisecond
transforms, but the explicit warmup makes the intent visible.

## Running Benchmarks

Prerequisites:

```bash
pip install asv psutil
```

Run benchmarks against the current environment:

```bash
asv run --python=same --quick HEAD^!
```

Compare two commits:

```bash
asv continuous --python=same HEAD~1 HEAD
```

View results in a browser:

```bash
asv publish
asv preview
```
