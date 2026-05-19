# mkl_fft ASV Benchmarks

Performance benchmarks for [mkl_fft](https://github.com/IntelPython/mkl_fft) using
[Airspeed Velocity (ASV)](https://asv.readthedocs.io/en/stable/).

### Coverage

| File | API | Transforms | Dtypes | Sizes/Shapes |
|------|-----|-----------|--------|-------------|
| `bench_fft1d.py` | `mkl_fft` | `fft`, `ifft`, `rfft`, `irfft` | float32, float64, complex64, complex128 | power-of-two and non-power-of-two |
| `bench_fftnd.py` | `mkl_fft` | `fft2`, `ifft2`, `rfft2`, `irfft2`, `fftn`, `ifftn`, `rfftn`, `irfftn` | float32, float64, complex64, complex128 | square and non-square/non-cubic |
| `bench_numpy_fft.py` | `mkl_fft.interfaces.numpy_fft` | All exported functions including Hermitian (`hfft`, `ihfft`) | float32, float64, complex64, complex128 | power-of-two |
| `bench_scipy_fft.py` | `mkl_fft.interfaces.scipy_fft` | All exported functions including Hermitian 2-D/N-D (`hfft2`, `hfftn`) | float32, float64, complex64, complex128 | square and cubic |
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
regardless of their total core count. Physical cores are read from
`/proc/cpuinfo` — hyperthreads are excluded per MKL recommendation.
