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

The suites above vary transform size, shape, and dtype while calling with
default arguments on a freshly allocated C-contiguous array. The suites below
hold the transform fixed and vary how it is *requested*, which selects between
backend dispatch paths that differ by orders of magnitude in cost:

| File | Varies | Why it matters |
|------|--------|----------------|
| `bench_axes.py` | `axes=` subsets; `axis=` position | A strict subset of axes is dispatched slice by slice over the complementary axes, so cost scales with slice count rather than transform size. A 1-D transform of a rank > 2 array is batched into one `DftiCompute` only when the axis is first or last. |
| `bench_args.py` | `norm=`, `s=`, `n=` | `norm` other than `None`/`"backward"` computes a scale factor in Python; an `s` that pads or truncates leaves the direct N-D path for `_iter_fftnd`. |
| `bench_layout.py` | input layout (C/F/strided), `out=` layout | A non-contiguous input is not one segment, so for rank > 2 the backend iterates one vector at a time. OneMKL requires `out` to have the same element strides as the input; otherwise `_pydfti` allocates a temporary and copies. |
| `bench_descriptor.py` | call sequence; per-call floor | The thread-local descriptor cache holds a single entry, so interleaving lengths, dtypes, or domains rebuilds it on every call. |

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

For **1-D** transforms, `_pydfti` keeps one DFTI descriptor in thread-local
storage and reuses it when the next call matches on rank, precision, domain,
and length. To avoid charging that one-time build to the first measured
iteration, each benchmark's `setup` performs an explicit warmup call after
preparing the input array. ASV's default `warmup_time` (0.1s) already
amortizes this for sub-millisecond transforms, but the explicit warmup makes
the intent visible.

Two limits of that cache are worth knowing when reading results:

- It holds a **single** descriptor, so a call sequence that alternates length,
  dtype, or forward domain rebuilds it every time rather than reusing it. Only
  `bench_descriptor.py` exercises that; every other suite repeats one shape and
  so always hits.
- The **N-D** entry points in `mkl_fft/src/mklfft.c.src` take no cache
  argument at all and build, commit, and free a descriptor on every call. The
  warmup in the `fft2`/`fftn` suites therefore warms MKL's own internal state
  but does not prime a descriptor for reuse.

### Paired controls

The `bench_descriptor.py` suites are written as pairs: an alternating call
sequence alongside a control that issues the same number of transforms with
unchanging parameters. Neither number means much alone — the quantity of
interest is the difference. `time_switch_direction` is a deliberate negative
control: forward and backward scales live on the same descriptor, so it should
sit close to `time_repeat`.

Several suites here also carry an in-suite control among their parameters:
the full-dimensional `axes` tuple in `bench_axes.py`, `norm=None` in
`bench_args.py`, and `out_kind="none"` in `bench_layout.py`.

## Running Benchmarks

Prerequisites:

```bash
pip install ".[benchmark]"
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
