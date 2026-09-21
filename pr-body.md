## Track

Performance (Track A), item 2 on the ranked list: an avoidable copy in
`_fft_utils.py` / interface layer — here `mkl_fft/interfaces/_scipy_fft.py`.

## What I examined

I read the root `AGENTS.md`, `.github/copilot-instructions.md`, and the nested
`AGENTS.md` files for `mkl_fft/`, `mkl_fft/interfaces/`, `mkl_fft/src/`, and
`mkl_fft/tests/` before touching anything. I then read `_pydfti.pyx`,
`_fft_utils.py`, `_mkl_fft.py`, `src/mklfft.c.src`, and the three interface
adapter files (`_numpy_fft.py`, `_scipy_fft.py`, `_float_utils.py`,
`_numpy_helper.py`) end to end, looking first for descriptor-handling or
memory-management bugs per the bug-fix priority list, then for avoidable
copies and per-call overhead per the performance priority list.

I specifically checked the `out=x` aliasing path in `_c2c_fft1d_impl`
(`_pydfti.pyx`) against the MKL descriptor being configured
`DFTI_PLACEMENT = DFTI_NOT_INPLACE` in the `*_out` C routines
(`src/mklfft.c.src`), since passing identical input/output pointers to a
not-in-place descriptor looked suspicious. It turned out `mkl_fft/tests/
test_fft1d.py` (`test_vector5`, `test_vector6`, `test_matrix4`) already
exercises exactly this case as a documented, presumably-passing feature
("fft in-place is the same as fft out-of-place"), so I did not treat this as
a new finding and left it alone — reopening settled, tested behaviour without
being able to run the suite myself would be speculative.

## What I found

In `mkl_fft/interfaces/_scipy_fft.py`, `hfft` and `hfftn` each computed the
Hermitian-conjugate of their input as two full passes over the array:

```python
x = np.array(x, copy=True)   # pass 1: copy every element
np.conjugate(x, out=x)        # pass 2: negate every imaginary part in place
```

`x` at this point has already been through `_validate_input`, which calls
`np.asarray(x)` (via `_supported_array_or_not_implemented`), so it is always
a plain `ndarray` (subclasses are already stripped, `subok=False` is
`np.asarray`'s default) — the explicit copy exists solely so that the
in-place `conjugate` does not mutate the caller's own array, not for any
type normalization. `np.conjugate(x)` called without `out=` already
allocates a fresh output array and computes the conjugate directly from the
input in a single pass, giving the identical result without the caller's
array ever being touched. The two-step form is therefore a copy taken
defensively where a single elementwise op already does the job — exactly the
"avoidable copies" pattern called out as the second-ranked optimization
target in the mandate. The sibling implementation `_numpy_fft.hfft` already
uses the single-call form (`np.conjugate(x)`), so this is also a small
internal inconsistency between the two interface adapters.

## The change

In both `hfft` and `hfftn` in `mkl_fft/interfaces/_scipy_fft.py`, replaced:

```python
x = np.array(x, copy=True)
np.conjugate(x, out=x)
```

with:

```python
x = np.conjugate(x)
```

`ihfft`/`ihfftn` were not touched: they conjugate the *output* of `rfft`/
`rfftn` in place, and that output buffer was just freshly allocated by
`mkl_fft.rfft`/`rfftn` for this call, so there is no caller-owned buffer at
risk and no redundant copy to remove — the one-pass in-place conjugate there
is already optimal.

## Why this is behaviour-preserving

- **Correctness of the result:** `np.conjugate(a)` and
  `(lambda b: (np.conjugate(b, out=b), b)[1])(np.array(a, copy=True))`
  compute the same values element-by-element; the only difference is whether
  the negation of the imaginary part happens into a copy-then-mutate buffer
  or straight into a freshly allocated one.
- **dtype:** unaffected — conjugate preserves the input dtype in both forms
  (real dtypes are a no-op copy, complex dtypes negate the imaginary part);
  `x` here is always a plain array of a dtype `_validate_input` already
  approved (not `float16`/`float128`/`complex256`).
- **Shape:** unaffected — `conjugate` is elementwise and shape-preserving in
  both forms.
- **Strides / memory layout:** both `np.array(x, copy=True)` and the
  `conjugate` ufunc default to `order='K'` (preserve layout as far as
  possible), so the freshly allocated array has the same layout
  characteristics the old two-step form produced.
- **Aliasing:** the old code copied specifically so the in-place conjugate
  would not mutate the caller's array. `np.conjugate(x)` without `out=`
  never writes into `x`, so the caller's array is equally untouched under
  the new code — the safety property the copy existed for is preserved by a
  different, cheaper mechanism.
- **norm/out parameters:** untouched by this diff; `hfft`/`hfftn` do not
  accept an `out=` argument to `mkl_fft.irfft`/`irfftn` in this file (`# Note:
  overwrite_x is not utilized`), so there is no interaction with an
  output buffer supplied by the caller.

## Mechanism, one sentence

Two full sequential passes over the array (allocate-and-copy, then negate
imaginary parts in place) are replaced by one pass that allocates and
negates at the same time, halving the memory traffic per element for every
call to `scipy.fft.hfft`/`hfftn` routed through `mkl_fft.interfaces`.

## Verification

I have no shell access in this environment, so I could not run `pytest`,
`black`, `isort`, `flake8`, `pylint`, or `cython-lint` myself. I re-read the
edited file in full after the change (shown above) to confirm the two edits
are the only difference from the original and that indentation/imports are
unchanged; `numpy` was already imported at module scope, so no import
changes were needed. I did not add a new test because this is a
performance-only change with no behavioural difference to assert beyond what
the existing `scipy.fft` compatibility tests already cover (`mkl_fft/tests/
third_party/scipy/test_basic.py` exercises `hfft`/`hfftn` through
`mkl_fft.interfaces.scipy_fft`); CI's existing suite is the check that the
result is still numerically identical, and CI's ASV benchmark is the
authority on whether it is faster.

Commands run: none (no shell in this environment). Local measurement: none
taken; this report states the mechanism (one memory pass instead of two) as
the basis for the hypothesis, per the mandate, and leaves confirmation to
CI's benchmark job.

## Rejected candidates

- Relaxing the exact-stride-match requirement before reusing a caller's
  `out=` array in `_c2c_fft1d_impl` (`_pydfti.pyx`), which is marked `TODO`
  in the source. This would remove a real defensive copy for a wider range
  of strided `out=` arrays, but the comment already flags it as needing
  careful validation of what MKL actually tolerates for input/output stride
  relationships, and getting it wrong risks silently wrong FFT output. Left
  as a candidate for a human with the ability to run the MKL-backed test
  suite.
- Reordering the dtype checks in `_downcast_float128_array`
  (`interfaces/_float_utils.py`) to test `isinstance(x, np.ndarray)` before
  the `longdouble`/`clongdouble` comparisons, to skip redundant work for the
  common plain-`ndarray` case. The saving is a couple of dtype equality
  checks per call — real but small enough that I could not state a clear
  single-sentence mechanism distinguishing it from noise, so I left it alone
  per "if you cannot say why a change is faster in one sentence, it is not a
  candidate."
- Treating the `out=x` (aliasing) path through a `DFTI_NOT_INPLACE`
  descriptor in `_pydfti.pyx`/`src/mklfft.c.src` as a bug. Rejected because
  existing tests (`test_fft1d.py::Test_mklfft_vector::test_vector5/6`,
  `Test_mklfft_matrix::test_matrix4`) already assert this exact case is
  correct and treat it as a supported feature; reopening it without being
  able to run those tests would be guessing against settled, tested
  behaviour.

## Left to humans

- The stride-relaxation `TODO`s in `_pydfti.pyx` (`_c2c_fft1d_impl`,
  `_r2c_fft1d_impl`, `_c2r_fft1d_impl`) for reusing a caller's `out=` array
  under a wider set of compatible strides than exact match / both-contiguous.
  This is the higher-ranked "descriptor handling" and "avoidable copies"
  opportunity in the mandate, but validating exactly what MKL DFTI accepts
  for mismatched input/output strides needs to be checked against MKL
  documentation and run against the real test suite, which I cannot do here.
- Any compiler/build flag changes (explicitly out of scope for me to
  implement per the mandate; I did not find a specific one to propose this
  run).
- `CHANGELOG.md` wording: not proposed, because this run made no
  user-visible behaviour change (performance-only), and the mandate only
  asks for changelog wording for bug fixes.
