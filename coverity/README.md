# Triaging Coverity Scan findings

Static analysis runs on [Coverity Scan](https://scan.coverity.com) via
[`.github/workflows/coverity.yml`](../.github/workflows/coverity.yml) (weekly + on
demand). Analysis runs on Black Duck's servers; triage is done in the Scan web UI.

Every outstanding finding is a false positive — either in Cython-generated
boilerplate in `_pydfti.c`, or in a handful of branches in our own `.pyx` that are
dead-by-construction (Cython already guarantees the guard, or the branch is an
intentional placeholder). This guide records the verified findings and how to keep
triage from resetting.

## Where findings come from

`cov-build` captures two C translation units (the `.h` files under
`mkl_fft/src/` are `#include`d into them, not compiled on their own):

- **Template-generated** `mklfft.c` (from `mkl_fft/src/mklfft.c.src` via
  `_vendored/process_src_template.py`, wired up in `meson.build`). This is *not*
  Cython — it is our oneMKL DFTI descriptor/compute logic, just type-specialized
  for float32/float64/complex64/complex128. It pulls in the hand-written helpers
  in `mkl_fft/src/multi_iter.h` and `mkl_fft/src/mklfft.h`. A real bug can surface
  here, so treat it like hand-written code, **not** boilerplate — review every
  finding.
- **Cython-generated** `_pydfti.c` (from `mkl_fft/_pydfti.pyx`): `__Pyx_*` /
  `__pyx_pw_*` / `__pyx_tp_*` helpers and wrappers are boilerplate — findings here
  are ~always false positives. `__pyx_pf_*` functions are the C translation of our
  `.pyx` bodies; a real `.pyx` bug could surface there, so keep them in scope
  (Coverity can't see Python-level invariants — see the `_allocate_result`
  OVERRUN fixed in [gh-364](https://github.com/IntelPython/mkl_fft/pull/364)).

## Keeping triage durable: the Cython pin

A Cython *version* bump regenerates `_pydfti.c` wholesale, which churns the
Coverity CIDs and silently drops their triage — the same boilerplate then returns
under new CIDs. So **Cython is pinned in `coverity.yml`** (not `pyproject.toml`,
so shipped wheels are unaffected). The pin works only because the build runs with
`--no-build-isolation`; bumping it means re-triaging the boilerplate.

## Reducing the noise: a Project Component

**Project Settings → Components** buckets defects by a path regex. Define one to
group (not hide) the Cython unit so it can be filtered out of view — path-based,
so it survives regeneration:

- **Name:** `Generated-code`  **Path regex:** `.*_pydfti\.c`

Do **not** group `mklfft.c` (or the `mkl_fft/src/*.h` helpers) — that is our DFTI
logic, not boilerplate. Group only — do **not** mark it *ignored*, as that also
drops the `__pyx_pf_*` bodies (see [declined](#evaluated-and-declined)).

## Review checklist

Don't blanket-ignore the generated file — prioritise instead:

1. **Findings in `mklfft.c` and `mkl_fft/src/*.h`** — review every one; this is
   our type-specialized oneMKL DFTI code and its inlined iterator/cache helpers,
   not boilerplate.
2. **High/Medium findings in `__pyx_pf_*`** — verify against `_pydfti.pyx`; if it's
   a Python-level invariant Coverity can't see, mark `False Positive` with a
   reason. If it's a genuine defect, fix the `.pyx` (as with the `_allocate_result`
   OVERRUN).
3. **Known false-positive families below** — carry the recorded disposition;
   match on **checker + mechanism**, not CID (CIDs reset on a Cython bump or an
   engine upgrade).

## Known false-positive families

Match on **checker + mechanism**, not CID — CIDs reset on a Cython bump or engine
upgrade. Helper names are from Cython 3.3.0 and vary between versions. All Minor
severity, no runtime or security impact. Every family below is triaged
**Intentional / Ignore**.

### Cython-generated boilerplate (in `_pydfti.c` `__Pyx_*` / `__pyx_tp_*` helpers)

| Family | Checker | Why it's a false positive |
| --- | --- | --- |
| `tp_traverse` slots — `__Pyx_Coroutine_traverse`, `__Pyx_CyFunction_traverse`, and the genexpr/closure scope traversals for `_genexpr` and `_get_element_strides` (one mechanism) | DEADCODE | Cython emits a uniform base-type traversal preamble `e = __Pyx_call_type_traverse(o, 1, v, a); if (e) return e;`. The traversed object derives from a base whose traverse contributes nothing, so the helper returns 0 and the early-return is dead. The preamble *is* needed for objects deriving from a GC type. |
| `__Pyx_VectorcallBuilder_AddArg`, `__Pyx_PyCode_New`, `__Pyx_ParseKeywordDict` | DEADCODE | Dead branches in version/ABI-guarded runtime helpers — arms selected out by the CPython version the wheel is built against, or by an argument shape the call site never produces. |
| `_c2r_fft1d_impl` reference-cleanup epilogue | UNUSED_VALUE | Cython's generated `finally`/decref epilogue assigns a temp (e.g. resets a borrowed slot to `NULL`) that is never read again before the function returns. |
| `__Pyx_Generator_Replace_StopIteration` | CHECKED_RETURN | Boilerplate deliberately drops the return value of a call whose result is not needed on that path. |

### Our `.pyx`, dead-by-construction (in `__pyx_pf_*` bodies)

| Family | Checker | Why it's a false positive |
| --- | --- | --- |
| Redundant `is NULL` guard after an array-returning call — `_process_arguments` and `_direct_fftnd` | DEADCODE | The `.pyx` has an explicit `if <array> is NULL:` guard, but because the variable is typed `cnp.ndarray`, Cython already emits its own NULL/error check right after the returning call (e.g. `PyArray_CheckFromAny`). By the time our guard runs the value is provably non-NULL, so our guard body is dead. Harmless defensive code — kept for readability. |
| `_c2r_fft1d_impl` in-place stub | DEADCODE | `in_place = 0` is assigned immediately before `if in_place:`, so the `# TODO: Provide in-place functionality` branch is provably unreachable. Intentional placeholder for future in-place `irfft` support. |

## Evaluated and declined

- **Modeling files** correct the behavior of *called* functions; our FPs are
  intraprocedural (dead branches, compile-time-constant guards, `#if`), which
  models can't reach.
- **Dropping the generated unit** (hard exclude), e.g. after `cov-build`:
  ```bash
  cov-manage-emit --dir cov-int --tu-pattern "file('.*_pydfti\\.c')" delete
  ```
  Also drops the `__pyx_pf_*` bodies, so it's disabled in favour of the checklist.
