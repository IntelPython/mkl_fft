# REVIEW.md - mkl_fft

Checks for every `mkl_fft` pull request, by human or automated reviewers. Layout and coding rules are in
AGENTS.md and CONTRIBUTING.md. Apply the sections for the files touched, plus Tests, Pull request text
and Automated pull requests when it applies. Each rule is a check: flag a failure with its section.

## How to verify
Say which of these you ran, and on what platform.

1. Build, test and lint with the commands in CONTRIBUTING.md. A fix's new test must fail on base.
2. For performance, run base against base (A/A) for noise, then base against the pull request, with the
   ASV modules in `benchmarks/` that exercise the changed path (see `benchmarks/AGENTS.md`).
3. Reference is NumPy's own FFT. Some NumPy builds come with `numpy.fft` already patched to `mkl_fft`
   (check `numpy.fft.fft.__module__`); then compare against NumPy's unpatched implementation.

## Correctness
- Same values, dtype, shape, result memory order (C/F) and errors as `numpy.fft`/`scipy.fft` for the
  same call; any difference is stated and justified in the pull request.
- The caller's input is never modified; nothing outside `out` is written. Whether oneMKL writes
  straight into `out=` is decided on the array handed to the backend (after casting and padding), not
  the caller's; getting it wrong gives wrong values and writes outside `out`.
- When a transform path, layout check or dtype dispatch changes, cover:
  - input and `out=` C-, F- or both-contiguous (`(m, 1)`), strided, negative-stride; rank 1, 2, 3+;
  - `s=`/`n=` shorter and longer than the input; `axes=` negative, repeated, empty;
  - every `norm`, with and without `out=` and `s=`;
  - float16, int, bool (cast); float32/64, complex64/128; longdouble if present;
  - zero-length and length-1 axes; NaN, inf, signed zero.
- "No behavior change" is proven for those cases by a test or recorded comparison, not prose.
- `MKL_INT` counts are checked on the full product of sizes, not one dimension.
- `workers` follows SciPy (negative counts back from the max, `scipy.fft.set_workers` honored),
  capped by oneMKL's maximum thread count, not `os.cpu_count()`.

## Patching NumPy and SciPy
- Patch/unpatch are idempotent, reentrant, thread-safe: no shared mutable originals, repeated patching
  restores the true originals, one thread's unpatch keeps another's patch, `is_patched()` is truthful.
- Startup patching works in the documented cases (scripts too); a startup error cannot break Python.
- SciPy and `mkl-service` stay optional: imports that need them are conditional.

## Tests
- Every behavior change and fix has a test that fails on base and passes on the pull request.
- Expected values come from an independent reference (step 3) or by hand, never `mkl_fft` itself.
- Extend or parametrize existing tests rather than duplicate them; parametrize rather than skip inside.
- Deterministic, no timing asserts, copyright header on new files. A failure too large for CI comes
  with a manual run's command and output.
- The CI job meant to cover the change ran its tests (a job can pass with zero).

## Performance claims
- Before and after on one machine, with hardware, threads and pinning, Python/NumPy/oneMKL versions
  and the command (ASV preferred). Results must not change: review it as a behavior change too.
- Noise is measured (repeats or A/A); a gain inside it is no speedup. "Unchanged" needs numbers too.
- Read every table row; any slowdown is called out and justified, or fixed. Memory or "one copy
  fewer" claims are shown by the memory benchmarks or a test.
- Benchmark edits keep the DFTI warmup in `setup`, setup and imports out of timed code, portable core
  detection. Changing the thread default or setup breaks comparison with old results.

## C backend and Cython
- Edit `mkl_fft/src/*.c.src` templates, never generated C. Every allocation and DFTI call has a
  checked failure path: no leak, Python exception set.
- The GIL is released only around code touching no Python objects, alike in contiguous, strided, 1-D
  and N-D paths.
- Update asserts and comments whose invariant the change breaks. No bare `except:`. Reuse existing
  helpers rather than copying logic.

## Packaging and CI
- `conda-recipe*/meta.yaml` floors match `pyproject.toml`; NumPy compatibility comes from `run_exports`.
- macOS: one `-Wl,-rpath` per path. Free-threaded builds keep their classifier and pins. Any OpenMP or
  threading library linked is shipped by the recipe.
- Formatters skip `*.c.src` (they break `@name@` placeholders) and `_vendored/`. Actions are pinned
  to a full commit SHA with a matching version comment.

## Automated pull requests
Automated = opened by the repository's automation account, or marked as AI-generated in its
description. Unless the pull request shows otherwise, assume the agent did not build, test or measure:
nothing has run until CI. Maintainers often push commits and edit text later: check who
wrote each commit and claim, and review the current state. Everything above applies, plus:

- One change: one optimization or one fix. Flag agent commits outside `mkl_fft/` and `mkl_fft/tests/`
  (build files, recipes, workflows, benchmarks, generated C). A maintainer adds `CHANGELOG.md` entries.
- A fix needs a test failing on base. A performance change is only a mechanism until a benchmark
  shows the gain beyond noise with no slowdown elsewhere, and a test shows results unchanged.
- No weakened tests: no removed test, new `skip`/`xfail`, looser tolerance or changed expected value
  unless upstream NumPy/SciPy behavior justifies it and you verified that behavior yourself.
- Edits to agent rule files (`AGENTS.md`, `REVIEW.md` and other agent instruction files) change
  future runs' rules: review them as policy, separately from code.
- Known risk spots: the `out=` direct-write check, `MKL_INT` on one dimension, a slowdown in one row.
- Text: no agent narration. Every number says who measured it, where and how. Title and description
  match the final scope.

## Pull request text
- User-visible changes (behavior, API, packaging, build output, measured speedup) get a `CHANGELOG.md`
  entry in the format CONTRIBUTING.md gives, linking the pull request and each fixed issue. Docs,
  tooling and CI-only changes usually need none.
- The description follows the template: what changed, why, what was verified and where, what was not.
  No agent scaffolding (prompts, "mandates", notes about reading instruction files).
