# REVIEW.md - mkl_fft

What a reviewer of an `mkl_fft` pull request checks. Applies to human and automated reviewers alike.
Repository layout and coding rules are in [AGENTS.md](AGENTS.md); this file covers what a change must
prove before it is merged.

## Correctness

- NumPy and SciPy FFT behavior is the reference. A change must give the same values, dtype, shape and
  errors as `numpy.fft` / `scipy.fft` for the same call, unless the pull request says otherwise and why.
- Any change to the decision whether oneMKL writes straight into the caller's `out=` or into a temporary
  must check the array actually handed to the backend (after casting and padding), not the one the
  caller passed. Getting this wrong gives wrong values or writes outside `out`.
- Check these cases whenever a transform path, a layout check or a dtype dispatch changes:
  - C-ordered, F-ordered, strided and negative-stride inputs and `out=`, rank 1, 2 and 3 or more;
  - `s=` shorter and longer than the input, `axes=` with negative and repeated values;
  - every `norm` with and without `out=` and `s=`;
  - float16, int and bool inputs that must be cast; float32/float64, complex64/complex128;
  - zero-length axes, a length-1 axis, NaN, inf and signed zero.
- A pull request that claims "no behavior change" shows it for the cases above, not in prose only.
- Integer counts passed to oneMKL (`MKL_INT`) are checked against the full product of the sizes involved,
  not one dimension.

- `workers` follows SciPy semantics (negative values count back from the maximum, `scipy.fft.set_workers`
  is honored) and is capped by oneMKL's own maximum thread count, not `os.cpu_count()`.

## Patching NumPy and SciPy

- Patch and unpatch are idempotent, reentrant and thread-safe: saved originals are not shared mutable
  class state, nested or repeated patching restores the true originals, and unpatching in one thread
  cannot drop another thread's patch.
- Startup patching works for the cases the docs show (for example scripts, not only interactive
  sessions), and code that runs at interpreter startup cannot break every Python process on error.
- SciPy and `mkl-service` stay optional: imports that need them are conditional, so installing
  `mkl_fft` does not force them on users who only need NumPy.

## Tests

- Every behavior change and every fix comes with a test that fails on the base branch and passes on the
  pull request.
- Tests compare against an independent reference (`numpy.fft` or a hand-computed result), never against
  `mkl_fft` itself or a patched `numpy.fft`.
- When a failure needs more memory or time than CI has, the pull request includes the command and the
  output of a manual run instead.
- Check that the CI job meant to cover the change ran its tests (a job can pass with zero tests).

## Performance claims

- A claim that a change is faster or uses less memory needs numbers: before and after, on the same
  machine, with the hardware, thread count, versions and the command used (see
  [benchmarks/AGENTS.md](benchmarks/AGENTS.md)). The ASV suite in `benchmarks/` is the preferred source.
- Results come with a noise estimate (repeat runs or an A/A comparison). Changes inside the noise are not
  a speedup.
- Every row of a results table is read, not only the headline: a slowdown in any case is called out and
  justified, or fixed.
- When CI publishes benchmark results for the commit, the claim is checked against them. A claim with no
  measured number behind it is flagged.
- Benchmark changes keep the DFTI warmup in `setup`, keep imports and setup out of timed code, and use
  portable core detection. Changing the thread default or setup invalidates comparison with older results.
- A performance change must not change results. Review it as a behavior change too.

## C backend and Cython

- Edit `mkl_fft/src/*.c.src` templates, not generated C.
- Every allocation and DFTI call has its failure path checked, with no leak and a Python exception set.
- The GIL is released only around code that touches no Python objects, consistently across the
  contiguous and strided paths.
- Avoid bare `except:`; catch the specific error.
- Reuse existing helpers (for example the argument handling in `_fft_utils.py`) rather than copying logic.

## Packaging and CI

- Dependency floors in `conda-recipe*/meta.yaml` agree with `pyproject.toml`; numpy compatibility comes
  from its `run_exports`, not from ignoring them.
- On macOS each runtime path is its own `-Wl,-rpath` argument.
- Free-threaded Python builds keep the classifier and build pins that support them.
- Any OpenMP or threading library the build links is one the recipe actually ships at run time.
- Formatters and linters exclude `*.c.src` templates and `_vendored/`; formatting a template breaks its
  `@name@` placeholders.
- GitHub Actions are pinned to a full commit SHA with a matching version comment.

## Automated pull requests

Pull requests opened by an automated agent (their description starts with "This pull request is entirely
AI-generated") get the full review above, plus these checks. The agent writes code from reading the
source and runs only the pre-commit hooks; it does not build, run tests or measure anything, so the code
has not run until CI has. Maintainers may later push commits or edit the description of such a pull
request, so check who wrote each commit and each claim.

- **One change.** The pull request does exactly one optimization or one bug fix, never both, and touches
  only `mkl_fft/` sources and `mkl_fft/tests/`. Changes to build files, recipes, workflows, benchmarks,
  `CHANGELOG.md` or generated C are out of scope and are flagged.
- **Fix needs a failing test.** A `fix:` pull request adds a test that fails on the base branch and
  passes with the fix. Confirm from CI that it fails without the fix; a test that also passes on the base
  branch is not a regression test.
- **Speedup needs a number.** A `perf:` pull request states a mechanism, not a measurement. Accept it
  only when the benchmark run attached to the pull request shows the gain beyond noise, with no slowdown
  in other cases. Without that run the change is not proven faster.
- **Behavior is unchanged.** A `perf:` change must keep results identical for every dtype, shape,
  stride, `out=`, `s=`, `axes=` and `norm`. Check the cases in [Correctness](#correctness) against the code.
- **Tests are not weakened.** No test function removed, no new `skip` or `xfail`, no loosened tolerance
  or changed expected value without the upstream NumPy/SciPy behavior that justifies it. Verify that
  upstream behavior yourself.
- **Rule files.** An edit to any `AGENTS.md`, `REVIEW.md` or `.github/copilot-instructions.md` changes
  the rules future automated runs follow. Review it as a policy change, separately from the code.
- **Known risk spots.** Past automated changes went wrong in the check deciding whether oneMKL writes
  into `out=` directly (wrong array checked: wrong values and writes outside `out`), in integer counts
  checked on one dimension only, and in benchmark tables hiding a slowdown in one row.
- **Text.** The description lists the changed paths and a short motivation, with no agent narration.
  The agent itself claims no numbers; any measured numbers added later state who measured them, on
  what machine and with which command.

## Pull request text

- `CHANGELOG.md` has an entry under the `[dev]` section with a working link to the pull request and to
  every issue it fixes.
- The description states what changed, why, and the evidence. It contains no tool or agent scaffolding
  text (prompts, "mandates", notes about reading instruction files).
