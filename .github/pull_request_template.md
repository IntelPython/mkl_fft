# Description

<!-- What changed and why. Link any related issues. -->

## Verification

<!-- The commands you ran, and the platform and versions you ran them on. -->

- Tests: <!-- e.g. `pytest mkl_fft/tests`, Python 3.12 / NumPy 2.x, Linux -->
- Lint: <!-- `pre-commit run --all-files` -->

## Not verified

<!--
Anything skipped or left to CI, and why. Examples: Windows, the Intel-channel
conda build, the benchmarks. Write "none" if you ran everything relevant.
-->

## Checklist

- [ ] NumPy/SciPy FFT API compatibility preserved, or the break is intentional and called out above.
- [ ] Behavior changes have tests in `mkl_fft/tests/`; bug fixes have a regression test.
- [ ] `CHANGELOG.md` updated under `## [dev]` with a `[gh-NNN]` link.

<!-- See CONTRIBUTING.md for the build and test workflow, and AGENTS.md for the module map. -->
