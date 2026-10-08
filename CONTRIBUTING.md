# Contributing to `mkl_fft`

This document covers the development workflow: how to get a working build, how
to run the checks, and what to include in a pull request.

For end-user installation and API usage, see [README.md](README.md). For a map
of the source tree, see [`AGENTS.md`](AGENTS.md), which links to the local
`AGENTS.md` files in directories that have their own rules. Security
vulnerabilities go through the process in [SECURITY.md](SECURITY.md).

---

## Development setup

Building requires a C compiler, oneMKL headers and libraries (`mkl-devel`), and
NumPy. A conda environment is the least surprising way to get them:

```sh
# add python=X.Y to target a specific interpreter
conda create -n mkl_fft-dev -c conda-forge python pip mkl-devel numpy \
    meson-python ninja cmake cython pytest
conda activate mkl_fft-dev
```

Then build in place, which reuses the environment's MKL and NumPy:

```sh
pip install -e . --no-build-isolation --verbose
```

`pyproject.toml` defines the supported Python range, and
`.github/workflows/build_pip.yml` is canonical for the versions CI covers.

SciPy and `mkl-service` are optional. They are needed only for the
`mkl_fft.interfaces.scipy_fft` adapter, and the tests that exercise it are
skipped without them. To use or test the SciPy interface, add both to the
environment:

```sh
conda install -c conda-forge scipy mkl-service
```

The matching pip extras are declared in `pyproject.toml`: `scipy_interface` for
the SciPy adapter, `test` for `pytest` plus both packages, and `benchmark` for
the ASV suite.

`README.md` documents the non-editable install paths, including the isolated
build that resolves its own `mkl` and `numpy`.

### Rebuilding

`meson-python` rebuilds the extension on import for editable installs, so
editing `.pyx`, `.c.src`, or `meson.build` and rerunning `pytest` is usually
enough. Generated sources and the compiled extension live under `build/<tag>/`
rather than in the source tree. If a build gets into a bad state, `rm -rf build`
and reinstall.

## Running the checks

```sh
pytest mkl_fft/tests          # test suite
pre-commit run --all-files    # lint and format hooks
```

Install the hooks once with `pre-commit install` and they run on each commit.
`.pre-commit-config.yaml` is the source of truth for the tooling.

Opening a pull request also runs CI, which builds and tests the package across
platforms and Python versions and runs various lint and static-analysis checks.

## Code style

Style is loose, and the pre-commit hooks enforce most of it:

- Python is formatted with `black` and `isort`, with a line length of 80.
- Cython is not touched by `black`. `isort` sorts its imports, `cython-lint`
  checks it against the same 80-column limit, and string literals use double
  quotes.
- C sources follow the repository's `.clang-format`.
- Otherwise, match the surrounding code.

## Dos and don'ts

**Do**

- Keep changes atomic and single-purpose.
- Preserve NumPy/SciPy FFT compatibility. This package is used as a drop-in
  replacement, so a behavioral difference is a bug even when the new behavior is
  arguably better. Call out an intentional break in the PR.
- Add tests in `mkl_fft/tests/` alongside behavior changes, and a regression
  test with every bug fix.
- Keep tests deterministic.
- Edit the `*.c.src` templates in `mkl_fft/src/` for C backend changes. The
  `.c` files are generated from them on every build.
- Keep patching reversible and observable: anything installed can be
  uninstalled, and `is_patched()` reports the truth.
- Cite the source-of-truth file for mutable details: `pyproject.toml`,
  `meson.build`, `conda-recipe*/meta.yaml`, `.github/workflows/`.
- Give benchmark numbers reproducible context — hardware, versions, and the
  command you ran.

**Don't**

- Commit generated artifacts, or hand-edit a generated `.c`.
- Hardcode versions, build flags, CI matrices, or channel URLs in documentation.
- Assert on timing or throughput in the test suite.
- Refactor `_vendored/` opportunistically. Keep local diffs minimal and send
  fixes upstream where you can.
- Introduce ISA-specific assumptions outside explicit build configuration.

## Submitting a change

Work on a branch: the `no-commit-to-branch` hook blocks direct commits to
`master` and `maintenance/*`.

If the change is user-visible — behavior, API, packaging, or build output — add
a `CHANGELOG.md` entry under `## [dev]` in the matching section, with a
`[gh-NNN](https://github.com/IntelPython/mkl_fft/pull/NNN)` link. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Docs,
tooling, and CI-only changes are usually left out.

Then open the PR and fill in the template, including what you verified locally
and what you left to CI.

By contributing you agree that your contributions are licensed under the
BSD-3-Clause terms in [LICENSE.txt](LICENSE.txt).
