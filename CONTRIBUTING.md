# Contributing to `mkl_fft`

This document covers the development
workflow: how to get a working build, how to run the checks, and how the code is
laid out so a change lands in the right layer.

For end-user installation and API usage, see [README.md](README.md). Security
vulnerabilities go through the process in [SECURITY.md](SECURITY.md).

---

## Development setup

Building requires a C compiler, oneMKL headers and libraries (`mkl-devel`), and
NumPy. A conda environment is the least surprising way to get them:

```sh
conda create -n mkl_fft-dev -c conda-forge python=3.12 pip mkl-devel numpy \
    meson-python ninja cmake cython pytest scipy mkl-service
conda activate mkl_fft-dev
```

Then build in place, which reuses the environment's MKL and NumPy:

```sh
pip install -e ".[test]" --no-build-isolation --verbose
```

The `[test]` extra pulls in `pytest`, `scipy`, and `mkl-service`. Other extras
are declared in `pyproject.toml`: `scipy_interface` for the SciPy adapter at
runtime and `benchmark` for the ASV suite.

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
`.pre-commit-config.yaml` is the source of truth for the tooling; today it
covers `black`, `isort`, `flake8`, `pylint` (errors only), `cython-lint`,
`clang-format`, `codespell`, `shellcheck`, `gitleaks`, and `actionlint`. Line
length is 80 for Python, Cython, and TOML.

### What CI runs

`.github/workflows/*.yml` is canonical for platform and Python matrices. In
outline:

| Workflow | Purpose |
| --- | --- |
| `conda-package.yml` | conda build and test against the Intel channel |
| `conda-package-cf.yml` | conda build and test against conda-forge only |
| `build_pip.yml` | editable pip build, including pre-release NumPy |
| `build-with-clang.yml` | build with the IntelLLVM `icx` compiler |
| `build-with-standard-clang.yml` | build with upstream clang |
| `pre-commit.yml` | lint and format |
| `coverity.yml` | static analysis (see `coverity/README.md`) |
| `openssf-scorecard.yml`, `zizmor.yml` | supply-chain and workflow security |

To reproduce a conda packaging failure locally, build the recipe the same way CI
does — `conda build --python <ver> --numpy <ver> -c <channels> --override-channels conda-recipe`
(or `conda-recipe-cf` for the conda-forge variant). The recipe directories are
canonical for packaging intent and dependency pins.

## How the code fits together

Build configuration lives in `pyproject.toml` (with `meson-python` as the build
backend) and `meson.build`. The version is read from `mkl_fft/_version.py` by
`meson.build`, so that file is the single place a version is set.

A transform call flows down through these layers:

```
mkl_fft.interfaces.numpy_fft / scipy_fft   drop-in NumPy/SciPy adapters
mkl_fft (__init__.py)                      public FFT API
mkl_fft/_mkl_fft.py, _fft_utils.py         argument handling, normalization, dispatch
mkl_fft/_pydfti.pyx                        Cython bindings
mkl_fft/src/mklfft.c.src  ->  mklfft.c     C backend, generated at build time
oneMKL DFTI
```

Directories:

- **`mkl_fft/`** — the package. `__init__.py` is the public API surface;
  `_mkl_fft.py` and `_fft_utils.py` hold the Python-level FFT logic;
  `_pydfti.pyx` is the Cython binding layer.
- **`mkl_fft/src/`** — the C backend, written as `*.c.src` templates. At build
  time `_vendored/process_src_template.py` expands `mklfft.c.src` into
  `mklfft.c`, which is compiled into the `_pydfti` extension. The generated
  `.c` is regenerated on every build, so only template edits survive.
- **`mkl_fft/interfaces/`** — adapters presenting `numpy.fft`- and
  `scipy.fft`-shaped APIs. `numpy_fft.py` and `scipy_fft.py` are the public
  modules; the `_`-prefixed siblings are implementation. Upstream signatures and
  semantics are the contract here.
- **`mkl_fft/`** patching layer — `patch.py`, `with_patch.py`, `_patch_numpy.py`,
  `_patch_startup.py`, and the `__main__.py` CLI implement the monkey-patching
  entry points documented in the README. The contract is that patching stays
  reversible and observable: anything installed can be uninstalled, and
  `is_patched()` reports the truth.
- **`mkl_fft/tests/`** — the suite. `helper.py` holds shared utilities and
  `third_party/` carries tests adapted from upstream projects.
- **`_vendored/`** — build-time code-generation helpers vendored from NumPy.
  They are excluded from `black` and `isort` in `pyproject.toml`.
- **`conda-recipe/`**, **`conda-recipe-cf/`** — Intel-channel and conda-forge
  packaging.
- **`benchmarks/`** — ASV benchmarks, run with the `benchmark` extra.

Each of these directories has an `AGENTS.md` stating the same boundaries for
coding agents; [`AGENTS.md`](AGENTS.md) at the root indexes them and is a useful
orientation map for humans too.

## Dos and don'ts

**Do**

- Keep changes atomic and single-purpose.
- Preserve NumPy/SciPy FFT compatibility. This package is used as a drop-in
  replacement, so a behavioral difference is a bug even when the new behavior is
  arguably better. Call out an intentional break in the PR.
- Add tests in `mkl_fft/tests/` alongside behavior changes, and a regression
  test with every bug fix.
- Keep tests deterministic.
- Edit the `*.c.src` templates for C backend changes.
- Keep patching reversible and observable.
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

Add a `CHANGELOG.md` entry under `## [dev]` in the matching section, with a
`[gh-NNN](https://github.com/IntelPython/mkl_fft/pull/NNN)` link. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Then open the PR and fill in the template, including what you verified locally
and what you left to CI.

By contributing you agree that your contributions are licensed under the
BSD-3-Clause terms in [LICENSE.txt](LICENSE.txt).
