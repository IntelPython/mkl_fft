# AGENTS.md — mkl_fft

## What this project is
NumPy-based Python interface to Intel® oneMKL FFT functions. Provides MKL-accelerated FFT for real/complex transforms. Part of Intel® Distribution for Python. Archetype: **python** (Cython + C extensions).

Layers: Python interfaces, Cython bindings (`_pydfti.pyx`), C backend (`src/*.c.src`).

## How it's structured
- `mkl_fft/interfaces/` — drop-in replacements for `numpy.fft`, `scipy.fft`
- `mkl_fft/src/` — C templates (`.c.src`)
- `mkl_fft/tests/` — pytest suite
- `conda-recipe/`, `conda-recipe-cf/` — Intel/conda-forge builds

Build: `pyproject.toml` + `setup.py`. Runtime: `mkl-service`, `numpy>=1.26.4`.

## How to work in it
- Keep changes atomic and single-purpose.
- Preserve NumPy/SciPy FFT API; document divergence in commit message.
- Pair changes with tests and docstrings.
- Never assume MKL or NumPy versions; use source-of-truth files.
- **C templates:** Edit only `*.c.src` files; generated `.c` is ephemeral (via `_vendored/conv_template.py`).
- **Local dev:** `conda create -n dev python numpy cython mkl-devel pytest && pip install -e .`

For agent policy: `.github/copilot-instructions.md`

## Where truth lives
- Build/config: `pyproject.toml`, `setup.py`
- Dependencies: `pyproject.toml` (`dependencies`, `optional-dependencies`), `conda-recipe/meta.yaml`, `conda-recipe-cf/meta.yaml`
- CI: `.github/workflows/`
- Style/lint: `.pre-commit-config.yaml`, `.flake8`, `pyproject.toml` (black/isort/cython-lint)
- API/contracts: `mkl_fft/__init__.py`, `mkl_fft/interfaces/`, NumPy/SciPy FFT docs
- Stable entry points: `python -m pip install .`, `pytest mkl_fft/tests`

## Directory map
No local AGENTS files — project is small enough for root-level guidance only.
