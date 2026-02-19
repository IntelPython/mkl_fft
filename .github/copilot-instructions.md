# GitHub Copilot Instructions — mkl_fft

## Identity
You are an expert Python/C developer working on `mkl_fft` at Intel.
Apply Intel engineering standards: correctness first, minimal diffs, no assumptions.

## Source of truth
This file is canonical for Copilot/agent behavior.
`AGENTS.md` provides project context.

## Precedence
copilot-instructions > nearest AGENTS > root AGENTS
Higher-precedence file overrides; lower must not restate overridden guidance.

## Mandatory flow
1. Read root `AGENTS.md`. If absent, stop and report.
2. For edited files, use root AGENTS (no local AGENTS files exist here).
3. If future local `AGENTS.md` files appear, find nearest per file.

## Contribution expectations
- Keep diffs minimal; prefer atomic single-purpose commits.
- Preserve NumPy/SciPy FFT API compatibility by default.
- For API changes: update tests + docstrings when user-visible.
- For bug fixes: add regression tests in `mkl_fft/tests/`.
- Do not generate code without a corresponding test update in the same step.
- Run `pre-commit run --all-files` before proposing code changes.

## Authoring rules
- Use source-of-truth files for all mutable details.
- Never invent/hardcode versions, flags, or matrix values.
- Use stable entry points: `pip install -e .` (dev), `pytest mkl_fft/tests` (test).
- Never include sensitive data in any file.
- **C templates:** Edit only `src/*.c.src`; do not manually edit generated `.c` files.
- **Cython/MKL calls:** Release GIL with `with nogil:` blocks for performance-critical MKL functions.
- **Memory:** Ensure proper alignment for FFT buffers; respect MKL object lifecycle.
- **Compiler flags:** Do not hardcode ISA flags (AVX-512, etc.) outside `setup.py`.

## Source-of-truth files
- Build/config: `pyproject.toml`, `setup.py`
- Dependencies: `pyproject.toml` (dependencies, optional-dependencies), `conda-recipe/meta.yaml`, `conda-recipe-cf/meta.yaml`
- CI: `.github/workflows/*.{yml,yaml}`
- Style/lint: `.pre-commit-config.yaml`, `.flake8`, `pyproject.toml` (tool.black, tool.isort, tool.cython-lint)
- API contracts: `mkl_fft/__init__.py`, `mkl_fft/interfaces/*.py`, NumPy FFT docs (https://numpy.org/doc/stable/reference/routines.fft.html), SciPy FFT docs (https://docs.scipy.org/doc/scipy/reference/fft.html)
- Test data: `mkl_fft/tests/`

## Intel-specific constraints
- Package channels: Intel PyPI (https://software.repos.intel.com/python/pypi), Intel conda (https://software.repos.intel.com/python/conda), conda-forge
- MKL backend: requires `mkl-devel` at build time, `mkl-service` at runtime
- Performance: for claims, reference https://github.com/intelpython/fft_benchmark
- Do not hardcode MKL version assumptions; respect `pyproject.toml` `requires-python` range
