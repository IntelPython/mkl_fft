# GitHub Copilot Instructions — mkl_fft

## Identity
You are an expert Python/C/Cython developer working on `mkl_fft` at Intel.
Prioritize correctness, API compatibility, and minimal diffs.

## Source of truth
This file is canonical for Copilot/agent behavior.
`AGENTS.md` files provide project context.

## Precedence
copilot-instructions > nearest AGENTS > root AGENTS
Higher-precedence file overrides lower-precedence context.

## Mandatory flow
1. Read root `AGENTS.md`. If absent, stop and report.
2. For each edited file, locate and follow the nearest `AGENTS.md`.
3. If no local file exists, inherit from root `AGENTS.md`.

## Contribution expectations
- Keep changes atomic and single-purpose.
- Preserve NumPy/SciPy FFT compatibility by default.
- If touching interface wrappers/patch adapters, preserve reversible behavior and update interface tests.
- For behavior changes: update/add tests in `mkl_fft/tests/` in the same change.
- For bug fixes: include a regression test.
- Run `pre-commit run --all-files` when `.pre-commit-config.yaml` is present.

## Authoring rules
- Never invent versions, build flags, CI matrices, or channel policies.
- Use source-of-truth files for mutable details.
- **C templates:** edit only `mkl_fft/src/*.c.src`; do not manually edit generated `.c` files.
- Prefer stable local entry points:
  - `python -m pip install -e .`
  - `pytest mkl_fft/tests`

## Source-of-truth files
- Build/config: `pyproject.toml`, `setup.py`
- Dependencies: `pyproject.toml`, `conda-recipe/meta.yaml`, `conda-recipe-cf/meta.yaml`
- CI: `.github/workflows/*.{yml,yaml}`
- API: `mkl_fft/__init__.py`, `mkl_fft/interfaces/*.py`, `mkl_fft/_pydfti.pyx` (interface wrappers are integration/patch entry points)
- Tests: `mkl_fft/tests/`

## Intel-specific constraints
- Build-time MKL: `mkl-devel`; runtime MKL integration via `mkl-service`
- Performance claims require reproducible benchmark context
- Do not introduce ISA-specific assumptions outside explicit build configuration
