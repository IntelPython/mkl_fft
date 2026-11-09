# AGENTS.md — conda-recipe/

The conda recipe for `mkl_fft`, built against conda-forge only.

## Scope
- conda-forge specific recipe files under this directory.

## Guardrails
- Align changes with how `intel-ci.yml` builds and tests the recipe (`ci/build.py`, `ci/test.py`).
- `conda_build_config.yaml`'s `python:` list is the Intel CI conda matrix, read by `ci/matrix.py`: one version per line, no selectors.
- Wheels don't come from this recipe: `intel-ci.yml` builds them from `pyproject.toml` with `ci/build.py`.
