# AGENTS.md — conda-recipe-cf/

The conda recipe for `mkl_fft`, built against conda-forge only.

## Scope
- conda-forge specific recipe files under this directory.

## Guardrails
- Align changes with `conda-package-cf.yml` workflow behavior.
- `conda_build_config.yaml`'s `python:` list is the Intel Python CI conda matrix, read by `ci/matrix.py`: one version per line, no selectors.
- Wheels don't come from this recipe: `intel-ci.yml` builds them from `pyproject.toml` with `ci/build.py`.
