# AGENTS.md — .github/

CI/CD workflows and repo automation.

## Workflows (source of truth)
- `intel-ci.yml` — Intel CI: builds and tests the conda packages of `conda-recipe` and the wheels, for the platforms and Python versions the recipe and `pyproject.toml` declare (`ci/matrix.py`); see `ci/README.md`; its `Agent guidance` job (`ci/guidance.py`) fails a pull request that adds a broken claim to a changed guidance file
- `build-with-clang.yml` — clang compatibility checks
- `build_pip.yml` — editable pip build pipeline
- `build-with-standard-clang.yml` — standard clang compatibility checks
- `pre-commit.yml` — lint/format checks
- `openssf-scorecard.yml` — security scanning

## Policy
- Treat workflow YAML as canonical for platform/Python matrices, except `intel-ci.yml`, whose conda matrix is `conda-recipe/conda_build_config.yaml` and whose wheel matrix is the Python classifiers in `pyproject.toml`.
- Avoid doc claims about CI coverage unless present in workflow config.
