# AGENTS.md — .github/

CI/CD workflows and repo automation.

## Workflows (source of truth)
- `conda-package.yml` — Intel channel conda build/test pipeline
- `conda-package-cf.yml` — conda-forge-oriented build/test pipeline
- `build-with-clang.yml` — clang compatibility checks
- `build_pip.yml` — editable pip build pipeline
- `build-with-standard-clang.yml` — standard clang compatibility checks
- `pre-commit.yml` — lint/format checks
- `openssf-scorecard.yml` — security scanning

## Policy
- Treat workflow YAML as canonical for platform/Python matrices.
- Avoid doc claims about CI coverage unless present in workflow config.
