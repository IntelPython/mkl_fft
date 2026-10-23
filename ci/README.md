# How Intel Python CI tests mkl_fft

Intel Python CI (gh-ci) builds and tests mkl_fft's conda-forge packages in GitHub Actions. Intel's Jenkins then tests each package again on Intel machines and scans it, and on a release tag publishes the packages it tested. Everything it needs from this repository is in this directory, one workflow, and the recipe it builds.

| File | Role |
|---|---|
| `../conda-recipe-cf/` | The recipe gh-ci builds. `conda_build_config.yaml` lists the Python versions. The recipe builds on Linux and Windows unless a platform-only skip in `meta.yaml`, such as `skip: true  # [win]`, rules one out. |
| `../.github/workflows/intel-ci.yml` | One job reads the matrix, one job per platform and Python version builds and uploads `mkl_fft <Platform> Python <version>`, and one job per package tests it. |
| `matrix.py` | Reads the platforms and Python versions from the recipe. A pull request gets the two newest regular versions. |
| `build.py` | Builds one Python version with conda-build, which runs the recipe's own tests. |
| `hooks.py` | The workflow's test jobs: installs one package into a clean conda-forge environment with the requirements in `config.yml`, and runs `test.py` the way Jenkins does. |
| `config.yml` | Tells gh-ci which workflow builds the packages, what its artifacts are called, and what the tests need. |
| `test.py` | Runs the installed package's test suite with pytest, in Actions and again in Jenkins. |

The scripts need only the Python standard library and are the same in every repository gh-ci tests, except that `matrix.py` here looks for `conda-recipe-cf/` first. Scanning is not the repository's to define: Jenkins runs its own scans.

**A pull request into `master`.** Actions builds the two newest regular Python versions on Linux and Windows and tests each package. Jenkins notices the pull request within 3 minutes, waits for that whole run, and then tests and scans every package at once. Each cell posts its own status on the pull request, and `intel-ci/pr` sums them up.

**A release tag.** Actions builds every declared version, and Jenkins tests and scans all of them before it publishes the release.

To support another Python, add it to `conda-recipe-cf/conda_build_config.yaml`; nothing else lists versions. Try it locally from a conda base environment with conda-build:

```bash
python ci/matrix.py                           # every platform and Python version
python ci/matrix.py --event pull_request      # what a pull request builds
python ci/build.py --python VERSION --output /tmp/out     # VERSION: one matrix.py lists
python ci/hooks.py --python VERSION --artifacts /tmp/out --work /tmp/hooks
```
