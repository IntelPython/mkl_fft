# AGENTS.md — mkl_fft

Entry point for agent context in this repo.

## What this project is
`mkl_fft` is a NumPy/SciPy-compatible FFT interface backed by Intel® oneMKL.
It provides accelerated FFT transforms while aiming to preserve upstream API behavior.

## Key components
- **Package:** `mkl_fft/`
- **Cython bindings:** `mkl_fft/_pydfti.pyx`
- **Template-based C backend:** `mkl_fft/src/*.c.src`
- **Interface adapters:** `mkl_fft/interfaces/`
- **Tests:** `mkl_fft/tests/`
- **Vendored helpers:** `_vendored/`
- **Packaging:** `conda-recipe/`, `conda-recipe-cf/`

## Build/runtime basics
- Build system: `pyproject.toml` + `setup.py`
- Build deps: `cython`, `numpy`, `mkl-devel`
- Runtime deps: `numpy`, `mkl-service`

## Development guardrails
- Preserve NumPy/SciPy FFT API compatibility unless change is explicitly requested.
- Edit template sources (`*.c.src`), not generated C artifacts.
- Pair behavior changes with tests and keep diffs minimal.
- Avoid hardcoding mutable versions/matrices/channels in docs.

## Where truth lives
- Build/config: `pyproject.toml`, `setup.py`
- Dependencies: `pyproject.toml`, `conda-recipe*/meta.yaml`
- CI/workflows: `.github/workflows/*.{yml,yaml}`
- Public API: `mkl_fft/__init__.py`, `mkl_fft/interfaces/`
- Tests: `mkl_fft/tests/`

For behavior policy, see `.github/copilot-instructions.md`.

## Directory map
Use nearest local `AGENTS.md` when present:
- `.github/AGENTS.md` — CI workflows and automation policy
- `mkl_fft/AGENTS.md` — package-level implementation context
- `mkl_fft/interfaces/AGENTS.md` — NumPy/SciPy interface adapters
- `mkl_fft/src/AGENTS.md` — C template backend rules
- `mkl_fft/tests/AGENTS.md` — test scope/conventions
- `conda-recipe/AGENTS.md` — Intel-channel conda packaging
- `conda-recipe-cf/AGENTS.md` — conda-forge recipe context
- `_vendored/AGENTS.md` — vendored tooling boundaries
