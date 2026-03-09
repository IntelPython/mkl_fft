# AGENTS.md — mkl_fft/

Core package implementation for MKL-backed FFT operations.

## Key files
- `__init__.py` — public package API surface
- `_pydfti.pyx` — Cython bindings for DFTI interactions
- `_mkl_fft.py` / `_fft_utils.py` — runtime FFT helper logic
- `interfaces/` — NumPy/SciPy adapter modules
- `src/` — C template backend (`*.c.src`)
- `tests/` — package tests (see local AGENTS in tests)

## Guardrails
- Preserve NumPy/SciPy-compatible behavior by default.
- Keep adapter, binding, and backend changes coordinated with tests.
- Treat interface wrappers as patch/integration points: behavior changes must stay explicit and reversible.
- Prefer minimal isolated edits around changed API paths.
