# AGENTS.md — mkl_fft/interfaces/

Adapter layer for `numpy.fft` / `scipy.fft` style interfaces.

## Scope
- Interface modules mapping user-facing calls to MKL-backed implementation.

## Guardrails
- Preserve function signatures and behavioral expectations unless explicitly requested.
- Keep compatibility with upstream NumPy/SciPy semantics in mind.
- Any user-visible behavior change here should include tests in `mkl_fft/tests/`.
