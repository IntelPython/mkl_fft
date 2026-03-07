# AGENTS.md — mkl_fft/src/

C template backend for FFT internals.

## Scope
- Template sources (`*.c.src`) used to generate C implementation artifacts.

## Guardrails
- Edit template files (`*.c.src`) rather than generated `.c` files.
- Keep template changes tightly scoped and validated by tests.
- Coordinate with Cython/binding/API layers when changing function behavior.
