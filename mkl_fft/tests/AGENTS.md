# AGENTS.md — mkl_fft/tests/

Test suite for FFT API compatibility and regressions.

## Expectations
- Behavior changes should include test updates in the same PR.
- Bug fixes require regression tests.
- Keep tests deterministic and avoid brittle performance-dependent assertions.

## Entry point
- `pytest mkl_fft/tests`
