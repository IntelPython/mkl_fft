# AGENTS.md — benchmarks/

ASV performance suite for `mkl_fft`.

## Scope
- `asv.conf.json` — ASV configuration, channels, and regression thresholds
- `benchmarks/` — benchmark modules (`bench_fft1d`, `bench_fftnd`,
  `bench_interfaces`, `bench_memory`) plus shared bases in `_utils.py`
- `README.md` — coverage table, threading model, and run commands

## Guardrails
- Treat `asv.conf.json` as canonical for ASV settings; treat `README.md` as
  canonical for what each module covers.
- Comparability across machines depends on the thread default in
  `benchmarks/__init__.py` and the DFTI warmup in each `setup`. Changing either
  invalidates comparison against existing results — call it out explicitly.
- Keep inputs deterministic; benchmarks seed their own RNG.
- Report performance numbers with reproducible context: hardware, thread count,
  versions, and the command used.
- Benchmark results under `.asv/` are local artifacts and are gitignored.
