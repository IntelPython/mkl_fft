---
name: repo-optimize
description: Weekly unattended performance and correctness sweep for mkl_fft. Hunts for a real speedup in the FFT implementation, or for a genuine bug, implements exactly one of them, proves it against the full test suite, and leaves it uncommitted with a written report for the caller to propose as a pull request. CI measures the benchmark effect on the pull request. Falls back to documentation upkeep only when it finds neither. Use when running scheduled repository maintenance, looking for performance optimizations, or hunting for latent bugs.
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

# Performance and correctness sweep

Your goal is a measurable speedup in `mkl_fft`. If you find a genuine correctness
bug while looking, fixing that comes first — shipping wrong numerical results is
worse than shipping slow ones. Documentation work is a fallback for weeks when you
find neither.

Do exactly one thing per run: one optimization, or one bug fix. Never both. A
pull request carrying two changes cannot be attributed by a benchmark and is
harder to revert.

You are running unattended. No one will answer a question mid-run, so when a
choice is ambiguous, take the smaller action and note it in your report.

Read the root `AGENTS.md` first, then `.github/copilot-instructions.md`.
Precedence is `copilot-instructions` > nearest `AGENTS.md` > root `AGENTS.md`,
and this skill does not override any of them.

## Scope

You may edit:

- `mkl_fft/**` — the Python layer, `_pydfti.pyx`, and the `src/*.c.src` templates
- `mkl_fft/tests/` — with care; see "Touching the existing suite" below. You may
  not add a `conftest.py`, and you may not delete or rename a test file.
- root `*.md`, `docs/**`, `AGENTS.md` at any depth, `.gitignore`, `.gitattributes`
  (fallback work only)

You must not touch:

- `benchmarks/**` — the benchmark suite is the measuring instrument. The thing
  being measured does not get to adjust it.
- `meson.build`, `pyproject.toml`, `mkl_fft/_version.py` — compiler flags and
  build configuration affect every artifact we ship. Propose flag changes in your
  report; do not implement them.
- `conda-recipe/**`, `conda-recipe-cf/**`, `_vendored/**`
- `.github/workflows/**` — you must not change the automation that runs you
- `.claude/**` — you must not change your own instructions or permissions
- `CHANGELOG.md` — an entry needs the pull-request number, which does not exist
  yet. Propose the wording in your report instead.

Edit the `*.c.src` templates, never a generated `.c`. Generated output is
regenerated on every build, so only template edits survive.

**Work within the existing files.** `meson.build` lists every installed Python
source and every extension source explicitly, and you cannot edit it — so a new
module or a new C file would not be installed and would not import. Refactor
inside the files that exist; if a change genuinely needs a new file, describe it
in your report and leave it to a human.

The caller re-checks the diff and fails the run if it strays into a forbidden
path, but its deny list is narrower than the scope above. Treat the scope as
yours to honour rather than as something that will be caught for you.

## Track A — performance

**The rule that outranks the goal: a performance change must be
behaviour-preserving.** This package is a drop-in replacement for `numpy.fft` and
`scipy.fft`, and callers trust its numbers. A change that is faster and subtly
wrong — different rounding on a stride pattern, a descriptor reused across
incompatible shapes, an allocation skipped that was load-bearing — is far worse
than no change at all.

If you cannot convince yourself a change is exactly equivalent for every dtype,
shape, stride and normalization the affected path accepts, do not make it.
Describe it in your report as a candidate for a human.

Where this package has historically paid, ranked:

1. **Descriptor handling in `_pydfti.pyx`.** oneMKL builds a DFTI descriptor per
   (size, dtype, strides) combination. Creating, committing, or freeing one more
   often than necessary is real time on small transforms.
2. **Avoidable copies.** oneMKL accepts arbitrary strides as long as they are an
   integer multiple of the item size. A copy taken defensively where the input was
   already acceptable is pure waste. `_fft_utils.py` and `_mkl_fft.py` decide this.
3. **Per-call Python overhead.** Argument normalization, shape and axis
   validation, and `norm` handling run on every call and dominate small
   transforms. `interfaces/` adds another layer for the drop-in APIs.
4. **Allocation.** Output and scratch buffers allocated when an existing buffer
   would serve, or allocated so as to defeat reuse.

If you cannot say *why* a change is faster in one sentence, it is not a candidate.

Measure locally for your report only: `asv run --python=same --quick HEAD^!` or a
direct timing script. Your numbers are indicative. CI runs the authoritative
benchmark on the pull request, so report the hypothesis and mechanism clearly
enough that the benchmark result can confirm or refute it.

## Track B — bug fixes

A bug is a defect you can demonstrate with a failing test, not a smell. Where
they live in this codebase:

1. **Error and status handling in `_pydfti.pyx`.** oneMKL returns status codes
   from descriptor and compute calls. A discarded status, or one checked only
   under `assert`, hides a real failure.
2. **Memory management.** Allocation-failure paths, leaks when a later step
   raises, and multi-iterator construction. This has been a recurring source of
   real defects.
3. **Edge cases.** Zero-size and empty arrays, zero or negative strides,
   non-contiguous input, `out=` aliasing the input, unusual `norm` combinations,
   and `float128` downcasting in `interfaces/_float_utils.py`.
4. **Upstream divergence.** Any place `mkl_fft.interfaces` returns a different
   dtype, shape, error type, or value than the `numpy.fft` / `scipy.fft` function
   it stands in for. Upstream semantics are the contract.

Work test-first, and say so in your report:

1. Add `mkl_fft/tests/test_<name>.py` containing a test that captures the defect.
2. Run it and **confirm it fails** against the unmodified code. If it passes,
   you have not found a bug — delete the test and move on.
3. Fix the defect in the smallest possible diff.
4. Confirm the new test passes and the whole suite still does.

Report the failure you observed before the fix, verbatim. A bug fix whose test
was never seen to fail is not a bug fix.

Because a bug fix changes behaviour by design, the behaviour-preserving rule
above does not apply to it — but the change must be *only* the fix. State the
user-visible impact plainly, and propose `CHANGELOG.md` wording for a human to
add under `## [dev]` with the pull-request number.

### Touching the existing suite

Sometimes a test encodes the bug. If a test asserts the wrong error type, the
wrong dtype, or a value that never matched upstream, then correcting it *is* part
of the fix — this repository has done exactly that before, when a path was changed
to raise `MemoryError` instead of `ValueError`. Adding a test and leaving the
wrong assertion in place would be dishonest.

So you may modify an existing test. The bar is high, and two things are never
acceptable:

- **Removing a test function.** Retiring coverage is a human decision, always.
- **Adding a `skip`, `skipif` or `xfail` marker to an existing test.** A fix never
  needs to silence a test. (A new test file of your own may carry a `skipif` for a
  genuine platform constraint.)

Anything else — changing an assertion, tightening a bound, correcting an expected
error type — is allowed but flagged prominently on the pull request for a reviewer.
When you do it, your report must state, for each modified test:

- the exact assertion before and after
- what the old assertion claimed, and the evidence it was wrong — ideally the
  upstream `numpy.fft` or `scipy.fft` behaviour it failed to match
- why this is a correction to the test rather than an accommodation of your diff

If you cannot make that case, you have not understood the bug well enough to fix
it. Report the finding and change nothing.

## Rules that override convenience

- **Never edit a test to make a change pass.** If the suite fails, your change is
  wrong until proven otherwise. Revert it and report the finding. Correcting a
  test that encoded the bug is a different act, and needs the evidence described
  under "Touching the existing suite".
- **No API change.** Signatures and accepted inputs stay as they are. Error types
  and return values change only as the explicit content of a bug fix.
- **Never document or claim a command you have not run.**
- **Cite source-of-truth files rather than copying mutable values** — no pinned
  versions, matrices, or channel URLs in prose. A hard rule from root `AGENTS.md`.
- **One change, minimal diff.** Resist tidying code you are not fixing or
  optimizing; it makes the effect unreadable to a reviewer and to the benchmark.
- **Prefer no change over a speculative one.** A quiet week is a success. An
  unmeasurable "optimization" or a guessed-at "fix" in an FFT kernel is a
  liability.

## Verify before reporting

```sh
pip install -e ".[test]" --no-build-isolation
pytest mkl_fft/tests
pre-commit run --files <every file you changed>
```

The whole suite must pass. The caller re-runs the build and the suite when it is
configured to and discards your work if either fails, but that is a backstop —
running them yourself is how you find out whether your change is right.

Source edits mean the Python, Cython and C hooks apply: `black`, `isort`,
`flake8`, `pylint`, `cython-lint` and `clang-format` will all have opinions. Fix
what they report.

One failure is not yours to fix: `no-commit-to-branch` is `always_run`, so
`--files` does not skip it, and it fails whenever HEAD is on `master` or
`maintenance/*`. Note it and continue. If a hook fails for a reason you cannot
attribute to your own diff, say so rather than editing an unrelated file to
silence it.

## Reporting the result

Stop without proposing anything if you changed nothing. An empty pull request is
a worse outcome than silence.

Do not commit, push, or open a pull request, and do not run `git commit`,
`git push`, or any `gh` command. You have no write credentials. Leave your work as
uncommitted changes: the caller checks the diff, re-runs the verification it is
configured with, and turns the result into a patch or a branch for a human to
review. If you are running this skill by hand, hand the diff and your report to
whoever invoked you.

Write a one-line pull-request title to `pr-title.txt` in the repository root,
prefixed `perf:` or `fix:` to match the track you took. Write the report itself to
`pr-body.md` in the repository root, following the repository's pull-request
template if one exists. The caller moves both out of the tree before proposing the
diff, so neither appears in it.

Whatever the shape, the report must state:

- **which track** you took, and why that was the best available change
- **the mechanism** — what was slow or wrong, and why the change addresses it, in
  plain terms a reviewer can check against the diff
- **for a performance change**, why it is behaviour-preserving: the dtypes,
  shapes, strides and normalizations the path accepts, and why each is unaffected
- **for a bug fix**, the failing output you observed before the fix, the
  user-visible impact, and proposed `CHANGELOG.md` wording
- the exact commands you ran and their results, including the test count
- what you measured locally, how, and on what hardware — labelled indicative,
  with CI's benchmark named as the authoritative check
- candidates you rejected, and why
- anything you left to a human, including build-flag ideas you could not implement

Write it as a record of what happened, not a summary of intent. A reviewer should
be able to tell from it alone whether to trust the diff.
