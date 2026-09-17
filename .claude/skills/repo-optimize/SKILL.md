---
name: repo-optimize
description: Weekly unattended maintenance sweep for mkl_fft. Improves documentation, contributor templates, agent-instruction files, and ignore rules, then leaves the changes uncommitted with a written report for the caller to commit and propose as a pull request. Never touches source, build configuration, or packaging. Use when running scheduled repository maintenance, or when asked to improve agent-readiness or contributor onboarding surfaces.
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

# Repository maintenance sweep

You are running unattended. No one will answer a question mid-run, so when a
choice is ambiguous, take the smaller action and note it in your report.

Read the root `AGENTS.md` first, then `.github/copilot-instructions.md`.
Precedence is `copilot-instructions` > nearest `AGENTS.md` > root `AGENTS.md`,
and this skill does not override any of them.

## Scope

You may only create or edit files matching:

- `*.md` in the repository root
- `docs/**`
- `AGENTS.md` at any depth
- `.github/pull_request_template.md`, `.github/ISSUE_TEMPLATE/**`
- `.gitignore`, `.gitattributes`

Everything else is out of scope. In particular, never touch:

- `mkl_fft/**` (including `.pyx`, `.c.src`, and tests), `_vendored/**`
- `meson.build`, `pyproject.toml`, `conda-recipe/**`, `conda-recipe-cf/**`
- `.github/workflows/**` — you must not change the automation that runs you
- `.claude/**` — you must not change your own instructions or permissions
- `CHANGELOG.md` — this sweep is never user-visible, so it gets no entry
- Anything requiring a version bump

If a finding can only be fixed outside this scope, do not fix it. Record it in
your report under "Out of scope" and move on. The workflow enforces these paths
independently and will fail the run if the diff steps outside them.

## Task list

Work through these in order. Stop at the first one that produces a substantial
change, so each pull request stays reviewable.

1. **Run the readiness scan.** If `$AGENTREADY_DIR` is set, the workflow has
   provisioned AgentReady there. Pass this repository as an absolute path —
   `npm --prefix` runs the script with its working directory set to the prefix,
   so a relative `.` would scan AgentReady itself:
   `npm --prefix "$AGENTREADY_DIR" run agentready -- scan "$PWD" --format markdown`.
   Address only findings whose fix lands inside Scope. Treat its score as a
   signal, not a target — do not add a file solely to move a number.
2. **`AGENTS.md` coverage.** Every directory holding source, tests, packaging,
   or benchmarks should have a local `AGENTS.md`, and the root "Directory map"
   should list all of them. Add missing ones in the style of the existing
   files: a one-line purpose, a `## Scope` list, and a `## Guardrails` list.
   You may add guardrails; never delete or weaken an existing one.
3. **Contributor templates.** A pull-request template and issue forms under
   `.github/ISSUE_TEMPLATE/`. Ask for the evidence a reviewer needs: what was
   verified locally, what was left to CI, versions, and a reproducer.
4. **Developer documentation.** `CONTRIBUTING.md` covering the build, the
   checks, and how the Python, Cython, and C template layers relate. Verify
   every command you document by running it, or leave it out.
5. **Ignore rules.** Developer-local and build artifacts that appear in
   `git status` but are absent from `.gitignore`.

## Rules that override convenience

- **Never document a command you have not run.** A wrong build instruction is
  worse than a missing one. If you cannot run it in this environment, say so in
  your report instead of guessing.
- **Cite source-of-truth files rather than copying mutable values.** No pinned
  Python versions, CI matrices, dependency versions, or channel URLs in prose —
  point at `pyproject.toml`, `.github/workflows/`, or `conda-recipe*/meta.yaml`.
  This is a hard rule from root `AGENTS.md` and reviewers enforce it.
- **Match the surrounding style.** These files are terse. Prefer a short
  imperative list over explanatory prose, and keep negatives in one place
  rather than sprinkling "do X, not Y" through descriptive text.
- **Prefer no change over a speculative one.** A quiet week is a success.

## Verify before opening anything

```sh
pre-commit run --files <every file you changed>
```

Pass only the files you touched, and fix what it reports. Because this sweep
changes documentation and configuration, the Python, Cython, C and shell hooks
normally have no matching files and report nothing.

One failure is not yours to fix: `no-commit-to-branch` is `always_run`, so
`--files` does not skip it, and it fails whenever HEAD is on `master` or
`maintenance/*`. Note it in the report and continue. If a hook fails for a
reason you cannot attribute to your own diff, say so rather than editing an
unrelated file to silence it.

You changed no source, so do not run or claim to have run the test suite.

## Reporting the result

Stop without proposing anything if you changed nothing. An empty pull request is
a worse outcome than silence.

Do not commit, push, or open a pull request, and do not run `git commit`,
`git push`, or any `gh` command. Leave your work as uncommitted changes. The
caller checks your diff against the allowed paths, then commits it to a single
reused branch so weekly runs do not stack. If you are running this skill outside
that automation, hand the diff and your report to whoever invoked you.

Write the pull-request body to `pr-body.md` in the repository root. Follow the
repository's pull-request template if one exists. Whatever the shape, it must
state:

- what changed and why, one line per file
- the exact commands you ran, and their results
- what you did not verify, and why
- findings you left alone, with the reason
- any `AGENTS.md` guardrail you added, called out for review

Write the body as a record of what happened, not a summary of intent. A
reviewer should be able to tell from it alone whether to trust the diff.
