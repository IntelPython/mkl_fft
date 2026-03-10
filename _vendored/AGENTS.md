# AGENTS.md — _vendored/

Vendored utilities used by build/code-generation flows.

## Guardrails
- Prefer updating upstream source when feasible; keep local vendored diffs minimal.
- Do not refactor vendored code opportunistically in unrelated PRs.
- If vendored logic affects generated sources, validate downstream build/test paths.
