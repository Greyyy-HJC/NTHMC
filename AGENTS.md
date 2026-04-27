# AGENTS.md

Project-specific instructions for coding agents working in this repository.

## Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State assumptions explicitly.
- If multiple interpretations exist, present them instead of picking silently.
- If something is unclear and cannot be resolved from the repository, ask before implementing.
- If a simpler approach exists, say so.

## Simplicity First

Write the minimum code that solves the requested problem.

- No features beyond what was asked.
- No speculative abstractions.
- No unnecessary configurability.
- Prefer locally understandable logic.

Ask: would a strong engineer consider this overcomplicated? If yes, simplify it.

## Surgical Changes

Touch only what is required for the task.

- Do not refactor unrelated code unless asked.
- Do not rewrite adjacent comments, formatting, or structure without need.
- Match the existing style of the repository.
- Clean up only the unused code created by your own changes.

Every changed line should trace back to the task.

## Goal-Driven Execution

Turn tasks into verifiable outcomes.

- Define what success looks like before changing code.
- Prefer tests or checks when they are appropriate.
- For multi-step work, keep a short plan and verify each step.
- Do not stop at implementation; verify the result.

## Workflow Hygiene

- Before each `git add` and `git commit`, check whether `.gitignore` needs to be updated.
- After each meaningful implementation pass, check whether `PROJECT_LOG.md` should be updated.
- Keep generated arrays, logs, plots, and checkpoints out of source-oriented directories.

## Project-Specific Rules

- Keep `2du1` and `2du2` structurally symmetric unless a documented physics or workflow reason requires a difference.
- Put shared reusable Python code under `src/nthmc`.
- Put system-specific configs and outputs under `2du1` or `2du2`.
- Avoid recreating the old flat `*_evaluation` pattern at the repository root.
- Use `evaluation/base` as the canonical evaluation example. Add new variants under `2du*/evaluation/<variant>/` only when needed.
- Store generated outputs in the established `dumps`, `logs`, `plots`, or `artifacts` directories.
- Do not copy large model checkpoints, gauge arrays, notebook outputs, or generated plots into source directories.

## Documentation Maintenance

- Keep `SPEC.md` aligned with structure changes.
- Update `PROJECT_LOG.md` when a meaningful change is made.
- Keep README content concise and human-facing.
- If multilingual docs are added later, keep their structure aligned.
