# PROJECT_LOG.md

This is an append-only development history for NTHMC.

## 2026-04-27

- Initialized the repository scaffold for neural transformed HMC research.
- Added required project initialization documents from `INIT.md`.
- Set up symmetric `2du1` and `2du2` workspaces while keeping U(2) as a structural placeholder.
- Reproduced the U(1) gauge generation, neural field-transformation training, and FT-HMC evaluation pipeline.
- Reorganized reusable Python code into `nthmc.core`, `nthmc.u1`, and `nthmc.u2`, with editable-install metadata in `pyproject.toml`.
- Added the U(1) add-cos optimized transformation path and corresponding model-training/evaluation entry points.
- Kept generated arrays, checkpoints, logs, plots, and diagnostics out of source-oriented directories with `.gitignore` coverage.
