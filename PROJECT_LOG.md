# PROJECT_LOG.md

This is an append-only development history for NTHMC.

## 2026-04-27

- Initialized the repository scaffold for neural transformed HMC research.
- Chose a shared-core layout with reusable future code under `src/nthmc`.
- Created symmetric domain workspaces for `2du1` and `2du2`, corresponding to 2D U(1) and 2D U(2) lattice gauge theory.
- Kept only `evaluation/base` as the canonical evaluation example to avoid recreating the flat model-variant layout from `Scaling_FT_HMC`.
- Added required project initialization documents from `INIT.md`.
- Reproduced the necessary U(1) base-model pipeline from `Scaling_FT_HMC`: gauge generation, base neural field-transformation training, and FT-HMC evaluation.
- Added shared implementation under `src/nthmc` and thin U(1) workflow scripts under `2du1`.
- Kept U(2) as a documented structural workspace because the reference project does not include a U(2) implementation.
- Restored Lightning Fabric DDP support for base model training.
- Removed redundant `data`, `analysis`, and `scaling` directories from both `2du1` and `2du2`.
- Removed fallback handling for missing `tqdm` and `matplotlib`; these are required runtime dependencies.
- Moved generated gauge arrays to `2du1/configs` and added `.gitignore` rules for generated data, checkpoints, logs, plots, and caches.
- Removed unused `hmc_tune` directories from both `2du1` and `2du2`.
