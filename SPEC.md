# SPEC.md

This file is the compact structural map for NTHMC.

## Project Shape

NTHMC studies neural transformed Hybrid Monte Carlo methods for:

- `2du1`: 2D U(1) lattice gauge theory
- `2du2`: 2D U(2) lattice gauge theory

The repository uses a shared-core layout. Common implementation belongs in `src/nthmc`, while physics-system workspaces keep configs, generated outputs, logs, plots, and trained models separate.

## Top-Level Structure

```text
2du1/                  # U(1)-specific configs, workflows, and outputs
2du2/                  # U(2)-specific configs, workflows, and outputs
src/nthmc/common/      # Shared U(1) observables and plotting helpers
src/nthmc/workflows/   # Standard HMC and FT-HMC workflow classes
src/nthmc/models.py    # Base CNN model
src/nthmc/field_transform.py # Base neural field transformation
tests/                 # Future tests
notebooks/             # Future notebooks
```

## Domain Workspace Structure

Both `2du1` and `2du2` must keep the same skeleton:

```text
configs/               # System-specific configuration files and generated gauge arrays
gauge_generation/      # Gauge generation workflow outputs
model_training/        # Neural transformation training outputs
evaluation/base/       # Canonical base evaluation example
artifacts/models/      # Trained model artifacts
```

Workflow directories use `dumps`, `logs`, and `plots` subdirectories for generated outputs. Script directories are present only where shell or batch submission helpers are expected.

## Entry Points

Implemented U(1) base-model entry points:

- `2du1/gauge_generation/generate.py`: generate U(1) gauge configurations with standard HMC.
- `2du1/model_training/train.py`: train the base neural field transformation from generated gauges with Lightning Fabric DDP.
- `2du1/evaluation/base/compare_fthmc.py`: evaluate a trained base transformation with FT-HMC.

The U(2) workspace is structural only for now. Add U(2)-specific implementations only after the representation, action, observables, and model interface are defined.

## Important Rules

- Keep U(1) and U(2) workspace structure symmetric unless there is a documented reason not to.
- Do not add many top-level evaluation variants.
- Put model-specific evaluation variants under `2du*/evaluation/<variant>/`.
- Keep large generated files, checkpoints, logs, and plots out of source-oriented directories.
- Keep shared implementation in `src/nthmc`; workflow scripts should be thin CLI wrappers.
- Store generated gauge arrays in `2du*/configs`, not workflow `dumps` or a separate `data` tree.
