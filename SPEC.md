# SPEC.md

This file is the compact structural map for NTHMC.

## Project Shape

NTHMC studies neural transformed Hybrid Monte Carlo methods for:

- `2du1`: 2D U(1) lattice gauge theory
- `2du2`: 2D U(2) lattice gauge theory

The repository uses a shared-core layout. Common implementation belongs in `src/nthmc/core`, while physics-system workspaces keep configs, generated outputs, logs, plots, and trained models separate. Python package names use `u1` and `u2` for importability; they correspond to the top-level `2du1` and `2du2` workspaces.

## Top-Level Structure

```text
2du1/                  # U(1)-specific configs, workflows, and outputs
2du2/                  # U(2)-specific configs, workflows, and outputs
src/nthmc/core/        # Shared helpers independent of a physics system
src/nthmc/u1/          # U(1) observables, models, transformations, and samplers
src/nthmc/u2/          # U(2) observables and samplers
pyproject.toml         # Setuptools package metadata for editable installs
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

Implemented U(2) entry points:

- `2du2/gauge_generation/generate.py`: generate U(2) gauge configurations with standard HMC.

The U(2) implementation currently covers gauge generation only. Internally it stores links as a U(1) phase plus an SU(2) unit quaternion and exports generated configs as complex `2x2` matrices. Model training and FT-HMC evaluation remain future work.

## Important Rules

- Keep U(1) and U(2) workspace structure symmetric unless there is a documented reason not to.
- Do not add many top-level evaluation variants.
- Put model-specific evaluation variants under `2du*/evaluation/<variant>/`.
- Keep large generated files, checkpoints, logs, and plots out of source-oriented directories.
- Keep shared implementation in `src/nthmc/core`; workflow scripts should be thin CLI wrappers.
- Store generated gauge arrays in `2du*/configs`, not workflow `dumps` or a separate `data` tree.
