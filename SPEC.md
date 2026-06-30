# SPEC.md

This file is the compact structural map for NTHMC.

## Project Shape

NTHMC studies neural transformed Hybrid Monte Carlo methods for:

- `2du1`: 2D U(1) lattice gauge theory
- `2du2`: 2D U(2) lattice gauge theory

The repository is now JAX-only. Shared implementation belongs in `src/nthmc/core`; physics-specific implementation belongs in `src/nthmc/u1` and `src/nthmc/u2`. System-specific configs, generated outputs, logs, plots, and trained models stay under `2du1` and `2du2`.

## Top-Level Structure

```text
2du1/                  # U(1)-specific configs, workflows, and outputs
2du2/                  # U(2)-specific configs, workflows, and outputs
src/nthmc/core/        # Shared JAX helpers and checkpoint utilities
src/nthmc/u1/          # U(1) observables, models, transforms, samplers
src/nthmc/u2/          # U(2) observables, models, transforms, samplers
tests/                 # Focused JAX tests
presentation/          # Result presentation notebooks and plots
```

Both `2du1` and `2du2` use the same workflow skeleton:

```text
configs/               # Generated gauge arrays
gauge_generation/      # Gauge generation scripts plus dumps/logs/plots
model_training/        # Optax training scripts plus dumps/logs/plots
evaluation/base/       # Canonical FT-HMC evaluation
evaluation/hmc/        # Standard HMC evaluation
artifacts/models/      # JAX .npz checkpoints
```

## Entry Points

- `2du*/gauge_generation/generate.py`: generate gauge configurations with JAX HMC.
- `2du*/model_training/train.py`: train JAX field-transform checkpoints with Optax and save `.npz`; `--data_parallel` enables single-node local-device `pmap` training.
- `2du*/evaluation/base/compare_fthmc.py`: evaluate a trained field transform with JAX FT-HMC.
- `2du*/evaluation/hmc/compare_hmc.py`: evaluate standard JAX HMC.
- `2du*/scripts/run_scaling.sh`: run gauge, training, HMC, and FT-HMC scaling workflows.

For GPU runs, install `jax[cuda12]` and `optax`. CLI `--device cuda` is accepted as an alias for JAX `gpu`; CPU remains available. Training defaults to one JAX device unless `--data_parallel` is set.

U(1) stores generated configs as compact angle arrays `[N, 2, L, L]`. U(1) field transforms use `base` and `addcos` CNN tags and exact analytic Jacobian log determinants.

U(2) stores generated configs as complex `2x2` matrices `[N, 2, L, L, 2, 2]`; training converts them to the internal split phase/quaternion representation. U(2) FT-HMC uses JAX CNN loop coefficients, attached plaquette/rectangle loops, and exact analytic active-link Jacobian blocks; standard U(2) HMC, observables, group ops, and Wilson force are JAX-native.

## Important Rules

- Keep `2du1` and `2du2` structurally symmetric unless a documented physics reason requires a difference.
- Put shared implementation in `src/nthmc/core`; workflow scripts should be thin CLI wrappers.
- Store generated gauge arrays in `2du*/configs`.
- Store generated outputs in workflow-local `dumps`, `logs`, `plots`, or `artifacts`.
- JAX checkpoints use `.npz`; older incompatible model checkpoints are not supported.
- `presentation/jax_benchmark_summary.md` keeps the historical pre-migration benchmark comparison.
