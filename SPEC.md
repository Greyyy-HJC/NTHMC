# SPEC.md

This file is the compact structural map for NTHMC.

## Project Shape

NTHMC studies neural transformed Hybrid Monte Carlo methods for:

- `2du1`: 2D U(1) lattice gauge theory
- `2du2`: 2D U(2) lattice gauge theory

The runtime is hybrid: JAX owns gauge generation, evaluation, and HMC, while PyTorch is the only model-training backend. Shared implementation belongs in `src/nthmc/core`; physics-specific JAX runtime and PyTorch training implementation live together under `src/nthmc/u1` and `src/nthmc/u2`. System-specific configs, generated outputs, logs, plots, and trained models stay under `2du1` and `2du2`.

## Top-Level Structure

```text
2du1/                  # U(1)-specific configs, workflows, and outputs
2du2/                  # U(2)-specific configs, workflows, and outputs
src/nthmc/core/        # Shared runtime, training, and checkpoint utilities
src/nthmc/u1/          # U(1) JAX runtime and PyTorch training implementation
src/nthmc/u2/          # U(2) JAX runtime and PyTorch training implementation
tests/                 # Focused JAX and PyTorch tests
presentation/          # Result presentation notebooks and plots
```

Both `2du1` and `2du2` use the same workflow skeleton:

```text
configs/               # Generated gauge arrays
gauge_generation/      # Gauge generation scripts plus dumps/logs/plots
model_training/        # PyTorch training scripts plus dumps/logs/plots
evaluation/base/       # Canonical FT-HMC evaluation
evaluation/hmc/        # Standard HMC evaluation
artifacts/models/      # PyTorch resume checkpoints and JAX inference exports
```

## Entry Points

- `2du*/gauge_generation/generate.py`: generate gauge configurations with JAX HMC.
- `2du*/model_training/train.py`: train field transforms with PyTorch eager autograd, save resumable `.pt`, and export the same weights as JAX-readable `.npz`; `--devices N` enables Fabric DDP and `--data_parallel` uses all visible devices.
- `2du*/evaluation/base/compare_fthmc.py`: evaluate a trained field transform with JAX FT-HMC.
- `2du*/evaluation/hmc/compare_hmc.py`: evaluate standard JAX HMC.
- `2du*/scripts/run_scaling.sh`: run gauge, training, HMC, and FT-HMC scaling workflows.

For GPU runs, install `jax[cuda12]`, `torch`, and `lightning`. Training defaults to one PyTorch device. Batch size is global across DDP ranks and must be divisible by the world size. Tail batches are copied to a fixed shape and excluded with a sample mask. Training inverse iterations use eager, differentiable, per-sample convergence checks and terminate when all valid samples converge or `inverse_max_iters` is reached. JAX field transforms expose runtime-only forward, inverse, logdet, and force operations.

U(1) stores generated configs as compact angle arrays `[N, 2, L, L]`. U(1) field transforms use the `base` CNN and exact analytic Jacobian log determinants.

U(2) stores generated configs as complex `2x2` matrices `[N, 2, L, L, 2, 2]`; training converts them to the internal split phase/quaternion representation. The base CNN uses the same compact scalar loop inputs as the pre-JAX implementation: 6 plaquette and 12 rectangle channels. U(2) FT-HMC uses JAX CNN loop coefficients, attached plaquette/rectangle loops, and exact analytic active-link Jacobian blocks; standard U(2) HMC, observables, group ops, and Wilson force are JAX-native.

## Important Rules

- Keep `2du1` and `2du2` structurally symmetric unless a documented physics reason requires a difference.
- Put shared implementation in `src/nthmc/core`; workflow scripts should be thin CLI wrappers.
- Store generated gauge arrays in `2du*/configs`.
- Store generated outputs in workflow-local `dumps`, `logs`, `plots`, or `artifacts`.
- Training writes `.pt` with optimizer/scheduler state and a matching canonical `.npz` for JAX evaluation. Continuation prefers `.pt` and can fall back to `.npz` weights with a fresh optimizer.
- `presentation/jax_benchmark_summary.md` keeps the historical pre-migration benchmark comparison.
