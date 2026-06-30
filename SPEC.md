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
src/nthmc/u2/          # U(2) observables, models, transformations, and samplers
pyproject.toml         # Setuptools package metadata for editable installs
tests/                 # Future tests
presentation/          # Result presentation notebooks and plots
```

## Domain Workspace Structure

Both `2du1` and `2du2` must keep the same skeleton:

```text
configs/               # System-specific configuration files and generated gauge arrays
gauge_generation/      # Gauge generation workflow outputs
model_training/        # Neural transformation training outputs
evaluation/base/       # Canonical base evaluation example
evaluation/hmc/        # Standard HMC baseline evaluation
artifacts/models/      # Trained model artifacts
```

Workflow directories use `dumps`, `logs`, and `plots` subdirectories for generated outputs. Script directories are present only where shell or batch submission helpers are expected.

## Entry Points

Implemented U(1) base-model entry points:

- `2du1/gauge_generation/generate.py`: generate U(1) gauge configurations with standard HMC.
- `2du1/model_training/train.py`: train the base neural field transformation from generated gauges with Lightning Fabric DDP.
- `2du1/evaluation/base/compare_fthmc.py`: evaluate a trained base transformation with PyTorch FT-HMC and write benchmark JSON.
- `2du1/evaluation/jax/compare_fthmc.py`: evaluate a trained base/addcos transformation with the experimental JAX FT-HMC backend and write benchmark JSON.
- `2du1/evaluation/hmc/compare_hmc.py`: evaluate standard HMC without a field transformation.
- `2du1/scripts/run_scaling.sh`: pre-run the U(1) scaling workflow before presentation analysis; it reuses existing 2048-config training ensembles and completed 16-epoch scaling checkpoints by default.

Implemented U(2) entry points:

- `2du2/gauge_generation/generate.py`: generate U(2) gauge configurations with standard HMC.
- `2du2/model_training/train.py`: train the base U(2) neural field transformation from generated gauges with Lightning Fabric DDP.
- `2du2/evaluation/base/compare_fthmc.py`: evaluate a trained U(2) base transformation with FT-HMC.
- `2du2/evaluation/hmc/compare_hmc.py`: evaluate standard U(2) HMC without a field transformation.
- `2du2/scripts/run_scaling.sh`: pre-run the U(2) base scaling workflow before presentation analysis; it mirrors the U(1) scaling grid and supports resumable gauge, training, and evaluation stages.

The experimental U(1) JAX backend lives in `src/nthmc/u1/jax_backend.py`. It converts frozen PyTorch U(1) checkpoints to JAX arrays at evaluation time, keeps training and checkpoint format unchanged, and compiles the FT-HMC thermalization/run chain with JAX. For GPU runs, install the CUDA JAX extra matching the environment, e.g. `jax[cuda12]` on the current CUDA-12-compatible setup; `2du1/evaluation/jax/compare_fthmc.py` bootstraps NVIDIA wheel library paths before importing JAX. This path is intentionally U(1)-only until 2du1 benchmarks show a clear steady-state speedup.

The U(2) implementation internally stores links as a U(1) phase plus an SU(2) unit quaternion and exports generated configs as complex `2x2` matrices. Training converts those matrix configs back to the split representation on load. Gauge links use the site-based convention `U^G_{x,mu} = G_x U_{x,mu} G^dagger_{x+mu}`. The base U(2) field transformation feeds gauge-invariant scalar plaquette and rectangle loop features to the CNN, learns four coefficient slots per attached loop, and builds the attached non-Abelian loop stack in the active link's site-`x` frame so central and traceless updates are gauge covariant. It evaluates the non-volume-preserving Jacobian as exact `4x4` active-link tangent blocks. Checkpoints from earlier U(2) base model definitions are not compatible with this transform. U(2) training exposes `--if_check_jac` for small manual-vs-autograd Jacobian checks; `--if_compile` is optional and cannot be combined with Jacobian checking. FT-HMC evaluation exposes an explicit evaluation-only compile path that freezes models and uses compiled callables only for force calculations.

Shared analysis helpers:

- `src/nthmc/core/plot_settings.py`: reusable matplotlib style settings.
- `src/nthmc/core/resampling.py`: bootstrap and jackknife helpers used by presentation notebooks.
- `presentation/jax_optimization.ipynb`: self-contained U(1) PyTorch-vs-JAX consistency and force-path benchmark notebook.
- `presentation/jax_benchmark_summary.md`: compact English summary of the 2du1 FT-HMC evaluation benchmark and a small 2du2 U(2) Wilson-force probe.

## Important Rules

- Keep U(1) and U(2) workspace structure symmetric unless there is a documented reason not to.
- Do not add many top-level evaluation variants.
- Put model-specific evaluation variants under `2du*/evaluation/<variant>/`.
- Use `evaluation/hmc` for standard HMC baselines and `evaluation/base` for base FT-HMC.
- Keep large generated files, checkpoints, logs, and plots out of source-oriented directories.
- Keep shared implementation in `src/nthmc/core`; workflow scripts should be thin CLI wrappers.
- Store generated gauge arrays in `2du*/configs`, not workflow `dumps` or a separate `data` tree.
