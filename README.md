# NTHMC

NTHMC is a research repository for neural transformed Hybrid Monte Carlo methods in two-dimensional lattice gauge theory.

The active codebase is JAX-only and covers:

- `2du1`: 2D U(1) lattice gauge theory
- `2du2`: 2D U(2) lattice gauge theory

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For direct source-tree use without an editable install:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

The default GPU target is a single NVIDIA GPU through `jax[cuda12]`; CPU runs are supported with `--device cpu`.

## Layout

```text
2du1/                  # U(1) configs, workflows, outputs
2du2/                  # U(2) configs, workflows, outputs
src/nthmc/core/        # Shared JAX helpers
src/nthmc/u1/          # U(1) observables, models, transforms, samplers
src/nthmc/u2/          # U(2) observables, models, transforms, samplers
tests/                 # Focused JAX tests
presentation/          # Presentation notebooks and benchmark notes
```

Each physics workspace uses:

```text
configs/
gauge_generation/{dumps,logs,plots}
model_training/{dumps,logs,plots}
evaluation/base/{dumps,logs,plots,scripts}
evaluation/hmc/{dumps,logs,plots}
artifacts/models
```

## U(1) Quick Run

```bash
cd 2du1/gauge_generation
../../.venv/bin/python generate.py --lattice_size 4 --beta 3.0 --n_configs 4 --n_thermalization 1 --n_steps 1 --no_tune_step_size --device cpu
```

```bash
cd ../model_training
../../.venv/bin/python train.py --lattice_size 4 --min_beta 3.0 --max_beta 3.0 --beta_gap 1.0 --n_epochs 1 --batch_size 2 --model_tag base --save_tag smoke --device cpu
```

```bash
cd ../evaluation/base
../../../.venv/bin/python compare_fthmc.py --lattice_size 4 --beta 3.0 --train_beta 3.0 --n_configs 2 --n_thermalization 1 --n_steps 1 --save_tag smoke --no_tune_step_size --device cpu
```

## U(2) Quick Run

```bash
cd 2du2/gauge_generation
../../.venv/bin/python generate.py --lattice_size 3 --beta 2.0 --n_configs 4 --n_thermalization 1 --n_steps 1 --no_tune_step_size --device cpu
```

```bash
cd ../model_training
../../.venv/bin/python train.py --lattice_size 3 --min_beta 2.0 --max_beta 2.0 --beta_gap 1.0 --n_epochs 1 --batch_size 2 --model_tag base --save_tag smoke --device cpu
```

```bash
cd ../evaluation/base
../../../.venv/bin/python compare_fthmc.py --lattice_size 3 --beta 2.0 --train_beta 2.0 --n_configs 2 --n_thermalization 1 --n_steps 1 --save_tag smoke --no_tune_step_size --device cpu
```

Generated configs live under `2du*/configs`; JAX checkpoints live under `2du*/artifacts/models` as `.npz`; workflow diagnostics stay in local `dumps` and `plots`.

## Notes

- `presentation/jax_benchmark_summary.md` preserves the historical benchmark comparison used to justify the migration.
- `presentation/Field_transform.html` keeps the browser-friendly field-transform derivation.
- See `SPEC.md` for the current structural map.
