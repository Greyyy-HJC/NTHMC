# NTHMC

NTHMC is a research repository for neural transformed Hybrid Monte Carlo methods in
two-dimensional lattice gauge theory:

- `2du1`: 2D U(1) lattice gauge theory
- `2du2`: 2D U(2) lattice gauge theory

The active pipeline is hybrid. JAX handles gauge generation, standard HMC, and FT-HMC
evaluation; PyTorch is the training backend and exports both resumable `.pt` checkpoints
and JAX-readable `.npz` parameters.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The default GPU target is a single NVIDIA GPU through `jax[cuda12]`. CPU runs are
supported with `--device cpu`. Repository Python commands and tests should use `.venv`.

## Layout

```text
2du1/                         # U(1) workflow entrypoints and generated-output skeleton
2du2/                         # U(2) workflow entrypoints and generated-output skeleton
src/nthmc/core/               # Shared runtime, checkpoint, plotting, and training helpers
src/nthmc/u1/                 # U(1) JAX runtime and PyTorch training implementation
src/nthmc/u2/                 # U(2) JAX runtime and PyTorch training implementation
tests/                        # Focused CPU-default tests
presentation/jax_benchmark/   # Archived full-pipeline benchmark evidence
presentation/docs/            # Field-transform derivation and reference material
```

Both physics workspaces use the same active structure:

```text
configs/
gauge_generation/{generate.py,gen_sub.sh,dumps,logs,plots,scripts}
model_training/{train.py,gen_sub.sh,dumps,logs,plots,scripts}
evaluation/base/{compare_fthmc.py,gen_sub.sh,dumps,logs,plots,scripts}
evaluation/hmc/{compare_hmc.py,gen_sub.sh,dumps,logs,plots,scripts}
artifacts/models/
scripts/run_scaling.sh
```

Gauge arrays, checkpoints, logs, and generated PBS `sub*.sh` files stay local. Plot PDFs
and evaluation CSV/JSON dumps are synchronized; `.gitkeep` files preserve empty output
directories.

## Quick Run

The commands below show the U(1) flow. Replace `2du1` with `2du2` and use a suitable
U(2) lattice/beta choice for the corresponding pipeline.

```bash
cd 2du1/gauge_generation
../../.venv/bin/python generate.py \
  --lattice_size 4 --beta 3.0 --n_configs 4 --n_thermalization 1 \
  --n_steps 1 --no_tune_step_size --device cpu
```

```bash
cd ../model_training
../../.venv/bin/python train.py \
  --lattice_size 4 --min_beta 3.0 --max_beta 3.0 --beta_gap 1.0 \
  --n_epochs 1 --batch_size 2 --model_tag base --save_tag smoke --device cpu
```

```bash
cd ../evaluation/base
../../../.venv/bin/python compare_fthmc.py \
  --lattice_size 4 --beta 3.0 --train_beta 3.0 --n_configs 2 \
  --n_thermalization 1 --n_steps 1 --model_tag base --save_tag smoke \
  --no_tune_step_size --device cpu
```

Standard HMC uses `evaluation/hmc/compare_hmc.py`. The symmetric scaling drivers under
`2du*/scripts/run_scaling.sh` run the complete gauge/training/HMC/FT-HMC workflow.

## L16 PBS Production

The `2du1` production point is `L=16, beta=3.0`; `2du2` trains at `L=16, beta=10.0`
and evaluates at beta 10, 12, 14, and 16. Gauge generation produces 4096 configurations,
and eight training seeds feed matching 2048-sample HMC and FT-HMC evaluations. Gauge
generation and standard HMC tune their step sizes automatically toward acceptance 0.70.
FT-HMC uses fixed step sizes (`0.35` for U(1), `0.10` for U(2)) because transformed-force
tuning is expensive.

Each `gen_sub.sh` starts with one experiment-settings block. Change values there; command
arguments, checkpoint/save tags, generated script names, and log paths are derived from
the same variables. FT-HMC script and log names include both training and evaluation beta
so different combinations do not overwrite each other.

Submit stages manually in this order:

1. Run `gauge_generation/gen_sub.sh` for each system.
2. After the gauge ensemble exists, run `model_training/gen_sub.sh`.
3. Run `evaluation/hmc/gen_sub.sh` independently.
4. After all checkpoints exist, run `evaluation/base/gen_sub.sh`.

Gauge generation produces one job; the other generators submit one job per configured
seed and evaluation beta. Set `GENERATE_ONLY=1` to inspect the generated
`scripts/sub_*.sh` files without calling `qsub`.

## Models and Checkpoints

`model_tag` remains part of the training and FT-HMC interfaces so new architectures can be
added without changing the workflow. The current JAX and PyTorch `choose_model` registries
contain only `base`; unknown tags fail explicitly. A new model must be implemented and
registered on both sides so PyTorch exports remain loadable by JAX evaluation.

Checkpoint filenames and metadata include `model_tag`. Training continuation prefers the
`.pt` checkpoint with optimizer state, while evaluation loads the matching `.npz` export.

## Benchmarks and Documentation

- [`presentation/jax_benchmark/jax_benchmark_summary.md`](presentation/jax_benchmark/jax_benchmark_summary.md) summarizes the final U(1)/U(2) historical-vs-current pipelines.
- Each benchmark directory contains final logs/dumps, provenance, checksums, and a
  `summarize.py` script that regenerates its summary and timing CSV.
- [`presentation/docs/Field_transform.html`](presentation/docs/Field_transform.html) is the browser-friendly field-transform derivation.
