# NTHMC

NTHMC is a research repository for neural transformed Hybrid Monte Carlo methods in two-dimensional lattice gauge theory.

The project is initialized with two target systems:

- `2du1`: 2D U(1) lattice gauge theory
- `2du2`: 2D U(2) lattice gauge theory

This repository is organized as a cleaned-up successor structure inspired by `Scaling_FT_HMC`. The goal is to keep shared algorithmic code reusable while keeping system-specific experiments and outputs easy to find.

## Repository Layout

```text
NTHMC/
├── 2du1/                  # 2D U(1) project area
├── 2du2/                  # 2D U(2) project area
├── src/nthmc/             # Python package: core, u1, and u2 modules
├── pyproject.toml         # Editable-install package metadata
├── tests/                 # Future tests
├── notebooks/             # Future exploratory notebooks
├── README.md              # Human-facing overview
├── SPEC.md                # Structural project map
├── AGENTS.md              # Agent workflow rules
├── CLAUDE.md              # Lightweight agent entry point
└── PROJECT_LOG.md         # Append-only project history
```

Each physics area uses the same workflow skeleton:

```text
configs/
gauge_generation/{dumps,logs,plots}
model_training/{dumps,logs,plots}
evaluation/base/{dumps,logs,plots,scripts}
artifacts/models
```

Only `evaluation/base/` is included as the canonical evaluation example. Additional model variants should be added deliberately under `evaluation/` when they are needed, not copied as many top-level directories.

## Setup

The current runnable baseline covers the 2D U(1) base model pipeline. Install the lightweight Python dependencies with:

```bash
cd /eagle/fthmc/run/NTHMC
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For environments where editable installs are not available, `pip install -r requirements.txt` plus `PYTHONPATH=src` is still sufficient for direct source-tree imports.

## 2D U(1) Base Workflow

The implemented U(1) flow mirrors the necessary parts of `Scaling_FT_HMC`: gauge generation, base model training, and FT-HMC evaluation.

```bash
cd /eagle/fthmc/run/NTHMC/2du1/gauge_generation
python generate.py --lattice_size 8 --beta 3.0 --n_configs 128 --n_thermalization 50 --no_tune_step_size
```

```bash
cd /eagle/fthmc/run/NTHMC/2du1/model_training
torchrun --standalone --nproc_per_node=2 train.py --lattice_size 8 --min_beta 3.0 --max_beta 3.0 --beta_gap 1.0 --n_epochs 1 --batch_size 8 --if_identity_init
```

```bash
cd /eagle/fthmc/run/NTHMC/2du1/evaluation/base
python compare_fthmc.py --lattice_size 8 --beta 3.0 --train_beta 3.0 --n_configs 32 --n_thermalization 20 --save_tag base_train_b3.0_L8_1331 --no_tune_step_size
```

Generated gauge arrays live in `2du1/configs`, trained checkpoints live in `2du1/artifacts/models`, and plots/CSV diagnostics stay under workflow-local `plots` and `dumps` directories.

## Current Scope

Shared implementation should live in `src/nthmc/core`. U(1)-specific implementation lives in `src/nthmc/u1`, and `src/nthmc/u2` is reserved for future U(2)-specific code. System-specific configuration and outputs stay under `2du1` or `2du2`.

The U(2) workspace is present but not implemented yet; the reference project only provided the U(1) flow.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
