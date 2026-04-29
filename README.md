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
├── presentation/          # Result presentation notebooks and plots
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
evaluation/hmc/{dumps,logs,plots}
artifacts/models
```

Only `evaluation/base/` is included as the canonical evaluation example. Additional model variants should be added deliberately under `evaluation/` when they are needed, not copied as many top-level directories.

## Setup

The current runnable baselines cover the 2D U(1) and U(2) base model pipelines. Install the lightweight Python dependencies with:

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

For U(1), links are stored as compact angles `theta_{x,mu}`. The local plaquette angle is

```text
theta_{x,01} = theta_{x,0} - theta_{x,1} - theta_{x+hat{1},0} + theta_{x+hat{0},1},
```

wrapped to `[-pi, pi)`. Diagnostics use the local plaquette

```text
p_x = cos(theta_{x,01})
```

and its volume average `<p> = (1 / V) sum_x p_x`. The HMC implementation uses the Wilson action without the additive constant,

```text
S_E = - beta * sum_x cos(theta_{x,01}).
```

This differs from `beta * sum_x (1 - cos(theta_{x,01}))` only by the constant `beta * V`, so it gives the same HMC dynamics and Metropolis decisions. The infinite-volume 2D U(1) theoretical plaquette is

```text
<p> = I_1(beta) / I_0(beta).
```

## 2D U(2) Base Workflow

The implemented U(2) path covers standard HMC gauge generation, base model training, and FT-HMC evaluation. Internally each link is represented as

```text
U_{x,mu} = exp(i phi_{x,mu}) S_{x,mu},  S_{x,mu} in SU(2),
```

where `phi` is a real phase angle and `S` is stored as an SU(2) unit quaternion. Generated configurations are exported as complex `2x2` matrices.

For a plaquette matrix `P_{x,01}`, the local normalized real plaquette is

```text
p_x = (1 / 2) ReTr(P_{x,01}).
```

Diagnostics record the volume average

```text
<p> = (1 / V) sum_x p_x.
```

The HMC action is the Wilson gauge action

```text
S_E = (beta / N_c) sum_x ReTr(1 - P_{x,01}),  N_c = 2
    = beta * sum_x (1 - p_x)
    = beta * V * (1 - <p>).
```

The plotted theoretical plaquette is the 2D U(2) one-plaquette result for this normalization. With `x = beta / 2` and modified Bessel functions `I_n`,

```text
Z_U2 = I_0(x)^2 - I_1(x)^2
<p> = I_1(x) * (I_0(x) - I_2(x)) / (2 * Z_U2).
```

The base U(2) field transformation uses plaquette and rectangle loop terms projected into trace/traceless sin-like and cos-like U(2) algebra components. It is not volume preserving: the FT-HMC action includes an exact Jacobian computed as `4x4` active-link tangent blocks. Older U(2) base checkpoints from the previous volume-preserving transform are not compatible and need to be retrained.

U(2) training runs in eager mode by default. Add `--if_check_jac` for small diagnostic runs that compare the manual active-link Jacobian with an autograd Jacobian, and add `--if_compile` only when the local PyTorch backend benefits from `torch.compile`.

```bash
cd /eagle/fthmc/run/NTHMC/2du2/gauge_generation
python generate.py --lattice_size 8 --beta 3.0 --n_configs 32 --n_thermalization 20 --n_steps 4 --no_tune_step_size
```

```bash
cd /eagle/fthmc/run/NTHMC/2du2/model_training
torchrun --standalone --nproc_per_node=1 train.py --lattice_size 8 --min_beta 3.0 --max_beta 3.0 --beta_gap 1.0 --n_epochs 1 --batch_size 8 --if_identity_init
```

```bash
cd /eagle/fthmc/run/NTHMC/2du2/evaluation/base
python compare_fthmc.py --lattice_size 8 --beta 3.0 --train_beta 3.0 --n_configs 32 --n_thermalization 20 --n_steps 4 --save_tag base_train_b3.0_L8_1331 --no_tune_step_size
```

Generated U(2) gauge arrays live in `2du2/configs` with shape `[N, 2, L, L, 2, 2]`. Training converts those matrices to the internal split phase/quaternion representation on load. Trained checkpoints live in `2du2/artifacts/models`, and plots/CSV diagnostics stay under workflow-local `plots` and `dumps` directories.

## Current Scope

Shared implementation should live in `src/nthmc/core`. U(1)-specific implementation lives in `src/nthmc/u1`, and U(2)-specific implementation lives in `src/nthmc/u2`. System-specific configuration and outputs stay under `2du1` or `2du2`.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
