#!/usr/bin/env bash
set -euo pipefail

# Match 2du2/evaluation/base production FT-HMC: train/eval β=10, seed 1107.
# L32/L64 can reuse the L16 checkpoint because CNN weights are lattice-independent.
lattice_size="${LATTICE_SIZE:-16}"
model_tag="${MODEL_TAG:-base}"
train_beta="${TRAIN_BETA:-10.0}"
evaluate_beta="${EVALUATE_BETA:-10.0}"
n_configs="${N_CONFIGS:-2048}"
n_thermalization="${N_THERMALIZATION:-200}"
n_steps="${N_STEPS:-10}"
ft_step_size="${FT_STEP_SIZE:-0.10}"
hmc_step_size="${HMC_STEP_SIZE:-0.10}"
n_profile_steps="${N_PROFILE_STEPS:-32}"
n_kernel_repeats="${N_KERNEL_REPEATS:-8}"
seed="${SEED:-1107}"
dump_suffix="${DUMP_SUFFIX:-}"
if (( lattice_size >= 64 )); then
    walltime="${WALLTIME:-12:00:00}"
elif (( lattice_size >= 32 )); then
    walltime="${WALLTIME:-06:00:00}"
else
    walltime="${WALLTIME:-02:00:00}"
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
MODEL_DIR="${REPO_ROOT}/2du2/artifacts/models"

if [[ ! "${model_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "model_tag must contain only letters, numbers, dots, underscores, or hyphens" >&2
    exit 1
fi

native_save_tag="${model_tag}_train_b${train_beta}_L${lattice_size}_${seed}"
fallback_lattice="${WEIGHT_LATTICE_SIZE:-16}"
fallback_seed="${WEIGHT_SEED:-${seed}}"
fallback_save_tag="${model_tag}_train_b${train_beta}_L${fallback_lattice}_${fallback_seed}"
if [[ -n "${SAVE_TAG:-}" ]]; then
    save_tag="${SAVE_TAG}"
elif [[ -f "${MODEL_DIR}/best_model_train_beta${train_beta}_${native_save_tag}.npz" ]]; then
    save_tag="${native_save_tag}"
else
    save_tag="${fallback_save_tag}"
fi
checkpoint="${MODEL_DIR}/best_model_train_beta${train_beta}_${save_tag}.npz"
if [[ ! -f "${checkpoint}" && "${GENERATE_ONLY:-0}" != "1" ]]; then
    echo "Missing U(2) checkpoint: ${checkpoint}" >&2
    exit 1
fi
if [[ ! -f "${checkpoint}" ]]; then
    echo "Warning: generating script without the missing checkpoint ${checkpoint}" >&2
fi

mkdir -p "${SCRIPT_DIR}/scripts" "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/dumps" "${SCRIPT_DIR}/plots"
run_tag="${model_tag}_L${lattice_size}_train_b${train_beta}_eval_b${evaluate_beta}_${seed}"
if [[ -n "${dump_suffix}" ]]; then
    run_tag="${run_tag}_${dump_suffix}"
fi
script_path="${SCRIPT_DIR}/scripts/sub_profile_${run_tag}.sh"
job_beta="${evaluate_beta%%.*}"

cat > "${script_path}" <<EOF
#!/bin/bash -l

#PBS -N u2_bn${lattice_size}b${job_beta}_${seed}
#PBS -A fthmc
#PBS -l select=1:ngpus=1
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=${walltime}
#PBS -o ${SCRIPT_DIR}/logs/profile_${run_tag}.log

set -euo pipefail
cd ${SCRIPT_DIR}
echo ">>> PBS_NODEFILE content:"
cat "\${PBS_NODEFILE}"
nvidia-smi
nvcc --version
date '+Start time: %Y-%m-%d %H:%M:%S'
export PYTHONPATH=${REPO_ROOT}/src:${REPO_ROOT}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

${REPO_ROOT}/.venv/bin/python profile_inference.py \\
    --lattice_size ${lattice_size} \\
    --beta ${evaluate_beta} \\
    --train_beta ${train_beta} \\
    --n_configs ${n_configs} \\
    --n_thermalization ${n_thermalization} \\
    --n_steps ${n_steps} \\
    --ft_step_size ${ft_step_size} \\
    --hmc_step_size ${hmc_step_size} \\
    --n_profile_steps ${n_profile_steps} \\
    --n_kernel_repeats ${n_kernel_repeats} \\
    --rand_seed ${seed} \\
    --model_tag ${model_tag} \\
    --save_tag ${save_tag} \\
    --device cuda \\
    --dump_suffix "${dump_suffix}"

date '+End time: %Y-%m-%d %H:%M:%S'
EOF
chmod +x "${script_path}"

printf 'Generated U(2) bottleneck profile script %s\n' "${script_path}"
printf 'Using checkpoint save_tag %s\n' "${save_tag}"
if [[ "${GENERATE_ONLY:-0}" == "1" ]]; then
    exit 0
fi
qsub "${script_path}"
