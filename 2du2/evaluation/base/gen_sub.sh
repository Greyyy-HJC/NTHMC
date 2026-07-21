#!/usr/bin/env bash
set -euo pipefail

# Experiment settings. Keep beta values in canonical float form because they
# are also used in checkpoint, script, and log filenames.
lattice_size=16
model_tag="base"
train_beta="10.0"
evaluate_betas=("10.0" "12.0" "14.0" "16.0")
n_configs=2048
n_thermalization=200
n_steps=10
ft_step_size="0.10"
n_tune_steps=1000
max_lag=64
seeds=(1029 1107 1331 1984 1999 2008 2017 2025)
tune_step_size=false

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
MODEL_DIR="${SCRIPT_DIR}/../../artifacts/models"

if [[ ! "${model_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "model_tag must contain only letters, numbers, dots, underscores, or hyphens" >&2
    exit 1
fi
case "${tune_step_size}" in
    true) tune_flag_line="" ;;
    false) tune_flag_line="    --no_tune_step_size \\"$'\n' ;;
    *) echo "tune_step_size must be true or false" >&2; exit 1 ;;
esac

for seed in "${seeds[@]}"; do
    save_tag="${model_tag}_train_b${train_beta}_L${lattice_size}_${seed}"
    checkpoint="${MODEL_DIR}/best_model_train_beta${train_beta}_${save_tag}.npz"
    if [[ ! -f "${checkpoint}" && "${GENERATE_ONLY:-0}" != "1" ]]; then
        echo "Missing U(2) checkpoint: ${checkpoint}" >&2
        exit 1
    fi
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Warning: generating script without the missing checkpoint ${checkpoint}" >&2
    fi
done

mkdir -p "${SCRIPT_DIR}/scripts" "${SCRIPT_DIR}/logs"
generated_scripts=()
for evaluate_beta in "${evaluate_betas[@]}"; do
    job_beta="${evaluate_beta%%.*}"
    for seed in "${seeds[@]}"; do
        save_tag="${model_tag}_train_b${train_beta}_L${lattice_size}_${seed}"
        run_tag="${model_tag}_L${lattice_size}_train_b${train_beta}_eval_b${evaluate_beta}_${seed}"
        script_path="${SCRIPT_DIR}/scripts/sub_fthmc_${run_tag}.sh"
        generated_scripts+=("${script_path}")
        cat > "${script_path}" <<EOF
#!/bin/bash -l

#PBS -N u2_f${lattice_size}b${job_beta}_${seed}
#PBS -A fthmc
#PBS -l select=1:ngpus=1
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=06:00:00
#PBS -o ${SCRIPT_DIR}/logs/fthmc_${run_tag}.log

set -euo pipefail
cd ${SCRIPT_DIR}
echo ">>> PBS_NODEFILE content:"
cat "\${PBS_NODEFILE}"
nvidia-smi
nvcc --version
date '+Start time: %Y-%m-%d %H:%M:%S'
export PYTHONPATH=${REPO_ROOT}/src:${REPO_ROOT}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

${REPO_ROOT}/.venv/bin/python compare_fthmc.py \\
    --lattice_size ${lattice_size} \\
    --beta ${evaluate_beta} \\
    --train_beta ${train_beta} \\
    --n_configs ${n_configs} \\
    --n_thermalization ${n_thermalization} \\
    --n_steps ${n_steps} \\
    --ft_step_size ${ft_step_size} \\
    --n_tune_steps ${n_tune_steps} \\
    --max_lag ${max_lag} \\
    --rand_seed ${seed} \\
    --model_tag ${model_tag} \\
    --save_tag ${save_tag} \\
${tune_flag_line}    --device cuda

date '+End time: %Y-%m-%d %H:%M:%S'
EOF
        chmod +x "${script_path}"
    done
done

printf 'Generated %d U(2) FT-HMC scripts in %s\n' "${#generated_scripts[@]}" "${SCRIPT_DIR}/scripts"
if [[ "${GENERATE_ONLY:-0}" == "1" ]]; then
    exit 0
fi
for script_path in "${generated_scripts[@]}"; do
    qsub "${script_path}"
done
