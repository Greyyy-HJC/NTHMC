#!/usr/bin/env bash
set -euo pipefail

# Experiment settings. Keep train_beta in canonical float form because it is
# also used in ensemble, checkpoint, script, and log filenames.
lattice_size=16
train_beta="10.0"
beta_gap="0.5"
n_epochs=16
batch_size=64
n_subsets=8
model_tag="base"
lr="0.001"
max_grad_norm="5.0"
plateau_patience=2
early_stop_patience=16
loss_weights=(1.0 1.0 1.0 1.0)
seeds=(1029 1107 1331 1984 1999 2008 2017 2025)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONFIG_PATH="${SCRIPT_DIR}/../configs/links_L${lattice_size}_beta${train_beta}.npy"

if [[ ! "${model_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "model_tag must contain only letters, numbers, dots, underscores, or hyphens" >&2
    exit 1
fi
if [[ ! -f "${CONFIG_PATH}" && "${GENERATE_ONLY:-0}" != "1" ]]; then
    echo "Missing U(2) training ensemble: ${CONFIG_PATH}" >&2
    exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Warning: generating scripts without the missing ensemble ${CONFIG_PATH}" >&2
fi

mkdir -p "${SCRIPT_DIR}/scripts" "${SCRIPT_DIR}/logs"
generated_scripts=()

for seed in "${seeds[@]}"; do
    save_tag="${model_tag}_train_b${train_beta}_L${lattice_size}_${seed}"
    script_path="${SCRIPT_DIR}/scripts/sub_train_${model_tag}_L${lattice_size}_b${train_beta}_${seed}.sh"
    generated_scripts+=("${script_path}")
    cat > "${script_path}" <<EOF
#!/bin/bash -l

#PBS -N u2_t${lattice_size}_${seed}
#PBS -A fthmc
#PBS -l select=1:ngpus=4
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=04:00:00
#PBS -o ${SCRIPT_DIR}/logs/train_${model_tag}_L${lattice_size}_b${train_beta}_${seed}.log

set -euo pipefail
cd ${SCRIPT_DIR}
echo ">>> PBS_NODEFILE content:"
cat "\${PBS_NODEFILE}"
nvidia-smi
nvcc --version
date '+Start time: %Y-%m-%d %H:%M:%S'
export PYTHONPATH=${REPO_ROOT}/src:${REPO_ROOT}

${REPO_ROOT}/.venv/bin/python train.py \\
    --lattice_size ${lattice_size} \\
    --min_beta ${train_beta} \\
    --max_beta ${train_beta} \\
    --beta_gap ${beta_gap} \\
    --n_epochs ${n_epochs} \\
    --batch_size ${batch_size} \\
    --n_subsets ${n_subsets} \\
    --model_tag ${model_tag} \\
    --save_tag ${save_tag} \\
    --rand_seed ${seed} \\
    --lr ${lr} \\
    --max_grad_norm ${max_grad_norm} \\
    --plateau_patience ${plateau_patience} \\
    --early_stop_patience ${early_stop_patience} \\
    --loss_weights ${loss_weights[*]} \\
    --data_parallel \\
    --device cuda

date '+End time: %Y-%m-%d %H:%M:%S'
EOF
    chmod +x "${script_path}"
done

printf 'Generated %d U(2) training scripts in %s\n' "${#generated_scripts[@]}" "${SCRIPT_DIR}/scripts"
if [[ "${GENERATE_ONLY:-0}" == "1" ]]; then
    exit 0
fi
for script_path in "${generated_scripts[@]}"; do
    qsub "${script_path}"
done
