#!/usr/bin/env bash
set -euo pipefail

# Experiment settings. Keep train_beta in canonical float form because it is
# also used in ensemble, checkpoint, script, and log filenames.
lattice_size="${LATTICE_SIZE:-16}"
train_beta="${TRAIN_BETA:-3.0}"
beta_gap="${BETA_GAP:-0.5}"
n_epochs="${N_EPOCHS:-16}"
batch_size="${BATCH_SIZE:-64}"
n_subsets="${N_SUBSETS:-8}"
model_tag="${MODEL_TAG:-base}"
run_tag="${RUN_TAG:-${model_tag}}"
lr="${LR:-0.001}"
max_grad_norm="${MAX_GRAD_NORM:-5.0}"
edge_weight="${EDGE_WEIGHT:-0}"
edge_power="${EDGE_POWER:-2}"
edge_delta="${EDGE_DELTA:-0}"
walltime="${WALLTIME:-02:00:00}"
read -r -a seeds <<< "${SEEDS:-1029 1107 1331 1984 1999 2008 2017 2025}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONFIG_PATH="${SCRIPT_DIR}/../configs/theta_L${lattice_size}_beta${train_beta}.npy"

if [[ ! "${model_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "model_tag must contain only letters, numbers, dots, underscores, or hyphens" >&2
    exit 1
fi
if [[ ! "${run_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_TAG must contain only letters, numbers, dots, underscores, or hyphens" >&2
    exit 1
fi
if [[ "${run_tag}" == "${model_tag}" && "${GENERATE_ONLY:-0}" != "1" && "${FORCE_PRODUCTION:-0}" != "1" ]]; then
    echo "Refusing to submit production-tag U(1) training (RUN_TAG=${run_tag}). Set RUN_TAG to a feature tag, GENERATE_ONLY=1 to inspect, or FORCE_PRODUCTION=1 to override." >&2
    exit 1
fi
if [[ ! -f "${CONFIG_PATH}" && "${GENERATE_ONLY:-0}" != "1" ]]; then
    echo "Missing U(1) training ensemble: ${CONFIG_PATH}" >&2
    exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Warning: generating scripts without the missing ensemble ${CONFIG_PATH}" >&2
fi

mkdir -p "${SCRIPT_DIR}/scripts" "${SCRIPT_DIR}/logs"
generated_scripts=()

for seed in "${seeds[@]}"; do
    save_tag="${run_tag}_train_b${train_beta}_L${lattice_size}_${seed}"
    script_path="${SCRIPT_DIR}/scripts/sub_train_${run_tag}_L${lattice_size}_b${train_beta}_${seed}.sh"
    generated_scripts+=("${script_path}")
    cat > "${script_path}" <<EOF
#!/bin/bash -l

#PBS -N u1_t${lattice_size}_${seed}
#PBS -A fthmc
#PBS -l select=1:ngpus=4
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=${walltime}
#PBS -o ${SCRIPT_DIR}/logs/train_${run_tag}_L${lattice_size}_b${train_beta}_${seed}.log

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
    --edge_weight ${edge_weight} \\
    --edge_power ${edge_power} \\
    --edge_delta ${edge_delta} \\
    --data_parallel \\
    --device cuda

date '+End time: %Y-%m-%d %H:%M:%S'
EOF
    chmod +x "${script_path}"
done

printf 'Generated %d U(1) training scripts in %s\n' "${#generated_scripts[@]}" "${SCRIPT_DIR}/scripts"
if [[ "${GENERATE_ONLY:-0}" == "1" ]]; then
    exit 0
fi
for script_path in "${generated_scripts[@]}"; do
    qsub "${script_path}"
done
