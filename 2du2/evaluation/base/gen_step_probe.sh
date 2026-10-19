#!/usr/bin/env bash
set -euo pipefail

# One short step-size probe per evaluation beta. Production FT-HMC reads the
# written step and runs with tuning disabled.
lattice_size="${LATTICE_SIZE:-32}"
model_tag="${MODEL_TAG:-base}"
train_beta="${TRAIN_BETA:-14.0}"
train_lattice="${TRAIN_LATTICE_SIZE:-32}"
probe_seed="${PROBE_SEED:-1029}"
n_steps="${N_STEPS:-10}"
ft_step_size="${FT_STEP_SIZE:-0.10}"
n_tune_steps="${N_TUNE_STEPS:-200}"
walltime="${WALLTIME:-02:00:00}"
read -r -a evaluate_betas <<< "${EVALUATE_BETAS:-14.0 16.0 18.0 20.0}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
save_tag="${model_tag}_train_b${train_beta}_L${train_lattice}_${probe_seed}"

mkdir -p "${SCRIPT_DIR}/scripts" "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/dumps"
generated_scripts=()
for evaluate_beta in "${evaluate_betas[@]}"; do
    job_beta="${evaluate_beta%%.*}"
    step_file="${SCRIPT_DIR}/dumps/ft_step_L${lattice_size}_beta${evaluate_beta}_s${probe_seed}.txt"
    script_path="${SCRIPT_DIR}/scripts/sub_step_L${lattice_size}_b${evaluate_beta}_s${probe_seed}.sh"
    generated_scripts+=("${script_path}")
    cat > "${script_path}" <<EOF
#!/bin/bash -l

#PBS -N u2_p${lattice_size}b${job_beta}
#PBS -A fthmc
#PBS -l select=1:ngpus=1
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=${walltime}
#PBS -o ${SCRIPT_DIR}/logs/step_L${lattice_size}_b${evaluate_beta}_s${probe_seed}.log

set -euo pipefail
cd ${SCRIPT_DIR}
date '+Start time: %Y-%m-%d %H:%M:%S'
export PYTHONPATH=${REPO_ROOT}/src:${REPO_ROOT}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

${REPO_ROOT}/.venv/bin/python tune_ft_step.py \\
    --lattice_size ${lattice_size} \\
    --beta ${evaluate_beta} \\
    --train_beta ${train_beta} \\
    --n_steps ${n_steps} \\
    --ft_step_size ${ft_step_size} \\
    --n_tune_steps ${n_tune_steps} \\
    --rand_seed ${probe_seed} \\
    --model_tag ${model_tag} \\
    --save_tag ${save_tag} \\
    --output ${step_file} \\
    --device cuda

date '+End time: %Y-%m-%d %H:%M:%S'
EOF
    chmod +x "${script_path}"
done

printf 'Generated %d U(2) FT step probes in %s\n' "${#generated_scripts[@]}" "${SCRIPT_DIR}/scripts"
if [[ "${GENERATE_ONLY:-0}" == "1" ]]; then
    exit 0
fi
for script_path in "${generated_scripts[@]}"; do
    qsub "${script_path}"
done
