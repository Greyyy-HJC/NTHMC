#!/usr/bin/env bash
set -euo pipefail

# Experiment settings. Keep beta in canonical float form because it is also
# used in generated filenames.
lattice_size=16
evaluate_beta="10.0"
n_configs=2048
n_thermalization=200
n_steps=10
step_size="0.10"
n_tune_steps=1000
max_lag=64
seeds=(1029 1107 1331 1984 1999 2008 2017 2025)
tune_step_size=true

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

case "${tune_step_size}" in
    true) tune_flag_line="" ;;
    false) tune_flag_line="    --no_tune_step_size \\"$'\n' ;;
    *) echo "tune_step_size must be true or false" >&2; exit 1 ;;
esac

mkdir -p "${SCRIPT_DIR}/scripts" "${SCRIPT_DIR}/logs"
generated_scripts=()

for seed in "${seeds[@]}"; do
    script_path="${SCRIPT_DIR}/scripts/sub_hmc_L${lattice_size}_b${evaluate_beta}_${seed}.sh"
    generated_scripts+=("${script_path}")
    cat > "${script_path}" <<EOF
#!/bin/bash -l

#PBS -N u2_h${lattice_size}_${seed}
#PBS -A fthmc
#PBS -l select=1:ngpus=1
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=01:00:00
#PBS -o ${SCRIPT_DIR}/logs/hmc_L${lattice_size}_b${evaluate_beta}_${seed}.log

set -euo pipefail
cd ${SCRIPT_DIR}
echo ">>> PBS_NODEFILE content:"
cat "\${PBS_NODEFILE}"
nvidia-smi
nvcc --version
date '+Start time: %Y-%m-%d %H:%M:%S'
export PYTHONPATH=${REPO_ROOT}/src:${REPO_ROOT}

${REPO_ROOT}/.venv/bin/python compare_hmc.py \\
    --lattice_size ${lattice_size} \\
    --beta ${evaluate_beta} \\
    --n_configs ${n_configs} \\
    --n_thermalization ${n_thermalization} \\
    --n_steps ${n_steps} \\
    --step_size ${step_size} \\
    --n_tune_steps ${n_tune_steps} \\
    --max_lag ${max_lag} \\
    --rand_seed ${seed} \\
${tune_flag_line}    --device cuda

date '+End time: %Y-%m-%d %H:%M:%S'
EOF
    chmod +x "${script_path}"
done

printf 'Generated %d U(2) HMC scripts in %s\n' "${#generated_scripts[@]}" "${SCRIPT_DIR}/scripts"
if [[ "${GENERATE_ONLY:-0}" == "1" ]]; then
    exit 0
fi
for script_path in "${generated_scripts[@]}"; do
    qsub "${script_path}"
done
