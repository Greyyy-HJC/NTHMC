#!/usr/bin/env bash
set -euo pipefail

# Experiment settings. Keep beta in canonical float form (for example, "3.0")
# because it is also used in generated filenames.
lattice_size=16
beta="3.0"
n_thermalization=2000
store_interval=20
n_configs=4096
n_steps=10
step_size="0.30"
n_tune_steps=2000
max_lag=64
rand_seed=1331
tune_step_size=true

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

case "${tune_step_size}" in
    true) tune_flag_line="" ;;
    false) tune_flag_line="    --no_tune_step_size \\"$'\n' ;;
    *) echo "tune_step_size must be true or false" >&2; exit 1 ;;
esac

mkdir -p "${SCRIPT_DIR}/scripts" "${SCRIPT_DIR}/logs"
script_path="${SCRIPT_DIR}/scripts/sub_gauge_L${lattice_size}_b${beta}_${rand_seed}.sh"

cat > "${script_path}" <<EOF
#!/bin/bash -l

#PBS -N u1_g${lattice_size}_${rand_seed}
#PBS -A fthmc
#PBS -l select=1:ngpus=1
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=02:00:00
#PBS -o ${SCRIPT_DIR}/logs/gauge_L${lattice_size}_b${beta}_${rand_seed}.log

set -euo pipefail
cd ${SCRIPT_DIR}
echo ">>> PBS_NODEFILE content:"
cat "\${PBS_NODEFILE}"
nvidia-smi
nvcc --version
date '+Start time: %Y-%m-%d %H:%M:%S'
export PYTHONPATH=${REPO_ROOT}/src:${REPO_ROOT}

${REPO_ROOT}/.venv/bin/python generate.py \\
    --lattice_size ${lattice_size} \\
    --beta ${beta} \\
    --n_thermalization ${n_thermalization} \\
    --store_interval ${store_interval} \\
    --n_configs ${n_configs} \\
    --n_steps ${n_steps} \\
    --step_size ${step_size} \\
    --n_tune_steps ${n_tune_steps} \\
    --max_lag ${max_lag} \\
    --rand_seed ${rand_seed} \\
${tune_flag_line}    --device cuda

date '+End time: %Y-%m-%d %H:%M:%S'
EOF
chmod +x "${script_path}"

printf 'Generated U(1) gauge script: %s\n' "${script_path}"
if [[ "${GENERATE_ONLY:-0}" == "1" ]]; then
    exit 0
fi
qsub "${script_path}"
