#!/usr/bin/env bash
set -euo pipefail

# CPU-node JAX FT-HMC inference profiles for U(2) base, train/eval β=10.
# Sophia DGX A100: 2× AMD EPYC 7742 (128 physical cores / 256 SMT), ~1 TB RAM.
# PBS queue is by-node select=1. JAX uses physical-core intra-op threads, one CPU device.
# CNN weights are L-independent; L32/L64 fall back to the L16 checkpoint when needed.
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
read -r -a lattice_sizes <<< "${LATTICE_SIZES:-16 32}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
MODEL_DIR="${REPO_ROOT}/2du2/artifacts/models"
PROFILE_PY="${REPO_ROOT}/presentation/2du2_bottleneck/profile_inference.py"

if [[ ! "${model_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "model_tag must contain only letters, numbers, dots, underscores, or hyphens" >&2
    exit 1
fi

resolve_save_tag() {
    local lattice_size="$1"
    local native_tag="${model_tag}_train_b${train_beta}_L${lattice_size}_${seed}"
    local fallback_lattice="${WEIGHT_LATTICE_SIZE:-16}"
    local fallback_seed="${WEIGHT_SEED:-${seed}}"
    local fallback_tag="${model_tag}_train_b${train_beta}_L${fallback_lattice}_${fallback_seed}"
    if [[ -n "${SAVE_TAG:-}" ]]; then
        printf '%s\n' "${SAVE_TAG}"
        return
    fi
    if [[ -f "${MODEL_DIR}/best_model_train_beta${train_beta}_${native_tag}.npz" ]]; then
        printf '%s\n' "${native_tag}"
        return
    fi
    printf '%s\n' "${fallback_tag}"
}

mkdir -p "${SCRIPT_DIR}/scripts" "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/dumps" "${SCRIPT_DIR}/plots"
generated_scripts=()
for lattice_size in "${lattice_sizes[@]}"; do
    if (( lattice_size >= 64 )); then
        walltime="${WALLTIME:-24:00:00}"
    elif (( lattice_size >= 32 )); then
        walltime="${WALLTIME:-12:00:00}"
    else
        walltime="${WALLTIME:-04:00:00}"
    fi
    save_tag="$(resolve_save_tag "${lattice_size}")"
    checkpoint="${MODEL_DIR}/best_model_train_beta${train_beta}_${save_tag}.npz"
    if [[ ! -f "${checkpoint}" && "${GENERATE_ONLY:-0}" != "1" ]]; then
        echo "Missing U(2) checkpoint: ${checkpoint}" >&2
        exit 1
    fi
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Warning: generating script without the missing checkpoint ${checkpoint}" >&2
    fi
    run_tag="${model_tag}_L${lattice_size}_train_b${train_beta}_eval_b${evaluate_beta}_${seed}"
    script_path="${SCRIPT_DIR}/scripts/sub_profile_cpu_${run_tag}.sh"
    generated_scripts+=("${script_path}")
    job_beta="${evaluate_beta%%.*}"
    cat > "${script_path}" <<EOF
#!/bin/bash -l

#PBS -N u2_c${lattice_size}b${job_beta}_${seed}
#PBS -A fthmc
#PBS -l select=1
#PBS -l filesystems=home:eagle
#PBS -q by-node
#PBS -j oe
#PBS -l walltime=${walltime}
#PBS -o ${SCRIPT_DIR}/logs/profile_cpu_${run_tag}.log

set -euo pipefail
cd ${SCRIPT_DIR}
echo ">>> PBS_NODEFILE content:"
cat "\${PBS_NODEFILE}"
echo ">>> CPU topology:"
lscpu || true
echo ">>> nproc=\$(nproc)"
free -h || true
date '+Start time: %Y-%m-%d %H:%M:%S'
export PYTHONPATH=${REPO_ROOT}/src:${REPO_ROOT}
phys_cores=\$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' ')
if [[ -z "\${phys_cores}" || "\${phys_cores}" -lt 1 ]]; then
    phys_cores=\$(nproc)
fi
echo ">>> Using \${phys_cores} physical CPU threads for XLA/OpenMP"
export OMP_NUM_THREADS="\${phys_cores}"
export MKL_NUM_THREADS="\${phys_cores}"
export OPENBLAS_NUM_THREADS="\${phys_cores}"
export NUMEXPR_NUM_THREADS="\${phys_cores}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"

${REPO_ROOT}/.venv/bin/python ${PROFILE_PY} \\
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
    --output_dir ${SCRIPT_DIR} \\
    --device cpu \\
    --dump_suffix "${dump_suffix}"

date '+End time: %Y-%m-%d %H:%M:%S'
EOF
    chmod +x "${script_path}"
done

printf 'Generated %d U(2) CPU profile scripts in %s\n' "${#generated_scripts[@]}" "${SCRIPT_DIR}/scripts"
if [[ "${GENERATE_ONLY:-0}" == "1" ]]; then
    exit 0
fi
for script_path in "${generated_scripts[@]}"; do
    qsub "${script_path}"
done
