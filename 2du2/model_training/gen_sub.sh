#!/usr/bin/env bash
set -euo pipefail

# Experiment settings. Keep train_beta in canonical float form because it is
# also used in ensemble, checkpoint, script, and log filenames.
lattice_size="${LATTICE_SIZE:-16}"
train_beta="${TRAIN_BETA:-10.0}"
beta_gap="${BETA_GAP:-0.5}"
n_epochs="${N_EPOCHS:-32}"
batch_size="${BATCH_SIZE:-64}"
n_subsets="${N_SUBSETS:-8}"
model_tag="${MODEL_TAG:-base}"
run_tag="${RUN_TAG:-${model_tag}}"
lr="${LR:-0.00025}"
max_grad_norm="${MAX_GRAD_NORM:-5.0}"
plateau_factor="${PLATEAU_FACTOR:-0.2}"
plateau_patience="${PLATEAU_PATIENCE:-0}"
early_stop_patience="${EARLY_STOP_PATIENCE:-4}"
mobility_weight="${MOBILITY_WEIGHT:-1.0}"
if_check_jac="${IF_CHECK_JAC:-false}"
walltime="${WALLTIME:-04:00:00}"
gpu_count="${GPUS:-4}"
pbs_host="${PBS_HOST:-}"
pbs_queue="${PBS_QUEUE:-by-gpu}"
read -r -a loss_weights <<< "${LOSS_WEIGHTS:-1.0 1.0 1.0 1.0}"
read -r -a seeds <<< "${SEEDS:-1029 1107 1331 1984 1999 2008 2017 2025}"

case "${if_check_jac}" in
    true) check_jac_flag_line="    --if_check_jac \\"$'\n' ;;
    false) check_jac_flag_line="" ;;
    *) echo "IF_CHECK_JAC must be true or false" >&2; exit 1 ;;
esac

case "${pbs_queue}" in
    by-gpu)
        select_spec="1:ngpus=${gpu_count}"
        ;;
    by-node)
        if [[ "${gpu_count}" -ne 8 ]]; then
            echo "PBS_QUEUE=by-node requires GPUS=8 because the full Sophia node is allocated" >&2
            exit 1
        fi
        select_spec="1"
        ;;
    *)
        echo "PBS_QUEUE must be by-gpu or by-node" >&2
        exit 1
        ;;
esac
if [[ -n "${pbs_host}" ]]; then
    if [[ ! "${pbs_host}" =~ ^sophia-gpu-[0-9]+$ ]]; then
        echo "PBS_HOST must be a Sophia GPU hostname" >&2
        exit 1
    fi
    if [[ "${pbs_queue}" == "by-node" ]]; then
        select_spec="1:host=${pbs_host}"
    else
        select_spec="1:host=${pbs_host}:ngpus=${gpu_count}"
    fi
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONFIG_PATH="${SCRIPT_DIR}/../configs/links_L${lattice_size}_beta${train_beta}.npy"

if [[ ! "${model_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "model_tag must contain only letters, numbers, dots, underscores, or hyphens" >&2
    exit 1
fi
if [[ ! "${run_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_TAG must contain only letters, numbers, dots, underscores, or hyphens" >&2
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
    save_tag="${run_tag}_train_b${train_beta}_L${lattice_size}_${seed}"
    script_path="${SCRIPT_DIR}/scripts/sub_train_${run_tag}_L${lattice_size}_b${train_beta}_${seed}.sh"
    generated_scripts+=("${script_path}")
    cat > "${script_path}" <<EOF
#!/bin/bash -l

#PBS -N u2_t${lattice_size}_${seed}
#PBS -A fthmc
#PBS -l select=${select_spec}
#PBS -l filesystems=home:eagle
#PBS -q ${pbs_queue}
#PBS -j oe
#PBS -l walltime=${walltime}
#PBS -o ${SCRIPT_DIR}/logs/train_${run_tag}_L${lattice_size}_b${train_beta}_${seed}.log

set -euo pipefail
cd ${SCRIPT_DIR}
echo ">>> PBS_NODEFILE content:"
cat "\${PBS_NODEFILE}"
nvidia-smi
visible_gpus=\$(nvidia-smi -L | wc -l)
if [[ "\${visible_gpus}" -ne ${gpu_count} ]]; then
    exec_host=\$(qstat -f "\${PBS_JOBID}" 2>/dev/null | awk '/^[[:space:]]*exec_host =/{print \$3; exit}' || true)
    assigned_json=\$(pbsnodes -av "\$(hostname -s)" 2>/dev/null | awk -F'= ' '/resources_available.assigned_gpus =/{print \$2; exit}' || true)
    job_number=\${PBS_JOBID%%.*}
    allocated_gpus=""
    if [[ -n "\${assigned_json}" ]]; then
        allocated_gpus=\$(ASSIGNED_GPUS_JSON="\${assigned_json}" PBS_JOB_NUMBER="\${job_number}" \
            ${REPO_ROOT}/.venv/bin/python -c 'import json, os; print(json.loads(os.environ["ASSIGNED_GPUS_JSON"])[os.environ["PBS_JOB_NUMBER"]])' \
            2>/dev/null || true)
    fi
    if [[ -z "\${allocated_gpus}" && "\${visible_gpus}" -eq 8 ]]; then
        chunk_slot=\$(sed -n 's#^[^/]*/\\([0-9][0-9]*\\)\\*[0-9][0-9]*\$#\\1#p' <<<"\${exec_host}")
        if [[ -n "\${chunk_slot}" ]]; then
            gpu_start=\$((chunk_slot * ${gpu_count}))
            if (( gpu_start + ${gpu_count} <= visible_gpus )); then
                allocated_gpus=\$(GPU_START="\${gpu_start}" GPU_COUNT=${gpu_count} \
                    ${REPO_ROOT}/.venv/bin/python -c 'import os; start=int(os.environ["GPU_START"]); count=int(os.environ["GPU_COUNT"]); print(",".join(str(start+i) for i in range(count)))')
            fi
        fi
    fi
    IFS=',' read -r -a allocated_ids <<< "\${allocated_gpus}"
    if [[ -n "\${allocated_gpus}" && "\${#allocated_ids[@]}" -eq ${gpu_count} ]]; then
        export CUDA_VISIBLE_DEVICES="\${allocated_gpus}"
        echo ">>> Device cgroup exposed \${visible_gpus} GPUs; exec_host=\${exec_host:-unknown}; using PBS-assigned CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}"
    else
        echo "Expected ${gpu_count} visible or PBS-assigned GPUs, found visible=\${visible_gpus} assigned=\${allocated_gpus:-none}; refusing an unsafe allocation" >&2
        exit 99
    fi
fi
torch_visible=\$(${REPO_ROOT}/.venv/bin/python -c 'import torch; print(torch.cuda.device_count())')
if [[ "\${torch_visible}" -ne ${gpu_count} ]]; then
    echo "Expected Torch to see ${gpu_count} GPUs, found \${torch_visible}; refusing an unsafe allocation" >&2
    exit 99
fi
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
${check_jac_flag_line}    --lr ${lr} \\
    --max_grad_norm ${max_grad_norm} \\
    --plateau_factor ${plateau_factor} \\
    --plateau_patience ${plateau_patience} \\
    --early_stop_patience ${early_stop_patience} \\
    --loss_weights ${loss_weights[*]} \\
    --mobility_weight ${mobility_weight} \\
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
