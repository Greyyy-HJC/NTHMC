#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/.venv/bin/python"}

LATTICE_SIZES_RAW=${LATTICE_SIZES:-"8 16"}
BETAS_RAW=${BETAS:-"4.0 5.0 6.0 7.0 8.0"}
SEEDS_RAW=${SEEDS:-"1029"}
read -r -a LATTICE_SIZES <<< "${LATTICE_SIZES_RAW}"
read -r -a BETAS <<< "${BETAS_RAW}"
read -r -a SEEDS <<< "${SEEDS_RAW}"

RUN_GAUGES=${RUN_GAUGES:-1}
RUN_TRAINING=${RUN_TRAINING:-1}
RUN_EVALS=${RUN_EVALS:-1}
RUN_HMC_EVALS=${RUN_HMC_EVALS:-1}
RUN_FTHMC_EVALS=${RUN_FTHMC_EVALS:-1}

TRAIN_BETA=${TRAIN_BETA:-"4.0"}
MODEL_TAG=${MODEL_TAG:-"base"}
DEVICE=${DEVICE:-"cuda"}

GAUGE_N_THERMALIZATION=${GAUGE_N_THERMALIZATION:-2000}
GAUGE_N_CONFIGS=${GAUGE_N_CONFIGS:-2048}
GAUGE_STORE_INTERVAL=${GAUGE_STORE_INTERVAL:-1}
GAUGE_N_STEPS=${GAUGE_N_STEPS:-10}
GAUGE_STEP_SIZE=${GAUGE_STEP_SIZE:-0.1}
GAUGE_N_TUNE_STEPS=${GAUGE_N_TUNE_STEPS:-1000}
GAUGE_NO_TUNE_STEP_SIZE=${GAUGE_NO_TUNE_STEP_SIZE:-0}
GAUGE_ACCEPT_RATE_MIN=${GAUGE_ACCEPT_RATE_MIN:-0.55}
GAUGE_ACCEPT_RATE_MAX=${GAUGE_ACCEPT_RATE_MAX:-0.90}
SKIP_EXISTING_GAUGES=${SKIP_EXISTING_GAUGES:-0}

TRAIN_N_EPOCHS=${TRAIN_N_EPOCHS:-16}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
TRAIN_N_SUBSETS=${TRAIN_N_SUBSETS:-8}
TRAIN_DATA_PARALLEL=${TRAIN_DATA_PARALLEL:-0}
SKIP_EXISTING_MODELS=${SKIP_EXISTING_MODELS:-0}

EVAL_N_THERMALIZATION=${EVAL_N_THERMALIZATION:-10}
EVAL_N_CONFIGS=${EVAL_N_CONFIGS:-2048}
EVAL_N_STEPS=${EVAL_N_STEPS:-10}
HMC_STEP_SIZE=${HMC_STEP_SIZE:-0.1}
HMC_STEP_SIZE_L8=${HMC_STEP_SIZE_L8:-0.25}
HMC_STEP_SIZE_L16=${HMC_STEP_SIZE_L16:-0.23}
HMC_STEP_SIZE_L8_BETA7_0=${HMC_STEP_SIZE_L8_BETA7_0:-0.20}
HMC_STEP_SIZE_L8_BETA8_0=${HMC_STEP_SIZE_L8_BETA8_0:-0.16}
HMC_STEP_SIZE_L16_BETA7_0=${HMC_STEP_SIZE_L16_BETA7_0:-0.18}
HMC_STEP_SIZE_L16_BETA8_0=${HMC_STEP_SIZE_L16_BETA8_0:-0.14}
FT_STEP_SIZE=${FT_STEP_SIZE:-0.1}
FT_STEP_SIZE_L8=${FT_STEP_SIZE_L8:-0.24}
FT_STEP_SIZE_L16=${FT_STEP_SIZE_L16:-0.225}
FT_STEP_SIZE_L8_BETA7_0=${FT_STEP_SIZE_L8_BETA7_0:-0.235}
FT_STEP_SIZE_L8_BETA8_0=${FT_STEP_SIZE_L8_BETA8_0:-0.19}
FT_STEP_SIZE_L16_BETA7_0=${FT_STEP_SIZE_L16_BETA7_0:-0.195}
FT_STEP_SIZE_L16_BETA8_0=${FT_STEP_SIZE_L16_BETA8_0:-0.17}
EVAL_N_TUNE_STEPS=${EVAL_N_TUNE_STEPS:-1000}
MAX_LAG=${MAX_LAG:-20}
EVAL_NO_TUNE_STEP_SIZE=${EVAL_NO_TUNE_STEP_SIZE:-1}
EVAL_ACCEPT_RATE_MIN=${EVAL_ACCEPT_RATE_MIN:-0.55}
EVAL_ACCEPT_RATE_MAX=${EVAL_ACCEPT_RATE_MAX:-0.95}
SKIP_EXISTING_EVALS=${SKIP_EXISTING_EVALS:-0}

GAUGE_NO_TUNE_ARGS=()
if [[ "${GAUGE_NO_TUNE_STEP_SIZE}" == "1" ]]; then
    GAUGE_NO_TUNE_ARGS=(--no_tune_step_size)
fi

EVAL_NO_TUNE_ARGS=()
if [[ "${EVAL_NO_TUNE_STEP_SIZE}" == "1" ]]; then
    EVAL_NO_TUNE_ARGS=(--no_tune_step_size)
fi

TRAIN_DATA_PARALLEL_ARGS=()
if [[ "${TRAIN_DATA_PARALLEL}" == "1" ]]; then
    TRAIN_DATA_PARALLEL_ARGS=(--data_parallel)
fi

gauge_config_path() {
    local lattice_size=$1
    echo "${REPO_ROOT}/2du2/configs/links_L${lattice_size}_beta${TRAIN_BETA}.npy"
}

gauge_accept_rate_path() {
    local lattice_size=$1
    echo "${REPO_ROOT}/2du2/gauge_generation/dumps/accept_rate_L${lattice_size}_beta${TRAIN_BETA}.csv"
}

best_model_path() {
    local lattice_size=$1
    local seed=$2
    echo "${REPO_ROOT}/2du2/artifacts/models/best_model_train_beta${TRAIN_BETA}_base_scaling_train_b${TRAIN_BETA}_L${lattice_size}_${seed}.npz"
}

hmc_topo_path() {
    local lattice_size=$1
    local beta=$2
    local seed=$3
    echo "${REPO_ROOT}/2du2/evaluation/hmc/dumps/topo_hmc_L${lattice_size}_beta${beta}_nsteps${EVAL_N_STEPS}_${seed}.csv"
}

hmc_accept_rate_path() {
    local lattice_size=$1
    local beta=$2
    local seed=$3
    echo "${REPO_ROOT}/2du2/evaluation/hmc/dumps/accept_rate_hmc_L${lattice_size}_beta${beta}_nsteps${EVAL_N_STEPS}_${seed}.csv"
}

fthmc_topo_path() {
    local lattice_size=$1
    local beta=$2
    local seed=$3
    local save_tag="base_scaling_train_b${TRAIN_BETA}_L${lattice_size}_${seed}"
    echo "${REPO_ROOT}/2du2/evaluation/base/dumps/topo_fthmc_L${lattice_size}_beta${beta}_nsteps${EVAL_N_STEPS}_${save_tag}.csv"
}

fthmc_accept_rate_path() {
    local lattice_size=$1
    local beta=$2
    local seed=$3
    local save_tag="base_scaling_train_b${TRAIN_BETA}_L${lattice_size}_${seed}"
    echo "${REPO_ROOT}/2du2/evaluation/base/dumps/accept_rate_fthmc_L${lattice_size}_beta${beta}_nsteps${EVAL_N_STEPS}_${save_tag}.csv"
}

hmc_step_size_for_lattice_beta() {
    local lattice_size=$1
    local beta=$2
    local beta_tag=${beta//./_}
    local beta_specific_var="HMC_STEP_SIZE_L${lattice_size}_BETA${beta_tag}"
    local beta_specific_value="${!beta_specific_var:-}"
    local lattice_specific_var="HMC_STEP_SIZE_L${lattice_size}"
    local lattice_specific_value="${!lattice_specific_var:-}"

    if [[ -n "${beta_specific_value}" ]]; then
        echo "${beta_specific_value}"
    elif [[ -n "${lattice_specific_value}" ]]; then
        echo "${lattice_specific_value}"
    else
        echo "${HMC_STEP_SIZE}"
    fi
}

fthmc_step_size_for_lattice_beta() {
    local lattice_size=$1
    local beta=$2
    local beta_tag=${beta//./_}
    local beta_specific_var="FT_STEP_SIZE_L${lattice_size}_BETA${beta_tag}"
    local beta_specific_value="${!beta_specific_var:-}"
    local lattice_specific_var="FT_STEP_SIZE_L${lattice_size}"
    local lattice_specific_value="${!lattice_specific_var:-}"

    if [[ -n "${beta_specific_value}" ]]; then
        echo "${beta_specific_value}"
    elif [[ -n "${lattice_specific_value}" ]]; then
        echo "${lattice_specific_value}"
    else
        echo "${FT_STEP_SIZE}"
    fi
}

has_enough_gauge_configs() {
    local config_path=$1
    "${PYTHON_BIN}" - "${config_path}" "${GAUGE_N_CONFIGS}" <<'PY'
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
required = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)

configs = np.load(path, mmap_mode="r")
raise SystemExit(0 if configs.shape[0] >= required else 1)
PY
}

has_accept_rate_in_range() {
    local accept_rate_path=$1
    local min_accept_rate=$2
    local max_accept_rate=$3
    "${PYTHON_BIN}" - "${accept_rate_path}" "${min_accept_rate}" "${max_accept_rate}" <<'PY'
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
min_accept_rate = float(sys.argv[2])
max_accept_rate = float(sys.argv[3])
if not path.exists():
    raise SystemExit(1)

accept_rate = float(np.atleast_1d(np.loadtxt(path))[0])
raise SystemExit(0 if min_accept_rate <= accept_rate <= max_accept_rate else 1)
PY
}

has_completed_training() {
    local model_path=$1
    "${PYTHON_BIN}" - "${model_path}" "${TRAIN_N_EPOCHS}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
required_epochs = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)

with np.load(path, allow_pickle=False) as data:
    metadata = json.loads(str(data["metadata_json"]))
epoch = metadata.get("epoch")
if epoch is None:
    raise SystemExit(1)

raise SystemExit(0 if int(epoch) >= required_epochs - 1 else 1)
PY
}

has_enough_eval_configs() {
    local topo_path=$1
    "${PYTHON_BIN}" - "${topo_path}" "${EVAL_N_CONFIGS}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
required = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)

with path.open() as handle:
    n_rows = sum(1 for line in handle if line.strip())

raise SystemExit(0 if n_rows >= required else 1)
PY
}

run_gauge_generation() {
    local lattice_size=$1
    local seed=$2

    (
        cd "${REPO_ROOT}/2du2/gauge_generation"
        "${PYTHON_BIN}" generate.py \
            --lattice_size "${lattice_size}" \
            --beta "${TRAIN_BETA}" \
            --n_thermalization "${GAUGE_N_THERMALIZATION}" \
            --store_interval "${GAUGE_STORE_INTERVAL}" \
            --n_configs "${GAUGE_N_CONFIGS}" \
            --n_steps "${GAUGE_N_STEPS}" \
            --step_size "${GAUGE_STEP_SIZE}" \
            --n_tune_steps "${GAUGE_N_TUNE_STEPS}" \
            --rand_seed "${seed}" \
            --device "${DEVICE}" \
            --max_lag "${MAX_LAG}" \
            "${GAUGE_NO_TUNE_ARGS[@]}"
    )
}

run_training() {
    local lattice_size=$1
    local seed=$2
    local save_tag="base_scaling_train_b${TRAIN_BETA}_L${lattice_size}_${seed}"

    (
        cd "${REPO_ROOT}/2du2/model_training"
        "${PYTHON_BIN}" train.py \
            --lattice_size "${lattice_size}" \
            --min_beta "${TRAIN_BETA}" \
            --max_beta "${TRAIN_BETA}" \
            --beta_gap 0.5 \
            --n_epochs "${TRAIN_N_EPOCHS}" \
            --batch_size "${TRAIN_BATCH_SIZE}" \
            --n_subsets "${TRAIN_N_SUBSETS}" \
            --model_tag "${MODEL_TAG}" \
            --save_tag "${save_tag}" \
            --rand_seed "${seed}" \
            --device "${DEVICE}" \
            "${TRAIN_DATA_PARALLEL_ARGS[@]}"
    )
}

run_hmc_evaluation() {
    local lattice_size=$1
    local beta=$2
    local seed=$3
    local hmc_step_size
    hmc_step_size=$(hmc_step_size_for_lattice_beta "${lattice_size}" "${beta}")

    (
        cd "${REPO_ROOT}/2du2/evaluation/hmc"
        "${PYTHON_BIN}" compare_hmc.py \
            --lattice_size "${lattice_size}" \
            --n_configs "${EVAL_N_CONFIGS}" \
            --beta "${beta}" \
            --n_thermalization "${EVAL_N_THERMALIZATION}" \
            --n_steps "${EVAL_N_STEPS}" \
            --step_size "${hmc_step_size}" \
            --n_tune_steps "${EVAL_N_TUNE_STEPS}" \
            --max_lag "${MAX_LAG}" \
            --rand_seed "${seed}" \
            --device "${DEVICE}" \
            "${EVAL_NO_TUNE_ARGS[@]}"
    )
}

run_fthmc_evaluation() {
    local lattice_size=$1
    local beta=$2
    local seed=$3
    local save_tag="base_scaling_train_b${TRAIN_BETA}_L${lattice_size}_${seed}"
    local ft_step_size
    ft_step_size=$(fthmc_step_size_for_lattice_beta "${lattice_size}" "${beta}")

    (
        cd "${REPO_ROOT}/2du2/evaluation/base"
        "${PYTHON_BIN}" compare_fthmc.py \
            --lattice_size "${lattice_size}" \
            --n_configs "${EVAL_N_CONFIGS}" \
            --beta "${beta}" \
            --train_beta "${TRAIN_BETA}" \
            --n_thermalization "${EVAL_N_THERMALIZATION}" \
            --n_steps "${EVAL_N_STEPS}" \
            --ft_step_size "${ft_step_size}" \
            --n_tune_steps "${EVAL_N_TUNE_STEPS}" \
            --max_lag "${MAX_LAG}" \
            --rand_seed "${seed}" \
            --model_tag "${MODEL_TAG}" \
            --save_tag "${save_tag}" \
            --device "${DEVICE}" \
            "${EVAL_NO_TUNE_ARGS[@]}"
    )
}

for lattice_size in "${LATTICE_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [[ "${RUN_GAUGES}" == "1" ]]; then
            config_path=$(gauge_config_path "${lattice_size}")
            accept_rate_path=$(gauge_accept_rate_path "${lattice_size}")
            if [[ "${SKIP_EXISTING_GAUGES}" == "1" ]] \
                && has_enough_gauge_configs "${config_path}" \
                && has_accept_rate_in_range "${accept_rate_path}" "${GAUGE_ACCEPT_RATE_MIN}" "${GAUGE_ACCEPT_RATE_MAX}"; then
                echo ">>> Reusing U(2) training gauges: ${config_path}"
            else
                echo ">>> Generating U(2) training gauges: L=${lattice_size}, beta=${TRAIN_BETA}, seed=${seed}"
                run_gauge_generation "${lattice_size}" "${seed}"
            fi
        fi

        if [[ "${RUN_TRAINING}" == "1" ]]; then
            model_path=$(best_model_path "${lattice_size}" "${seed}")
            if [[ "${SKIP_EXISTING_MODELS}" == "1" ]] && has_completed_training "${model_path}"; then
                echo ">>> Reusing trained U(2) base transform: ${model_path}"
            else
                echo ">>> Training U(2) base transform: L=${lattice_size}, beta=${TRAIN_BETA}, seed=${seed}"
                run_training "${lattice_size}" "${seed}"
            fi
        fi

        if [[ "${RUN_EVALS}" == "1" ]]; then
            for beta in "${BETAS[@]}"; do
                if [[ "${RUN_HMC_EVALS}" == "1" ]]; then
                    hmc_path=$(hmc_topo_path "${lattice_size}" "${beta}" "${seed}")
                    hmc_accept_rate_path=$(hmc_accept_rate_path "${lattice_size}" "${beta}" "${seed}")
                    if [[ "${SKIP_EXISTING_EVALS}" == "1" ]] \
                        && has_enough_eval_configs "${hmc_path}" \
                        && has_accept_rate_in_range "${hmc_accept_rate_path}" "${EVAL_ACCEPT_RATE_MIN}" "${EVAL_ACCEPT_RATE_MAX}"; then
                        echo ">>> Reusing standard U(2) HMC evaluation: ${hmc_path}"
                    else
                        echo ">>> Evaluating standard U(2) HMC: L=${lattice_size}, beta=${beta}, step_size=$(hmc_step_size_for_lattice_beta "${lattice_size}" "${beta}"), seed=${seed}"
                        run_hmc_evaluation "${lattice_size}" "${beta}" "${seed}"
                    fi
                fi

                if [[ "${RUN_FTHMC_EVALS}" == "1" ]]; then
                    fthmc_path=$(fthmc_topo_path "${lattice_size}" "${beta}" "${seed}")
                    fthmc_accept_rate_path=$(fthmc_accept_rate_path "${lattice_size}" "${beta}" "${seed}")
                    if [[ "${SKIP_EXISTING_EVALS}" == "1" ]] \
                        && has_enough_eval_configs "${fthmc_path}" \
                        && has_accept_rate_in_range "${fthmc_accept_rate_path}" "${EVAL_ACCEPT_RATE_MIN}" "${EVAL_ACCEPT_RATE_MAX}"; then
                        echo ">>> Reusing U(2) FT-HMC evaluation: ${fthmc_path}"
                    else
                        echo ">>> Evaluating U(2) FT-HMC: L=${lattice_size}, beta=${beta}, train_beta=${TRAIN_BETA}, step_size=$(fthmc_step_size_for_lattice_beta "${lattice_size}" "${beta}"), seed=${seed}"
                        run_fthmc_evaluation "${lattice_size}" "${beta}" "${seed}"
                    fi
                fi
            done
        fi
    done
done
