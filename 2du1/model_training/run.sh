# Define common parameters
LATTICE_SIZE=16
MIN_BETA=1.0
MAX_BETA=1.0
BETA_GAP=0.5
N_EPOCHS=8
BATCH_SIZE=64
N_SUBSETS=8
N_WORKERS=0
MODEL_TAG='base'
BASE_SAVE_TAG='base_train_b1.0_L16'
SEED=1029

"../../.venv/bin/python" train.py \
    --lattice_size ${LATTICE_SIZE} --min_beta ${MIN_BETA} --max_beta ${MAX_BETA} --beta_gap ${BETA_GAP} \
    --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --n_subsets ${N_SUBSETS} --n_workers ${N_WORKERS} \
    --model_tag ${MODEL_TAG} --save_tag "${BASE_SAVE_TAG}_${SEED}" --rand_seed ${SEED} --device gpu
