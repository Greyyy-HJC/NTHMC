# Define common parameters
LATTICE_SIZE=8
MIN_BETA=2.0
MAX_BETA=2.0
BETA_GAP=0.5
N_EPOCHS=8
BATCH_SIZE=2
N_SUBSETS=8
MODEL_TAG='base'
BASE_SAVE_TAG='base_train_b2.0_L8'
SEED=1029

torchrun --standalone --nproc_per_node=1 train.py \
    --lattice_size ${LATTICE_SIZE} --min_beta ${MIN_BETA} --max_beta ${MAX_BETA} --beta_gap ${BETA_GAP} \
    --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --n_subsets ${N_SUBSETS} \
    --model_tag ${MODEL_TAG} --save_tag "${BASE_SAVE_TAG}_${SEED}" --rand_seed ${SEED} --if_identity_init --if_check_jac
