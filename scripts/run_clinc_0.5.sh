#!/bin/bash

# Default parameters
DATASET="clinc"
KNOWN_RATIO=0.5
SEEDS=(2)
GPU_ID=0

NUM_WARMUP_EPOCHS=100
NUM_TRAIN_EPOCHS=40

NUM_HIGH_SAMPLE=8
NUM_MID_SAMPLE=12
NUM_LOW_SAMPLE=16

LR_ENCODER=1e-4
LR_PROJ=1e-3

RENAME_START_EPOCH=5
SAVE_PHASE2_MODEL_AT_EPOCHS=$RENAME_START_EPOCH

# Environment variables
export OPENBLAS_NUM_THREADS=24
export OMP_NUM_THREADS=24

# LLM configuration
export LLM_MODEL=""
export LLM_BASE_URL=""
export LLM_API_KEY=""

# Training loop
for SEED in "${SEEDS[@]}"; do
    python main.py \
        --bert_model ./pretrained_models/bert-base-uncased \
        --dataset $DATASET \
        --known_cls_ratio $KNOWN_RATIO \
        --cluster_num_factor 1.0 \
        --labeled_ratio 0.1 \
        --seed $SEED \
        --gpu_id $GPU_ID \
        --num_warmup_epochs ${NUM_WARMUP_EPOCHS} \
        --num_train_epochs ${NUM_TRAIN_EPOCHS} \
        --train_batch_size 48 \
        --eval_batch_size 48 \
        --lr_encoder $LR_ENCODER \
        --lr_proj $LR_PROJ \
        --temperature 0.07 \
        --save_results_path results/${DATASET}_${KNOWN_RATIO}_${SEED} \
        --use_known_labeled_data \
        --use_novel_clustered_data \
        --use_known_unlabeled_data \
        --hardneg_weight 0.5 \
        --hardneg_topk 5 \
        --use_mlp_projection \
        --mlp_hidden_dim 768 \
        --high_conf_ratio 0.15 \
        --low_conf_ratio 0.25 \
        --high_conf_ratio_end 0.4 \
        --num_high_sample $NUM_HIGH_SAMPLE \
        --num_mid_sample $NUM_MID_SAMPLE \
        --num_low_sample $NUM_LOW_SAMPLE \
        --label_refine_start_epoch $RENAME_START_EPOCH \
        --save_phase2_model_at_epochs $SAVE_PHASE2_MODEL_AT_EPOCHS
done
