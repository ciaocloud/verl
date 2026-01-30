#!/usr/bin/env bash
set -xeuo pipefail

# Experiment config
MODEL_SIZE="0.5B"
DATASET="gsm8k"
DATE=$(date +%m%d)  # MMDDHH format (e.g., 012014)

export GOOGLE_APPLICATION_CREDENTIALS=/wx-gcs-key.json 
export WANDB_API_KEY=$(cat /workspace/wx-wandb-api-key.txt)
export N_GPUS=1
export BASE_MODEL="Qwen/Qwen2.5-${MODEL_SIZE}-Instruct"
export DATA_DIR="/workspace/data"
export PROJ_NAME='verl-lotis'
export EXP_NAME="LOTIS-${MODEL_SIZE}-${DATASET}-${DATE}"

# Use verl's default checkpoint path
export CKPT_DIR="checkpoints/${PROJ_NAME}/${EXP_NAME}"
export TENSORBOARD_DIR=/workspace/tensorboard_logs/${EXP_NAME}
export RAY_ADDRESS='local'

# TRAIN_DATA="${DATA_DIR}/train_${DATASET}.parquet"
# VAL_DATA="${DATA_DIR}/tiny_val_100.parquet"
TRAIN_DATA="${DATA_DIR}/gsm8k/train.parquet"
VAL_DATA="${DATA_DIR}/gsm8k/test.parquet"

# Start async GCS checkpoint uploader in background
nohup python verl/lab/gcs_checkpoint.py --watch $CKPT_DIR 2>&1 &

nohup python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=verl/lab/reward.py \
    custom_reward_function.name=compute_score \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.lotis.length_weight.enable=True \
    actor_rollout_ref.actor.lotis.length_weight.num_rbf_kernels=5 \
    actor_rollout_ref.actor.lotis.length_weight.alpha_lr=0.01 \
    actor_rollout_ref.actor.lotis.tis_weight.enable=True \
    actor_rollout_ref.actor.lotis.tis_weight.beta_init=1.0 \
    actor_rollout_ref.actor.lotis.tis_weight.beta_lr=0.01 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.logger=['console','tensorboard','wandb'] \
    trainer.log_val_generations=1 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=10 \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.total_epochs=3  2>&1 &
