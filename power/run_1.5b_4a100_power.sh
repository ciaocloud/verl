#!/usr/bin/env bash
set -xeuo pipefail

MODEL_SIZE="Math-1.5B"
DATASET="dapo17k"
DATE=$(date +%m%d%H)  # MMDDHH format (e.g., 012015)
ALGS="power-rloo-batch"

export N_GPUS=4
export BASE_MODEL="Qwen/Qwen2.5-${MODEL_SIZE}"
# export BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-${MODEL_SIZE}"
export DATA_DIR="/workspace/data"
export PROJ_NAME='go-verl'
export EXP_NAME="${ALGS}-${MODEL_SIZE}-${DATASET}-${DATE}"
export CKPT_DIR="checkpoints/${PROJ_NAME}/${EXP_NAME}"

TRAIN_DATA="${DATA_DIR}/train_${DATASET}.parquet"
VAL_DATA="${DATA_DIR}/test_128.parquet"
# validation_data_dir="/workspace/testlog/val/${EXP_NAME}"
# rollout_data_dir="/workspace/testlog/rollout/${EXP_NAME}"

export TENSORBOARD_DIR=/workspace/tensorboard_logs/${EXP_NAME}
export RAY_ADDRESS='local'

# Start async GCS checkpoint uploader in background
nohup python verl/power/gcs_checkpoint.py --watch $CKPT_DIR 2>&1 &

nohup python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=verl/power/reward.py \
    custom_reward_function.name=compute_score \
    algorithm.adv_estimator=rloo_batch_std \
    actor_rollout_ref.actor.policy_loss.loss_mode=power_grpo \
    +actor_rollout_ref.actor.policy_loss.length_alpha=0.5 \
    algorithm.use_kl_in_reward=False \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=256 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.logger=['console','tensorboard'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.total_epochs=3 2>&1 &

echo "Training started in background."
echo "GCS upload logs: gcs_upload_${EXP_NAME}.log"
echo "Monitor logs with: tail -f exp_${EXP_NAME}.log"