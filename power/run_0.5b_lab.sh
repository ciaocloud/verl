#!/usr/bin/env bash
set -xeuo pipefail

export N_GPUS=1
export BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export DATA_DIR="/workspace/data"
export PROJ_NAME='verl_power_grpo'
export EXP_NAME="GRPO-0.5B-simpleRL-1"
export TENSORBOARD_DIR=/workspace/tensorboard_logs/${EXP_NAME}
export RAY_ADDRESS='local'

TRAIN_DATA="${DATA_DIR}/train_simplerl.parquet"
VAL_DATA="${DATA_DIR}/tiny_val_100.parquet"
validation_data_dir="/testlog/val"
rollout_data_dir="/testlog/rollout"

python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=verl/power/reward.py \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    trainer.logger=['console','tensorboard'] \
    trainer.log_val_generations=3 \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.total_epochs=3 > log_${EXP_NAME}.txt
