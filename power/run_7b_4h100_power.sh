#!/usr/bin/env bash
set -xeuo pipefail

BASE_MODEL="Qwen/Qwen2.5-Math-7B"
DATASET="dapo17k"
DATE=$(date +%m%d%H) 
ALGS="DAPO-reproduce"

adv_estimator=grpo
loss_agg_mode="token-mean"

train_prompt_bsz=512
n_resp_per_prompt=16
train_prompt_mini_bsz=32

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

export N_GPUS=4
export PROJ_NAME='verl-4xH100'
export EXP_NAME="${ALGS}-${BASE_MODEL}-${DATASET}-${DATE}"
export CKPT_DIR="checkpoints/${PROJ_NAME}/${EXP_NAME}"
export TENSORBOARD_DIR=/workspace/tensorboard_logs/${EXP_NAME}
export RAY_ADDRESS='local'

export DATA_DIR="/workspace/data"
TRAIN_DATA="${DATA_DIR}/train_${DATASET}.parquet"
VAL_DATA="${DATA_DIR}/test_128.parquet"

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0
enable_filter_groups=True
filter_groups_metric=acc

sp_size=2
gen_tp=2
ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))

    # algorithm.adv_estimator=rloo_batch_std \
    # actor_rollout_ref.actor.policy_loss.loss_mode=power_grpo \
    # +actor_rollout_ref.actor.policy_loss.length_alpha=0.5 \
    # actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \

nohup python verl/power/gcs_checkpoint.py --watch $CKPT_DIR 2>&1 &

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 nohup python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=verl/power/reward.py \
    custom_reward_function.name=compute_score \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    trainer.logger=['console','tensorboard'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.total_training_steps=100 2>&1 &

    # reward_model.reward_manager=dapo \
    # +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    # +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    # +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    # +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    # +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
