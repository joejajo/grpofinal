#!/usr/bin/env bash
set -euo pipefail
set -x

# --- user-configurable env vars ---
BASE_DIR="${BASE_DIR:-/home/runner/work/grpofinal/grpofinal}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
VERL_DATA_DIR="${VERL_DATA_DIR:-${BASE_DIR}/verl_run/data_quadratic_verl}"
TRAIN_IN="${TRAIN_IN:-${BASE_DIR}/Dataset/quad_medhard_train.parquet}"
VAL_IN="${VAL_IN:-${BASE_DIR}/Dataset/quad_medhard_eval.parquet}"

PROJECT_NAME="${PROJECT_NAME:-verl_quadratic_roots}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_05b_grpo_v1}"

NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"

# --- prepare VERL-format parquet ---
python3 "${BASE_DIR}/verl_run/prepare_quadratic_verl_data.py" \
  --train_in "${TRAIN_IN}" \
  --val_in "${VAL_IN}" \
  --out_dir "${VERL_DATA_DIR}"

# --- train with VERL GRPO ---
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  trainer.val_before_train=True \
  data.train_files="${VERL_DATA_DIR}/train.parquet" \
  data.val_files="${VERL_DATA_DIR}/val.parquet" \
  data.train_batch_size=24 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='left' \
  data.shuffle=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1.5e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=24 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.002 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=12 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  reward.custom_reward_function.path="${BASE_DIR}/verl_run/quadratic_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.default_local_dir="${OUTPUT_ROOT:-${BASE_DIR}/Output}" \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.total_epochs=10 \
  "$@"
