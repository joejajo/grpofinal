#!/bin/bash -l
#SBATCH --job-name=verl_quad_lora16
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail


# ── Paths ─────────────────────────────────────────────────────────────────────
export BASE_DIR="/home/woody/iwi7/iwi7107h/GRPO_Quad"
export OUTPUT_ROOT="${BASE_DIR}/Output/result_${SLURM_JOB_ID}"
export VERL_RUN_DIR="${BASE_DIR}/verl_run"

TRAIN_PARQUET="${BASE_DIR}/Dataset/quad_medhard_train.parquet"
VAL_PARQUET="${BASE_DIR}/Dataset/quad_medhard_eval.parquet"
VERL_DATA_DIR="${VERL_RUN_DIR}/data_quadratic_verl"

export MODEL_PATH="/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct"
SIF="/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif"

export TENSORBOARD_DIR="${OUTPUT_ROOT}/tensorboard"

# ── Pre-flight (host-side) ────────────────────────────────────────────────────
mkdir -p "${OUTPUT_ROOT}" "${TENSORBOARD_DIR}" "${VERL_DATA_DIR}" "/tmp/r${SLURM_JOB_ID}"
ulimit -n 65536 2>/dev/null || true

apptainer --version
echo "===== Job info ====="
echo "SLURM_JOB_ID:        ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST:  ${SLURM_JOB_NODELIST:-<not set>}"
echo "SLURM_GPUS_ON_NODE:  ${SLURM_GPUS_ON_NODE:-<not set>}"
echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "SIF image:           ${SIF}"
echo "OUTPUT_ROOT:         ${OUTPUT_ROOT}"
echo "===================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "===================="

# ── CUPTI fix: symlink host nsight-systems libcupti to expected name ─────────
mkdir -p /tmp/joel_cupti_fix
ln -sf /opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/libcupti.so.12.9 \
       /tmp/joel_cupti_fix/libcupti.so.12

# ── Apptainer environment (under --cleanenv only APPTAINERENV_* are passed in)
_NV="/usr/local/lib/python3.12/dist-packages/nvidia"
export APPTAINERENV_LD_PRELOAD="\
${_NV}/cuda_cupti/lib/libcupti.so.12:\
${_NV}/nccl/lib/libnccl.so.2:\
${_NV}/cublas/lib/libcublas.so.12:\
${_NV}/cublas/lib/libcublasLt.so.12"

export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH="${BASE_DIR}/verl_src:${VERL_RUN_DIR}"
export APPTAINERENV_LD_LIBRARY_PATH="/tmp/joel_cupti_fix:/.singularity.d/libs"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_VLLM_DISABLE_COMPILE_CACHE=1
export APPTAINERENV_UV_THREADPOOL_SIZE=4
export APPTAINERENV_PYTHONWARNINGS="ignore:Unsupported processor type"
export APPTAINERENV_WANDB_MODE=disabled
export APPTAINERENV_WANDB_DISABLED=true
export APPTAINERENV_TMPDIR=/tmp
export APPTAINERENV_RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}"
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_TF_ENABLE_ONEDNN_OPTS=0
export APPTAINERENV_NCCL_IB_DISABLE=1
export APPTAINERENV_NCCL_P2P_DISABLE=1
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export APPTAINERENV_RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found on host"; exit 1; }

# ── Run (LoRA rank=16, alpha=32) ──────────────────────────────────────────────
apptainer exec --cleanenv --nv \
  --bind "${BASE_DIR}:${BASE_DIR}" \
  --bind "/home/woody/iwi7/iwi7107h/models:/home/woody/iwi7/iwi7107h/models" \
  --bind /tmp/joel_cupti_fix:/tmp/joel_cupti_fix \
  "${SIF}" \
  bash -lc "
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    export PYTHONPATH='${BASE_DIR}/verl_src:${VERL_RUN_DIR}'
    export LD_LIBRARY_PATH='/tmp/joel_cupti_fix:/.singularity.d/libs'

    python3 '${VERL_RUN_DIR}/prepare_quadratic_verl_data.py' \
      --train_in '${TRAIN_PARQUET}' \
      --val_in   '${VAL_PARQUET}' \
      --out_dir  '${VERL_DATA_DIR}'

    # ── Background checkpoint manager: keep only the 2 most recent ───────────
    _CKPT_DIR='${OUTPUT_ROOT}'
    ( while true; do
        sleep 180
        _n=\$(ls -d \"\${_CKPT_DIR}\"/global_step_* 2>/dev/null | wc -l)
        if [ \"\${_n}\" -gt 2 ]; then
          ls -d \"\${_CKPT_DIR}\"/global_step_* 2>/dev/null \
            | sort -t_ -k3 -n \
            | head -n \$(( _n - 2 )) \
            | xargs -r rm -rf
          echo \"[ckpt] Deleted old checkpoints, kept 2 latest\"
        fi
      done ) &
    _CLEANUP_PID=\$!

    python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      trainer.val_before_train=True \
      data.train_files='${VERL_DATA_DIR}/train.parquet' \
      data.val_files='${VERL_DATA_DIR}/val.parquet' \
      data.train_batch_size=24 \
      data.max_prompt_length=512 \
      data.max_response_length=1024 \
      data.filter_overlong_prompts=True \
      data.shuffle=True \
      actor_rollout_ref.model.path='${MODEL_PATH}' \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.model.lora_rank=16 \
      actor_rollout_ref.model.lora_alpha=32 \
      actor_rollout_ref.model.target_modules=all-linear \
      actor_rollout_ref.actor.optim.lr=1.5e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=24 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=0.002 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
      actor_rollout_ref.rollout.n=12 \
      actor_rollout_ref.rollout.layered_summon=True \
      actor_rollout_ref.rollout.load_format=safetensors \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      algorithm.use_kl_in_reward=False \
      reward.custom_reward_function.path='${VERL_RUN_DIR}/quadratic_reward.py' \
      reward.custom_reward_function.name=compute_score \
      trainer.critic_warmup=0 \
      trainer.logger='[\"console\",\"tensorboard\"]' \
      trainer.project_name=verl_quadratic_roots \
      trainer.experiment_name=qwen25_05b_lora16 \
      trainer.n_gpus_per_node=1 \
      trainer.nnodes=1 \
      trainer.default_local_dir='${OUTPUT_ROOT}' \
      trainer.save_freq=50 \
      trainer.test_freq=50 \
      trainer.total_epochs=10 \
      trainer.max_actor_ckpt_to_keep=2

    # ── Stop background cleanup; list surviving checkpoints ───────────────────
    kill \"\${_CLEANUP_PID}\" 2>/dev/null || true
    echo \"=== Saved checkpoints (compare step numbers against TensorBoard val/test_score_mean) ===\"
    ls -d \"\${_CKPT_DIR}\"/global_step_* 2>/dev/null | sort -t_ -k3 -n || echo \"(none)\"
  "

echo "TensorBoard logdir: ${TENSORBOARD_DIR}"
echo "Job complete: ${SLURM_JOB_ID}"
