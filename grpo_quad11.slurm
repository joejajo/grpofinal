#!/bin/bash -l
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=09:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.err
#SBATCH --export=NONE

# =============================================================================
#  GRPO Quadratic Roots — v11 SLURM script
#
#  Key changes vs v10 (grpo_quad10.slurm):
#  ─────────────────────────────────────────────────────────────────────────────
#  HYPERPARAMETER          v10         v11         REASON
#  ─────────────────────────────────────────────────────────────────────────────
#  max_steps               1200        1500        longer run to exploit better config
#  learning_rate           1e-6        2e-6        more aggressive lr with beta KL guard
#  warmup_steps            80          60          less warmup needed with beta>0
#  beta (KL penalty)       0.00        0.001       prevents catastrophic policy drift
#  num_generations         4           8           2x more samples per prompt -> better GRPO variance
#  grad_accum_steps        16          16          kept; effective=16, prompts/step=16/8=2
#  temperature             0.50        0.70        CRITICAL: diversity needed for reward signal
#  top_p                   0.90        0.95        slightly wider nucleus for exploration
#  lora_r                  16          32          more capacity for math computation
#  lora_alpha              32          64          keep 2:1 ratio with lora_r
#  lora_dropout            0.05        0.05        unchanged
#  vllm_gpu_mem            0.30        0.45        more headroom for larger generations batch
#  eval_every_steps        0 (off)     150         periodic eval is now ENABLED
#  eval_max_samples        0 (all)     200         cap eval to keep it fast
#  eval_batch_size         8           8           unchanged
#  min_think_chars         120         80          easier to earn reasoning bonus early on
#  reward_debug_every      25          20          slightly more frequent debug output
#  debug_show_n            6           8           show more completions per debug dump
#  time limit              7h          9h          extra wall time for 1500 steps + evals
#  ─────────────────────────────────────────────────────────────────────────────
#
#  Why these choices push accuracy:
#    1. temperature 0.70  — GRPO needs diverse rewards across the group to compute
#                           meaningful advantages. At 0.50 the model generates near-
#                           identical completions; the variance is too low to learn.
#    2. num_generations=8 — 8 samples per prompt gives a much better reward mean/std
#                           estimate for the advantage normalisation.
#    3. MLP LoRA (v11 py) — gate/up/down_proj additions let the model actually update
#                           the arithmetic computation layers, not just attention heads.
#    4. lora_r=32         — more low-rank directions for expressive adaptation.
#    5. beta=0.001        — tiny KL penalty anchors the policy; with beta=0 and a
#                           continuous reward, the policy tends to find degenerate
#                           short-cuts in the reward surface.
#    6. eval enabled       — you can now monitor exact accuracy during training and
#                           catch early saturation or collapse.
# =============================================================================

unset SLURM_EXPORT_ENV
set -eo pipefail

module purge
module load cuda/12.4.1
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate /home/woody/iwi7/iwi7107h/conda_envs/grpo_quad_v13/
set -u

BASE_DIR="/home/woody/iwi7/iwi7107h/GRPO_Quad"
TRAIN_PARQUET="${BASE_DIR}/Dataset/quad_diverse_train.parquet"   # 1510 rows, 10 prompt templates
TEST_PARQUET="${BASE_DIR}/Dataset/quad_diverse_eval.parquet"     # 400 rows, 4 held-out templates
MODEL_DIR="/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct/"
OUT_DIR="${BASE_DIR}/Output/result_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Point at the v11 script
SCRIPT="${BASE_DIR}/grpo_quad_train_v11.py"

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port 29500 \
  "${SCRIPT}" \
  --model_name              "${MODEL_DIR}" \
  --train_parquet           "${TRAIN_PARQUET}" \
  --test_parquet            "${TEST_PARQUET}" \
  --output_dir              "${OUT_DIR}" \
  --tb_log_dir              "${OUT_DIR}/tb" \
  \
  --max_steps               1500 \
  --learning_rate           2e-6 \
  --warmup_steps            60 \
  --seed                    42 \
  --beta                    0.001 \
  --num_generations         8 \
  --per_device_batch_size   1 \
  --grad_accum_steps        16 \
  --max_prompt_length       512 \
  --max_completion_length   1024 \
  \
  --eval_every_steps        150 \
  --eval_max_samples        0 \
  --eval_batch_size         8 \
  --eval_max_new_tokens     512 \
  \
  --disable_csv_export \
  --grpo_max_samples        0 \
  \
  --lora_r                  32 \
  --lora_alpha              64 \
  --lora_dropout            0.05 \
  \
  --temperature             0.70 \
  --top_p                   0.95 \
  \
  --use_vllm \
  --vllm_gpu_memory_utilization 0.45 \
  \
  --min_think_chars         80 \
  --reward_debug_every      20 \
  --debug_show_n            8 \
  --debug_print_equation \
  --debug_print_effective

echo "TensorBoard logdir: ${OUT_DIR}/tb"
echo "Job complete: ${SLURM_JOB_ID}"
