#!/bin/bash -l
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.err
#SBATCH --export=NONE

# =============================================================================
#  GRPO Quadratic Roots — v12 SLURM script
#
#  Key changes vs v11 (grpo_quad11.slurm):
#  ─────────────────────────────────────────────────────────────────────────────
#  HYPERPARAMETER          v11         v12         REASON
#  ─────────────────────────────────────────────────────────────────────────────
#  dataset                 quad_diverse quad_medhard  no easy zone; model saturated easy in v11
#  train rows              1510        2560          more medium+hard samples
#  eval rows               400         500           wider eval (|a| up to 5)
#  max_steps               1500        2000          harder problems need more steps
#  learning_rate           2e-6        1.5e-6        conservative; noisier gradients on hard data
#  warmup_steps            60          100           longer warmup for harder task
#  beta (KL penalty)       0.001       0.002         stronger anchor; harder task drifts more
#  num_generations         8           12            wider group -> better advantage estimation
#  per_device_batch_size   1           2             effective=24, prompts/step=24/12=2
#  grad_accum_steps        16          12            reduced 16→12 so 2*12=24 divisible by num_generations=12
#  temperature             0.70        0.80          harder problems need more generation diversity
#  top_p                   0.95        0.95          unchanged
#  lora_r                  32          48            more capacity for harder arithmetic
#  lora_alpha              64          96            keep 2:1 ratio with lora_r
#  lora_dropout            0.05        0.05          unchanged
#  vllm_gpu_mem            0.45        0.50          12 generations needs more vllm headroom
#  eval_every_steps        150         200           less frequent eval (2000 steps total)
#  eval_max_samples        200         300           larger eval sample with 500-row eval set
#  min_think_chars         80          80            unchanged
#  time limit              9h          12h           extra wall time for 2000 steps
#  ─────────────────────────────────────────────────────────────────────────────
#
#  Why these choices push accuracy beyond v11's 77.5% ceiling:
#    1. No easy samples  — 65% of v11 steps were zero-gradient (all-same reward on easy prompts).
#                          Removing easy zone means every step has useful signal.
#    2. num_generations=12 — better reward mean/std estimate per prompt for advantage normalisation.
#    3. temperature=0.80  — more diverse completions -> non-zero reward variance -> real gradients.
#    4. lora_r=48         — harder problems (larger a, larger roots) need more representational capacity.
#    5. beta=0.002        — harder problems cause faster policy drift; stronger KL keeps training stable.
# =============================================================================

unset SLURM_EXPORT_ENV
set -eo pipefail

module purge
module load cuda/12.4.1
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate myenv
set -u

BASE_DIR="/home/woody/iwi7/iwi7107h/GRPO_Quad"
TRAIN_PARQUET="${BASE_DIR}/Dataset/quad_medhard_train.parquet"   # 2560 rows, medium+hard only
TEST_PARQUET="${BASE_DIR}/Dataset/quad_medhard_eval.parquet"     # 500 rows, |a| up to 5, held-out templates
MODEL_DIR="/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct/"
OUT_DIR="${BASE_DIR}/Output/result_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SCRIPT="${BASE_DIR}/grpo_quad_train_v12.py"

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
  --max_steps               2000 \
  --learning_rate           1.5e-6 \
  --warmup_steps            100 \
  --seed                    42 \
  --beta                    0.002 \
  --num_generations         12 \
  --per_device_batch_size   2 \
  --grad_accum_steps        12 \
  --max_prompt_length       512 \
  --max_completion_length   1024 \
  \
  --eval_every_steps        200 \
  --eval_max_samples        300 \
  --eval_batch_size         8 \
  --eval_max_new_tokens     512 \
  \
  --disable_jsonl_export \
  --train_csv_path          "${OUT_DIR}/train_generations.csv" \
  --eval_csv_path           "${OUT_DIR}/eval_generations.csv" \
  --generation_csv_every_steps 50 \
  --grpo_max_samples        0 \
  \
  --lora_r                  48 \
  --lora_alpha              96 \
  --lora_dropout            0.05 \
  \
  --temperature             0.80 \
  --top_p                   0.95 \
  \
  --use_vllm \
  --vllm_gpu_memory_utilization 0.50 \
  \
  --min_think_chars         80 \
  --reward_debug_every      20 \
  --debug_show_n            8 \
  --debug_print_equation \
  --debug_print_effective

echo "TensorBoard logdir: ${OUT_DIR}/tb"
echo "Job complete: ${SLURM_JOB_ID}"
