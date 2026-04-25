#!/bin/bash -l
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=07:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.err
#SBATCH --export=NONE

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
TRAIN_PARQUET="${BASE_DIR}/Dataset/quadratic_grpo_train.parquet"
TEST_PARQUET="${BASE_DIR}/Dataset/quadratic_grpo_test.parquet"
MODEL_DIR="/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct/"
OUT_DIR="${BASE_DIR}/Output/result_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SCRIPT="${BASE_DIR}/grpo_quad_train_v10.py"

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  --main_process_port 29500 \
  "${SCRIPT}" \
  --model_name              "${MODEL_DIR}" \
  --train_parquet           "${TRAIN_PARQUET}" \
  --test_parquet            "${TEST_PARQUET}" \
  --output_dir              "${OUT_DIR}" \
  --tb_log_dir              "${OUT_DIR}/tb" \
  \
  --max_steps               1200 \
  --learning_rate           1e-06 \
  --warmup_steps            80 \
  --seed                    42 \
  --beta                    0.00 \
  --num_generations         4 \
  --per_device_batch_size   1 \
  --grad_accum_steps        16 \
  --max_prompt_length       512 \
  --max_completion_length   1024 \
  --eval_every_steps        0 \
  --eval_max_samples        0 \
  --eval_batch_size         8 \
  --eval_max_new_tokens     1024 \
  --disable_csv_export \
  --grpo_max_samples        0 \
  \
  --lora_r                  16 \
  --lora_alpha              32 \
  --lora_dropout            0.05 \
  \
  --temperature             0.50 \
  --top_p                   0.90 \
  \
  --use_vllm \
  --vllm_gpu_memory_utilization 0.30 \
  \
  --min_think_chars         120 \
  --reward_debug_every      25 \
  --debug_show_n            6 \
  --debug_print_equation \
  --debug_print_effective

echo "TensorBoard logdir: ${OUT_DIR}/tb"
