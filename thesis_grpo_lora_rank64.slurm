#!/bin/bash -l
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --job-name=thesis_grpo_lora_rank64
#SBATCH --time=12:00:00
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
conda activate myenv
set -u

BASE_DIR="/home/woody/iwi7/iwi7107h/GRPO_Quad"
TRAIN_PARQUET="${BASE_DIR}/Dataset/quad_medhard_train.parquet"
TEST_PARQUET="${BASE_DIR}/Dataset/quad_medhard_eval.parquet"
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
PLOT_SCRIPT="${BASE_DIR}/plot_grpo_results.py"
PLOTS_DIR="${OUT_DIR}/plots"

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
  --per_device_batch_size   1 \
  --grad_accum_steps        24 \
  --max_prompt_length       512 \
  --max_completion_length   1024 \
  \
  --eval_every_steps        50 \
  --eval_max_samples        0 \
  --eval_batch_size         8 \
  --eval_max_new_tokens     512 \
  \
  --disable_csv_export \
  --jsonl_export_dir        "${OUT_DIR}/jsonl" \
  --jsonl_num_examples      48 \
  --jsonl_train_every_steps 100 \
  --grpo_max_samples        500 \
  \
  --lora_r                  64 \
  --lora_alpha              128 \
  --lora_dropout            0.05 \
  \
  --temperature             0.80 \
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

mkdir -p "${PLOTS_DIR}"
if [ -f "${PLOT_SCRIPT}" ]; then
  python "${PLOT_SCRIPT}" --exp_dir "${OUT_DIR}" --save_dir "${PLOTS_DIR}" || true
fi

echo "TensorBoard logdir: ${OUT_DIR}/tb"
echo "Plots directory: ${PLOTS_DIR}"
echo "Job complete: ${SLURM_JOB_ID}"
