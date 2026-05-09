#!/bin/bash -l
#SBATCH --job-name=thesis_pgrpo_lora16
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/GRPOQUADTHESIS/Output/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/GRPOQUADTHESIS/Output/%x_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -eo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate /home/woody/iwi7/iwi7107h/conda_envs/grpo_vllm011
set -u

# ── HF cache ──────────────────────────────────────────────────────────────────
export HF_HOME=/home/woody/iwi7/iwi7107h/hf_home
export HF_HUB_CACHE=/home/woody/iwi7/iwi7107h/hf_home/hub
export HF_DATASETS_CACHE=/home/woody/iwi7/iwi7107h/hf_home/datasets
unset TRANSFORMERS_CACHE

# ── FAU proxy ─────────────────────────────────────────────────────────────────
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# ── Verification ──────────────────────────────────────────────────────────────
echo "Python path: $(which python)"
python --version

python - <<'PY'
import torch, transformers, trl, vllm, accelerate, peft
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("transformers:", transformers.__version__)
print("trl:", trl.__version__)
print("vllm:", vllm.__version__)
print("accelerate:", accelerate.__version__)
print("peft:", peft.__version__)
PY

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR="/home/woody/iwi7/iwi7107h/GRPOQUADTHESIS"
TRAIN_PARQUET="${BASE_DIR}/Dataset/quad_medhard_train.parquet"
TEST_PARQUET="${BASE_DIR}/Dataset/quad_medhard_eval.parquet"
MODEL_DIR="/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct"
OUT_DIR="${BASE_DIR}/Output/result_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

SCRIPT="${BASE_DIR}/grpo_quad_train_v12.py"
PLOT_SCRIPT="${BASE_DIR}/plot_grpo_results.py"
PLOTS_DIR="${OUT_DIR}/plots"

# ── Runtime env ───────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# ── Training (pure GRPO, LoRA rank=16 — RTX 3080 conservative settings) ──────
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
  --learning_rate           1e-6 \
  --warmup_steps            100 \
  --seed                    42 \
  --beta                    0.002 \
  --num_generations         8 \
  --per_device_batch_size   1 \
  --grad_accum_steps        8 \
  --max_prompt_length       512 \
  --max_completion_length   512 \
  \
  --eval_every_steps        50 \
  --eval_max_samples        200 \
  --eval_batch_size         8 \
  --eval_max_new_tokens     512 \
  \
  --generation_csv_every_steps 0 \
  --jsonl_export_dir        "${OUT_DIR}/jsonl" \
  --jsonl_num_examples      48 \
  --jsonl_train_every_steps 100 \
  --grpo_max_samples        800 \
  \
  --loss_type               grpo \
  \
  --lora_r                  16 \
  --lora_alpha              32 \
  --lora_dropout            0.05 \
  \
  --temperature             0.80 \
  --top_p                   0.95 \
  \
  --use_vllm \
  --vllm_gpu_memory_utilization 0.35 \
  \
  --min_think_chars         80 \
  --reward_debug_every      20 \
  --debug_show_n            8 \
  --debug_print_equation \
  --debug_print_effective

# ── Post-training plots ───────────────────────────────────────────────────────
mkdir -p "${PLOTS_DIR}"
if [ -f "${PLOT_SCRIPT}" ]; then
  python "${PLOT_SCRIPT}" --exp_dir "${OUT_DIR}" --save_dir "${PLOTS_DIR}" || true
fi

echo "TensorBoard logdir: ${OUT_DIR}/tb"
echo "Plots directory:    ${PLOTS_DIR}"
echo "Job complete:       ${SLURM_JOB_ID}"
