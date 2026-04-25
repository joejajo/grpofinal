#!/bin/bash -l
#SBATCH --job-name=verl_quad_grpo
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail

module purge
module load cuda/12.4.1
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate myenv
set -u

BASE_DIR="/home/woody/iwi7/iwi7107h/GRPO_Quad"
mkdir -p "${BASE_DIR}/Output/result_${SLURM_JOB_ID}"
cd "${BASE_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Optional: point to local model path for faster startup.
export MODEL_PATH="/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct/"

bash "${BASE_DIR}/verl_run/run_verl_grpo_quadratic.sh"
