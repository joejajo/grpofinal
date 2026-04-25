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

export PROJECT_ROOT="${BASE_DIR}"
export OUTPUT_ROOT="${BASE_DIR}/Output/result_${SLURM_JOB_ID}"
export ONESHOT_DIR="${PROJECT_ROOT}"

export WANDB_MODE=disabled
export TENSORBOARD_DIR="${OUTPUT_ROOT}/tensorboard"

SDPO_DIR="${PROJECT_ROOT}/SDPO"
PKGS_DIR="${PROJECT_ROOT}/pkgs"
export PYTHONPATH="${ONESHOT_DIR}:${SDPO_DIR}:${PKGS_DIR}:${PYTHONPATH:-}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# NOTE: These paths are intentionally pinned for the current container/runtime
# stack (Python 3.12 + CUDA 12.x NVIDIA wheels on this cluster image).
_NV="/usr/local/lib/python3.12/dist-packages/nvidia"
export APPTAINERENV_LD_PRELOAD="\
${_NV}/cuda_cupti/lib/libcupti.so.12:\
${_NV}/nccl/lib/libnccl.so.2:\
${_NV}/cublas/lib/libcublas.so.12:\
${_NV}/cublas/lib/libcublasLt.so.12"

unset VLLM_ATTENTION_BACKEND
unset ROCR_VISIBLE_DEVICES
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export APPTAINERENV_UV_THREADPOOL_SIZE=4
export PYTHONWARNINGS="ignore:Unsupported processor type"
export TMPDIR=/tmp
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

mkdir -p "${OUTPUT_ROOT}" "${TENSORBOARD_DIR}" "${RAY_TMPDIR}"

# Optional: point to local model path for faster startup.
export MODEL_PATH="/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct/"

bash "${BASE_DIR}/verl_run/run_verl_grpo_quadratic.sh"
