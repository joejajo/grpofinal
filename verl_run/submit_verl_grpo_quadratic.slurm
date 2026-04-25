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
module load apptainer

BASE_DIR="/home/woody/iwi7/iwi7107h/GRPO_Quad"
mkdir -p "${BASE_DIR}/Output/result_${SLURM_JOB_ID}"
cd "${BASE_DIR}"

export PROJECT_ROOT="${BASE_DIR}"
export OUTPUT_ROOT="${BASE_DIR}/Output/result_${SLURM_JOB_ID}"
export VERL_RUN_DIR="${BASE_DIR}/verl_run"

SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

export MODEL_PATH="/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct/"
export TENSORBOARD_DIR="${OUTPUT_ROOT}/tensorboard"

export PYTHONPATH="${VERL_RUN_DIR}:${PYTHONPATH:-}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

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
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export TMPDIR=/tmp
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

ulimit -n 65536 2>/dev/null || true

mkdir -p "${OUTPUT_ROOT}" "${TENSORBOARD_DIR}" "${RAY_TMPDIR}"

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found on host"; exit 1; }

apptainer exec \
  --nv \
  --bind "${BASE_DIR}:${BASE_DIR}" \
  --bind "/home/woody/iwi7/iwi7107h/models:/home/woody/iwi7/iwi7107h/models" \
  "${SIF}" \
  bash "${BASE_DIR}/verl_run/run_verl_grpo_quadratic.sh"
