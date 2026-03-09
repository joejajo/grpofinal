#!/bin/bash -l
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/result_%j/%x_%j.err
#SBATCH --export=NONE

# =============================================================================
#  ES Quadratic Roots — 1-GPU SLURM script
#
#  Single vLLM engine on 1 RTX 3080.
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
DATA_PARQUET="${BASE_DIR}/Dataset/quad_medhard_train.parquet"
MODEL_DIR="/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct/"
OUT_DIR="${BASE_DIR}/Output/result_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SCRIPT="${BASE_DIR}/Esquadratics/es_finetuning_quadratic_accl.py"

python "${SCRIPT}" \
  --model_name              "${MODEL_DIR}" \
  --cuda_devices            0 \
  --num_engines             1 \
  --population_size         20 \
  --num_iterations          500 \
  --sigma                   0.001 \
  --alpha                   0.0005 \
  --data_sample             200 \
  --data_path               "${DATA_PARQUET}" \
  --experiment_dir          "${OUT_DIR}/es_experiment" \
  --max_tokens              1024 \
  --dtype                   float16 \
  --csv_every_iters         50 \
  --verbose

echo "ES experiment dir: ${OUT_DIR}/es_experiment"
echo "Job complete: ${SLURM_JOB_ID}"
