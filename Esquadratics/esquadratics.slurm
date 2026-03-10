#!/bin/bash -l
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:4
#SBATCH --time=12:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/Esquadratics/outputs/result_%j/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/Esquadratics/outputs/result_%j/%x_%j.err
#SBATCH --export=NONE

# =============================================================================
#  ES Quadratic Roots — 4-GPU SLURM script
#
#  4 vLLM engines across 4 RTX 3080s (full accelerated setup).
#  Seeds are distributed round-robin across engines for parallelism.
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

BASE_DIR="/home/woody/iwi7/iwi7107h/Esquadratics"
DATA_PARQUET="${BASE_DIR}/Dataset/quad_medhard_train.parquet"
EVAL_PARQUET="${BASE_DIR}/Dataset/quad_medhard_eval.parquet"
MODEL_DIR="/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct/"
OUT_DIR="${BASE_DIR}/outputs/result_${SLURM_JOB_ID}"
NUM_ITERS=500
mkdir -p "${OUT_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# cd into project dir so vLLM can resolve "utils.worker_extn.WorkerExtension"
cd "${BASE_DIR}"

SCRIPT="${BASE_DIR}/es_finetuning_quadratic_accl.py"

python "${SCRIPT}" \
  --model_name              "${MODEL_DIR}" \
  --cuda_devices            0,1,2,3 \
  --num_engines             4 \
  --population_size         30 \
  --num_iterations          ${NUM_ITERS} \
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

# =============================================================================
#  Post-training evaluation on held-out eval set
# =============================================================================
echo ""
echo "=========================================="
echo "Starting post-training evaluation..."
echo "=========================================="

# Find the final model checkpoint (the training script saves it here)
ES_EXP_DIR=$(find "${OUT_DIR}/es_experiment" -maxdepth 1 -type d -name "quadratic_nccl_*" | head -1)
FINAL_WEIGHTS="${ES_EXP_DIR}/model_saves/final_model_iteration_${NUM_ITERS}/pytorch_model.pth"

if [ -f "${FINAL_WEIGHTS}" ]; then
    EVAL_SCRIPT="${BASE_DIR}/es_eval_quadratic.py"
    EVAL_OUT="${OUT_DIR}/eval_results"
    mkdir -p "${EVAL_OUT}"

    python "${EVAL_SCRIPT}" \
      --model_dir      "${MODEL_DIR}" \
      --weights_path   "${FINAL_WEIGHTS}" \
      --eval_data      "${EVAL_PARQUET}" \
      --out_dir        "${EVAL_OUT}" \
      --max_tokens     1024 \
      --dtype          float16 \
      --tb_logdir      "${ES_EXP_DIR}"

    echo "Eval results: ${EVAL_OUT}"
else
    echo "WARNING: Final weights not found at ${FINAL_WEIGHTS}"
    echo "Skipping post-training evaluation."
fi

echo "Job complete: ${SLURM_JOB_ID}"
