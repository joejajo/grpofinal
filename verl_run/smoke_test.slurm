#!/bin/bash -l
#SBATCH --job-name=verl_smoke
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/smoke_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/GRPO_Quad/Output/smoke_%j.err
#SBATCH --export=NONE

# Smoke test for the verl Apptainer pipeline.
# Verifies: image loads, CUDA/NCCL/cuBLAS preload paths exist, verl imports,
# vllm imports, dataset prep produces correct schema, reward function returns
# the expected score, GPU is visible.

unset SLURM_EXPORT_ENV
set -euo pipefail


export BASE_DIR="/home/woody/iwi7/iwi7107h/GRPO_Quad"
export VERL_RUN_DIR="${BASE_DIR}/verl_run"
TRAIN_PARQUET="${BASE_DIR}/Dataset/quad_medhard_train.parquet"
VAL_PARQUET="${BASE_DIR}/Dataset/quad_medhard_eval.parquet"
TMP_OUT="/tmp/verl_smoke_${SLURM_JOB_ID}"
SIF="/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif"

mkdir -p "${TMP_OUT}"

# ── Pre-flight diagnostics ───────────────────────────────────────────────────
apptainer --version
echo "===== Job info ====="
echo "SLURM_JOB_ID:        ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST:  ${SLURM_JOB_NODELIST:-<not set>}"
echo "SLURM_GPUS_ON_NODE:  ${SLURM_GPUS_ON_NODE:-<not set>}"
echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "SIF image:           ${SIF}"
echo "===================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "===================="

# ── CUPTI fix: symlink host nsight-systems libcupti to expected name ─────────
mkdir -p /tmp/joel_cupti_fix
ln -sf /opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/libcupti.so.12.9 \
       /tmp/joel_cupti_fix/libcupti.so.12

# ── Apptainer environment (passed in via APPTAINERENV_ prefix under --cleanenv)
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH="${BASE_DIR}/verl_src:${VERL_RUN_DIR}"
export APPTAINERENV_LD_LIBRARY_PATH="/tmp/joel_cupti_fix:/.singularity.d/libs"

# Same LD_PRELOAD wiring as the real slurm scripts — we test it here.
_NV="/usr/local/lib/python3.12/dist-packages/nvidia"
export APPTAINERENV_LD_PRELOAD="\
${_NV}/cuda_cupti/lib/libcupti.so.12:\
${_NV}/nccl/lib/libnccl.so.2:\
${_NV}/cublas/lib/libcublas.so.12:\
${_NV}/cublas/lib/libcublasLt.so.12"

apptainer exec --cleanenv --nv \
  --bind "${BASE_DIR}:${BASE_DIR}" \
  --bind "/home/woody/iwi7/iwi7107h/models:/home/woody/iwi7/iwi7107h/models" \
  --bind /tmp/joel_cupti_fix:/tmp/joel_cupti_fix \
  "${SIF}" \
  bash -lc "
    set -e
    export PYTHONNOUSERSITE=1
    export PYTHONPATH='${BASE_DIR}/verl_src:${VERL_RUN_DIR}'
    export LD_LIBRARY_PATH='/tmp/joel_cupti_fix:/.singularity.d/libs'

    echo '════════ 1. Python & key versions ════════'
    python3 --version
    python3 -c 'import sys; print(\"sys.executable:\", sys.executable)'

    echo '════════ 2. LD_PRELOAD path existence ════════'
    for p in \$(echo \$LD_PRELOAD | tr ':' '\n'); do
      if [ -f \"\$p\" ]; then echo \"OK   \$p\"; else echo \"MISS \$p\"; fi
    done

    echo '════════ 3. Imports ════════'
    python3 -c 'import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.version.cuda)'
    python3 -c 'import vllm; print(\"vllm:\", vllm.__version__)'
    python3 -c 'import verl; print(\"verl:\", getattr(verl, \"__version__\", \"unknown\"))'
    python3 -c 'import torch; print(\"GPU visible:\", torch.cuda.is_available(), \"count:\", torch.cuda.device_count())'

    echo '════════ 4. verl reward function entry-point key ════════'
    # Recent verl uses top-level 'custom_reward_function'; older uses 'reward.custom_reward_function'.
    python3 -c \"
import verl.trainer.main_ppo as m
import inspect, os
src = inspect.getsourcefile(m)
print('main_ppo source:', src)
with open(src) as f: code = f.read()
top  = 'custom_reward_function' in code and 'config.custom_reward_function' in code
nest = 'config.reward.custom_reward_function' in code or 'reward.custom_reward_function' in code
print('top-level path used in code:', top)
print('nested path used in code   :', nest)
\"

    echo '════════ 5. Dataset prep produces expected schema ════════'
    python3 '${VERL_RUN_DIR}/prepare_quadratic_verl_data.py' \
      --train_in '${TRAIN_PARQUET}' \
      --val_in   '${VAL_PARQUET}'   \
      --out_dir  '${TMP_OUT}'

    python3 -c \"
import datasets
d = datasets.load_dataset('parquet', data_files={'t': '${TMP_OUT}/train.parquet'})['t']
print('rows:', len(d))
print('columns:', d.column_names)
assert set(d.column_names) >= {'data_source','prompt','reward_model','extra_info'}, d.column_names
ex = d[0]
print('data_source:', ex['data_source'])
print('prompt[:1]:', ex['prompt'][:1])
print('reward_model:', ex['reward_model'])
\"

    echo '════════ 6. Reward function returns expected scores ════════'
    python3 -c \"
import sys
sys.path.insert(0, '${VERL_RUN_DIR}')
from quadratic_reward import compute_score
# (a, b, c) = (1, -5, 6)  =>  x^2 - 5x + 6 = (x-2)(x-3)  =>  roots {2, 3}
gt = {'r1': 2, 'r2': 3}
extra = {'a': 1, 'b': -5, 'c': 6, 'r1': 2, 'r2': 3}
# Long reasoning trigger keywords for full math+format+reasoning bonus.
reasoning = 'using the quadratic formula and discriminant b^2-4ac to factor the equation, I get '
cases = [
  # (name, completion, expected_lower_bound, expected_upper_bound)
  ('clean exact + reason',    reasoning + r'\boxed{2, 3}',  0.95, 1.00),
  ('clean exact (swapped)',   reasoning + r'\boxed{3, 2}',  0.95, 1.00),
  ('boxed only (no reason)',  r'\boxed{2, 3}',              0.92, 1.00),
  ('one root',                reasoning + r'\boxed{2, 9}',  0.30, 0.55),
  ('wrong roots',             reasoning + r'\boxed{7, 9}', -0.01, 0.20),
  ('unparseable',             r'no answer here',           -0.01, 0.01),
  ('multiple boxed pairs',    r'\boxed{2,3} extra \boxed{2,3}', -0.01, 0.70),
  ('trailing junk after box', r'\boxed{2, 3} foo bar',      0.00, 0.70),
]
ok = True
for name, sol, lo, hi in cases:
  got = compute_score('quadratic/roots', sol, gt, extra)
  in_range = (lo <= got <= hi)
  flag = 'OK' if in_range else 'FAIL'
  print(f'{flag}  {name:25s} got={got:.4f} expected in [{lo}, {hi}]')
  ok = ok and in_range
sys.exit(0 if ok else 1)
\"

    echo '════════ 7. Tiny verl dry-run (1 step, no train) ════════'
    # Try TOP-LEVEL custom_reward_function first (recent verl).
    if python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        trainer.val_before_train=False \
        data.train_files='${TMP_OUT}/train.parquet' \
        data.val_files='${TMP_OUT}/val.parquet' \
        data.train_batch_size=2 \
        data.max_prompt_length=512 \
        data.max_response_length=64 \
        actor_rollout_ref.model.path='/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct/' \
        actor_rollout_ref.actor.ppo_mini_batch_size=2 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
        actor_rollout_ref.rollout.n=2 \
        custom_reward_function.path='${VERL_RUN_DIR}/quadratic_reward.py' \
        custom_reward_function.name=compute_score \
        trainer.logger='[\"console\"]' \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=1 \
        trainer.default_local_dir='${TMP_OUT}/ckpt' \
        2>&1 | tail -40 \
      ; then
        echo 'TOP-LEVEL custom_reward_function = OK'
    else
        echo 'TOP-LEVEL custom_reward_function FAILED — will try nested form:'
        python3 -m verl.trainer.main_ppo \
          algorithm.adv_estimator=grpo \
          trainer.val_before_train=False \
          data.train_files='${TMP_OUT}/train.parquet' \
          data.val_files='${TMP_OUT}/val.parquet' \
          data.train_batch_size=2 \
          data.max_prompt_length=512 \
          data.max_response_length=64 \
          actor_rollout_ref.model.path='/home/woody/iwi7/iwi7107h/models/Qwen2.5-0.5B-Instruct/' \
          actor_rollout_ref.actor.ppo_mini_batch_size=2 \
          actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
          actor_rollout_ref.rollout.name=vllm \
          actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
          actor_rollout_ref.rollout.n=2 \
          reward.custom_reward_function.path='${VERL_RUN_DIR}/quadratic_reward.py' \
          reward.custom_reward_function.name=compute_score \
          trainer.logger='[\"console\"]' \
          trainer.n_gpus_per_node=1 \
          trainer.nnodes=1 \
          trainer.total_epochs=1 \
          trainer.total_training_steps=1 \
          trainer.default_local_dir='${TMP_OUT}/ckpt' \
          2>&1 | tail -40
        echo 'NESTED reward.custom_reward_function tried'
    fi

    echo '════════ smoke test complete ════════'
  "
