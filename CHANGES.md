# Changelog

## 2026-04-25 — SLURM rank ablation suite + conda env + vLLM 0.18.x API fixes

### 1. Complete LoRA rank ablation SLURM suite

All four thesis experiment scripts are in place, covering the full LoRA rank ablation
study plus a full-parameter baseline:

| Script | LoRA config | Partition / GPU |
|---|---|---|
| `thesis_grpo_lora_rank16.slurm` | `--lora_r 16 --lora_alpha 32 --lora_dropout 0.05` | `rtx3080` |
| `thesis_grpo_lora_rank32.slurm` | `--lora_r 32 --lora_alpha 64 --lora_dropout 0.05` | `rtx3080` |
| `thesis_grpo_lora_rank64.slurm` | `--lora_r 64 --lora_alpha 128 --lora_dropout 0.05` | `rtx3080` |
| `thesis_grpo_fullweight.slurm` | `--disable_lora` (full weights) | **`a100`** |

Common hyperparameters across all four scripts:
- `max_steps=2000`, `warmup_steps=100`, `seed=42`, `beta=0.002`
- LR: `1.5e-6` (LoRA runs) / `1e-6` (full-weight run)
- `num_generations=12`, `per_device_batch_size=1`, `grad_accum_steps=24`
- `max_prompt_length=512`, `max_completion_length=1024`
- `temperature=0.80`, `top_p=0.95`
- `eval_every_steps=50`, `jsonl_num_examples=48`, `grpo_max_samples=500`
- Dataset: `quad_medhard_train.parquet` / `quad_medhard_eval.parquet`
- Model: `Qwen2.5-0.5B-Instruct`
- vLLM: `--use_vllm --vllm_gpu_memory_utilization 0.45`

---

### 2. Conda environment updated in all SLURM scripts

All 9 SLURM job scripts were updated to activate the correct conda environment.

**Affected files:**
- `thesis_grpo_lora_rank16.slurm`
- `thesis_grpo_lora_rank32.slurm`
- `thesis_grpo_lora_rank64.slurm`
- `thesis_grpo_fullweight.slurm`
- `grpo_quad10.slurm`
- `grpo_quad11.slurm`
- `grpo_quad12.slurm`
- `es_quad_1gpu.slurm`
- `es_quad_4gpu.slurm`

**Change:**
```diff
- conda activate myenv
+ conda activate /home/woody/iwi7/iwi7107h/conda_envs/grpo_quad_v13/
```

---

### 3. vLLM 0.18.x API compatibility fixes

#### `Esquadratics/utils/worker_extn.py`

The `StatelessProcessGroup` class moved between vLLM releases. Replaced the single
hard-coded import with a cascading fallback that tries all known locations:

```python
# Before
from vllm.distributed.utils import StatelessProcessGroup

# After — tries three known import paths before raising ImportError
def _import_stateless_pg():
    try:
        from vllm.distributed.utils import StatelessProcessGroup
        return StatelessProcessGroup
    except ImportError:
        pass
    try:
        from vllm.distributed.device_communicators.pynccl_wrapper import StatelessProcessGroup
        return StatelessProcessGroup
    except ImportError:
        pass
    try:
        from vllm.distributed.communication_op import StatelessProcessGroup
        return StatelessProcessGroup
    except ImportError:
        pass
    raise ImportError("Could not import StatelessProcessGroup from vllm.")
```

#### `Esquadratics/es_eval_quadratic.py`

Fixed `torch.load` call to include `weights_only=True` (required from PyTorch 2.0+;
suppresses FutureWarning in PyTorch 2.10.0 and is best practice for checkpoint loading):

```diff
- es_state = torch.load(args.weights_path, map_location="cpu")
+ es_state = torch.load(args.weights_path, map_location="cpu", weights_only=True)
```

#### `grpo_quad_train_v10.py`, `grpo_quad_train_v11.py`, `grpo_quad_train_v12.py`

Updated module docstring compatibility note:

```diff
- Compatible with: TRL 0.24.x + vLLM 0.10.2
+ Compatible with: TRL 0.24.x + vLLM 0.18.x
```

---

### 4. How to clone / push this repo to HPC

Run the following commands **on the HPC login node** (e.g. after SSH-ing in).

#### 4.1 Check / install git

```bash
# git is usually provided as a module on HPC systems
module load git          # try this first

# if not available as a module, install via conda
conda activate /home/woody/iwi7/iwi7107h/conda_envs/grpo_quad_v13/
conda install -y git

# verify
git --version
```

#### 4.2 One-time git identity setup (skip if already done)

```bash
git config --global user.name  "Your Name"
git config --global user.email "your@email.com"
```

#### 4.3 Clone the repo to HPC

```bash
cd /home/woody/iwi7/iwi7107h/
git clone https://github.com/joejajo/grpofinal.git GRPO_Quad
cd GRPO_Quad
```

If you use SSH keys (recommended for HPC):

```bash
git clone git@github.com:joejajo/grpofinal.git GRPO_Quad
```

#### 4.4 Pull latest changes (if repo already cloned)

```bash
cd /home/woody/iwi7/iwi7107h/GRPO_Quad
git fetch origin
git pull origin main        # or: git pull origin copilot/update-dependencies
```

#### 4.5 Push local changes back to GitHub from HPC

```bash
# stage and commit
git add -A
git commit -m "your message"

# push (uses stored credentials or SSH key)
git push origin main
```

> **Tip — GitHub Personal Access Token (PAT):**  
> If prompted for a password, use a GitHub PAT (not your account password).  
> Generate one at *GitHub → Settings → Developer settings → Personal access tokens*.  
> To avoid re-entering it each session, run:
> ```bash
> git config --global credential.helper store
> git push   # enter username + PAT once; stored in ~/.git-credentials
> ```

---

### Summary of files changed

| File | Change type |
|---|---|
| `thesis_grpo_lora_rank16.slurm` | conda env (already existed) |
| `thesis_grpo_lora_rank32.slurm` | new file |
| `thesis_grpo_lora_rank64.slurm` | conda env (already existed) |
| `thesis_grpo_fullweight.slurm` | conda env / A100 partition (already existed) |
| `grpo_quad10.slurm` | conda env |
| `grpo_quad11.slurm` | conda env |
| `grpo_quad12.slurm` | conda env |
| `es_quad_1gpu.slurm` | conda env |
| `es_quad_4gpu.slurm` | conda env |
| `grpo_quad_train_v10.py` | docstring |
| `grpo_quad_train_v11.py` | docstring |
| `grpo_quad_train_v12.py` | docstring |
| `Esquadratics/utils/worker_extn.py` | API fix — robust import |
| `Esquadratics/es_eval_quadratic.py` | API fix — `torch.load` |
| `CHANGES.md` | **this file** |
