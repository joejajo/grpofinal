# Changelog

## 2026-04-25 — vLLM 0.18.x / TRL 0.24.0 API Fixes + conda env + LoRA rank ablation

### 1. Conda environment updated in all SLURM scripts

All 8 SLURM job scripts were updated to activate the correct conda environment.

**Affected files:**
- `thesis_grpo_lora_rank16.slurm`
- `thesis_grpo_lora_rank32.slurm` *(new — see below)*
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

### 2. New thesis SLURM script: LoRA rank 32

Added `thesis_grpo_lora_rank32.slurm` to complete the LoRA rank ablation study.
The thesis experiment set now covers all four training configurations:

| Script | LoRA config | GPU |
|---|---|---|
| `thesis_grpo_lora_rank16.slurm` | `--lora_r 16 --lora_alpha 32` | RTX 3080 |
| `thesis_grpo_lora_rank32.slurm` | `--lora_r 32 --lora_alpha 64` | RTX 3080 |
| `thesis_grpo_lora_rank64.slurm` | `--lora_r 64 --lora_alpha 128` | RTX 3080 |
| `thesis_grpo_fullweight.slurm` | `--disable_lora` (full weights) | A100 |

All four scripts share the same hyperparameters (v12 defaults):
- `max_steps=2000`, `learning_rate=1.5e-6`, `beta=0.002`
- `num_generations=12`, `per_device_batch_size=1`, `grad_accum_steps=24`
- `temperature=0.80`, `top_p=0.95`
- `eval_every_steps=50`, `jsonl_num_examples=48`
- Dataset: `quad_medhard_train.parquet` / `quad_medhard_eval.parquet`
- Model: `Qwen2.5-0.5B-Instruct`

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

### Summary of files changed

| File | Change type |
|---|---|
| `thesis_grpo_lora_rank32.slurm` | **New file** |
| `thesis_grpo_lora_rank16.slurm` | conda env |
| `thesis_grpo_lora_rank64.slurm` | conda env |
| `thesis_grpo_fullweight.slurm` | conda env |
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
