# VERL run files for this quadratic-roots project

This folder contains all files needed to run GRPO with **VERL** for your current quadratic-roots case.

## Files

- `prepare_quadratic_verl_data.py`  
  Converts your existing parquet files (`equation,a,b,c,r1,r2,user_message`) into VERL RLHF parquet format.

- `quadratic_reward.py`  
  Custom reward function used by VERL (`reward.custom_reward_function`) to score `\boxed{r1, r2}` outputs.

- `run_verl_grpo_quadratic.sh`  
  Main launcher (local or inside container) that:
  1) prepares VERL-format parquet, then
  2) runs `python3 -m verl.trainer.main_ppo` with GRPO settings.

- `submit_verl_grpo_quadratic.slurm`  
  SLURM submission script for your cluster style (module + conda + single RTX3080).

## Quick start

From repository root (`/home/runner/work/grpofinal/grpofinal`):

```bash
chmod +x /home/runner/work/grpofinal/grpofinal/verl_run/run_verl_grpo_quadratic.sh
python /home/runner/work/grpofinal/grpofinal/verl_run/prepare_quadratic_verl_data.py \
  --train_in /home/runner/work/grpofinal/grpofinal/Dataset/quad_medhard_train.parquet \
  --val_in /home/runner/work/grpofinal/grpofinal/Dataset/quad_medhard_eval.parquet \
  --out_dir /home/runner/work/grpofinal/grpofinal/verl_run/data_quadratic_verl

bash /home/runner/work/grpofinal/grpofinal/verl_run/run_verl_grpo_quadratic.sh
```

## Running on your HPC via SLURM

Copy `verl_run/` to your HPC project directory (already consistent with your existing `/home/woody/.../GRPO_Quad` style), then:

```bash
sbatch /home/woody/iwi7/iwi7107h/GRPO_Quad/verl_run/submit_verl_grpo_quadratic.slurm
```

## Important notes

- These scripts assume VERL is already installed in your environment/container.
- If you use the direct VERL image, this setup is generally more stable than mixing extra packages in an old env.
- Override defaults with environment variables (`BASE_DIR`, `MODEL_PATH`, `TRAIN_IN`, `VAL_IN`, etc.).
